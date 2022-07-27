function [Xref_mat,Xk_mat,P_mat,Pbk_mat,resids] = ls_ekf(obs_data,intfcn,H_fcn,inputs)
%This script implements a Least Squares Extended Kalman Filter

%Inputs:
% obs_data = matrix of [t_obs stat obs]

%Outputs:
% Xref_mat = nxl matrix, each column is state vector Xref for that epoch
% P_mat = nxnxl matrix, each nxn portion is state covariance P for that
% epoch
% resids = pxl residuals matrix, each column is residuals for that epoch


%Breakout observation data
t_obs = obs_data(:,1);
stat = obs_data(:,2);
obs = obs_data(:,3:end);

%Number of epochs and observations per epoch
l = size(obs,1);
p = size(obs,2);

%Break out input data
Rk = inputs.Rk;
Xo_ref = inputs.Xo;
Po_bar = inputs.Po;
invPo_bar = inv(Po_bar);
n = length(inputs.Xo);

%Initialize
xo_bar = zeros(n,1);     %State deviation vector at t=0
xo_hat = zeros(n,1);     %Best Estimate of state deviation vector
resids = zeros(p,l);

%Integrator Options
%ode45
ode45_error = 1e-12;
ode45_options = odeset('RelTol',ode45_error,'AbsTol',ode45_error);

%Initialize values
xhat = xo_bar;
P = Po_bar;
Xref = Xo_ref;
phi0 = eye(n);
phi0_v = reshape(phi0,n^2,1);
conv_flag = 0;

%Begin Kalman Filter
for i = 1:l

    if i == 1
        t_prior = 0;
    else
        t_prior = t_obs(i-1);
    end
    
    %Step A: Read the next observation
    ti = t_obs(i);
    stati = stat(i);
    Yi = obs(i,:)';
    
    %Set convergence flag to use EKF
    if i > 100
        conv_flag = 1;        
        %Don't use EKF for big gaps
        if ti - t_prior > 30
            conv_flag = 0;
        end        
    end
    
    
    %Initialize 
    Xref_prior = Xref;
    xhat_prior = xhat;
    P_prior = P;    
    tin = [t_prior ti];
    int0 = [Xref_prior;phi0_v];

    %Step B: Integrate X* and phi    
    tin = [t_prior ti];
    int0 = [Xref_prior;phi0_v];
    if ti == t_prior
        intout = int0';
    else
        [tout,intout] = ode45(intfcn,tin,int0,ode45_options,inputs);
    end

    %Extract the values for use in later calculations
    xout = intout(end,:);
    Xref = xout(1:n)';
    phi_v = xout(n+1:end)';
    phi = reshape(phi_v,n,n);   

    %Step C: Time Update, a priori state and covariance at ti
    xbar = phi*xhat_prior;
    Pbar = phi*P_prior*phi';

    %Step D: Measurement Update 
    [Hi_til,Gi] = feval(H_fcn,Xref,stati,inputs.s101,inputs.s337,inputs.s394,inputs.theta0,inputs.dtheta,ti);
    yi = Yi - Gi;
    Ki = Pbar*Hi_til'*inv(Hi_til*Pbar*Hi_til' + Rk);
    
    %Predicted Residuals
    Bk = yi - Hi_til*xbar;
    P_bk = Rk + Hi_til*Pbar*Hi_til';
    
    %Step E: Compute best estimate state and covariance at ti
    xhat = xbar + Ki*(yi - Hi_til*xbar);
    %P = (eye(n) - Ki*Hi_til)*Pbar;
    
    %Joseph Form
    P = (eye(n) - Ki*Hi_til)*Pbar*(eye(n) - Ki*Hi_til)' + Ki*Rk*Ki';

    %Calculate post-fit residuals
    %Note, yi are pre-fit residuals
    Xk = Xref + xhat;
    resids(:,i) = yi - Hi_til*xhat;
    
    %After filter convergence, update reference trajectory
    if conv_flag
        Xref = Xk;
        xhat = zeros(n,1);
    end

    %Save data to output
    Xref_mat(:,i) = Xref;
    Xk_mat(:,i) = Xk;
    P_mat(:,:,i) = P;
    Pbk_mat(:,:,i) = P_bk;

end











