function [Xref_mat,Xk_mat,Xck_mat,P_mat,Pck_mat,resids] = ls_ckf_cc_filter(obs_data,intfcn,H_fcn,inputs)
%This script implements a Least Squares Conventional Kalman Filter.  It
%contains computations for Consider Covariance Filter (does use the
%Consider Covariance information while running the CKF).

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
cbar = inputs.cbar;
Pcc_bar = inputs.Pcc_bar;
Pxc_bar = inputs.Pxc_bar;

%Initialize
xo_bar = zeros(n,1);     %State deviation vector at t=0
xo_hat = zeros(n,1);     %Best Estimate of state deviation vector
resids = zeros(p,l);
Mxx_bar = invPo_bar;
Mxc_bar = zeros(size(Pxc_bar));
Mcc_bar = inv(Pcc_bar);
Sk = zeros(n,1);


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
theta0 = zeros(n,1);
theta0_v = theta0;

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
    
    %Initialize 
    Xref_prior = Xref;
    xhat_prior = xhat;
    P_prior = P;    
    Sk_prior = Sk;
    tin = [t_prior ti];
    int0 = [Xref_prior;phi0_v;theta0_v];

    %Step B: Integrate X* and phi    
    tin = [t_prior ti];
    int0 = [Xref_prior;phi0_v;theta0_v];
    if ti == t_prior
        intout = int0';
    else
        [tout,intout] = ode45(intfcn,tin,int0,ode45_options,inputs);
    end

    %Extract the values for use in later calculations
    xout = intout(end,:);
    Xref = xout(1:n)';
    phi_v = xout(n+1:n+n^2)';
    phi = reshape(phi_v,n,n);   
    theta_v = xout(n+n^2+1:end)';
    theta = reshape(theta_v,n,1);

    %Step C: Time Update, a priori state and covariance at ti
    xbar = phi*xhat_prior;
    Pbar = phi*P_prior*phi';
    
    Sk_bar = phi*Sk_prior + theta;
    xck_bar = xbar + Sk_bar*cbar;
    Pck_bar = Pbar + Sk_bar*Pcc_bar*Sk_bar';
    Pxck_bar = Sk_bar*Pcc_bar;
    

    %Step D: Measurement Update 
    [Hx_til,Hc_til,Gi] = feval(H_fcn,Xref,stati,inputs,ti);
    yi = Yi - Gi;
    Ki = Pbar*Hx_til'*inv(Hx_til*Pbar*Hx_til' + Rk);
    Kck = Pck_bar*Hx_til'*inv(Hx_til*Pck_bar*Hx_til' + Rk);
    
    %Predicted Residuals
    Bk = yi - Hx_til*xbar;
    P_bk = Rk + Hx_til*Pbar*Hx_til';
    
    if ti - t_prior > 100
        P_bk;
    end

    %Step E: Compute best estimate state and covariance at ti
    xhat = xbar + Ki*(yi - Hx_til*xbar);
    %P = (eye(n) - Ki*Hi_til)*Pbar;
    
    %Joseph Form
    P = (eye(n) - Ki*Hx_til)*Pbar*(eye(n) - Ki*Hx_til)' + Ki*Rk*Ki';
    
    Sk = (eye(n) - Ki*Hx_til)*Sk_bar - Ki*Hc_til;
    xck_hat = xbar + Kck*(yi - Hx_til*xbar);
    Pck = P + Sk*Pcc_bar*Sk';
    Pxck = Sk*Pcc_bar;

    %Calculate post-fit residuals
    %Note, yi are pre-fit residuals
    Xk = Xref + xhat;
    resids(:,i) = yi - Hx_til*xck_hat;

    %Save data to output
    Xref_mat(:,i) = Xref;
    Xk_mat(:,i) = Xk;
    Xck_mat(:,i) = Xref + xck_hat;
    P_mat(:,:,i) = P;
    Pbk_mat(:,:,i) = P_bk;
    Pck_mat(:,:,i) = Pck;


end

