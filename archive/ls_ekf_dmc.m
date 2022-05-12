function [Xref_mat,Xk_mat,P_mat,Pbk_mat,resids] = ls_ekf_dmc(obs_data,intfcn,H_fcn,inputs)
%This script implements a Least Squares Extended Kalman Filter
%Includes Dynamic Model Compensation

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
ux_sig = inputs.usig(1);
uy_sig = inputs.usig(2);
uz_sig = inputs.usig(3);
Qrci_flag = inputs.Qrci_flag;
tau_x = inputs.tau(1);   %sec
tau_y = inputs.tau(2);   %sec
tau_z = inputs.tau(3);   %sec

%Initialize
xo_bar = zeros(n,1);     %State deviation vector at t=0
xo_hat = zeros(n,1);     %Best Estimate of state deviation vector
resids = zeros(p,l);
Bx = 1/tau_x;
By = 1/tau_y;
Bz = 1/tau_z;

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
    dt = ti - t_prior;    
    
    %Set convergence flag to use EKF
    if i > 100
        conv_flag = 1;        
        %Don't use EKF for big gaps
        if dt > 10
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
    %Dynamic Model Compensation  
    
    Q = zeros(9);
    
    %Only use DMC for small gaps after convergence
    if conv_flag
        %X components
        B = Bx;
        usig = ux_sig;
        Q(1,1) = usig^2*((1/(3*B^2))*dt^3 - (1/B)^3*dt^2 + (1/B)^4*dt*(1 - 2*exp(-B*dt)) + (1/(2*B^5))*(1 - exp(-2*B*dt)));
        Q(1,4) = usig^2*((1/(2*B^2))*dt^2 - (1/B)^3*dt*(1 - exp(-B*dt)) + (1/B)^4*dt*(1 - exp(-B*dt)) - (1/(2*B^4))*(1 - exp(-2*B*dt)));
        Q(1,7) = usig^2*((1/(2*B^3))*(1 - exp(-2*B*dt)) - (1/B)^2*dt*exp(-B*dt));

        Q(4,1) = Q(1,4);
        Q(4,4) = usig^2*((1/B^2)*dt - (2/B^3)*(1 - exp(-B*dt)) + (1/(2*B^3))*(1 - exp(-2*B*dt)));
        Q(4,7) = usig^2*((1/(2*B^2))*(1 + exp(-2*B*dt)) - (1/B^2)*exp(-B*dt));

        Q(7,1) = Q(1,7);
        Q(7,4) = Q(4,7);
        Q(7,7) = usig^2*((1/(2*B^2))*(1 - exp(-2*B*dt)));

        %Y components
        B = By;
        usig = uy_sig;
        Q(2,2) = usig^2*((1/(3*B^2))*dt^3 - (1/B)^3*dt^2 + (1/B)^4*dt*(1 - 2*exp(-B*dt)) + (1/(2*B^5))*(1 - exp(-2*B*dt)));
        Q(2,5) = usig^2*((1/(2*B^2))*dt^2 - (1/B)^3*dt*(1 - exp(-B*dt)) + (1/B)^4*dt*(1 - exp(-B*dt)) - (1/(2*B^4))*(1 - exp(-2*B*dt)));
        Q(2,8) = usig^2*((1/(2*B^3))*(1 - exp(-2*B*dt)) - (1/B)^2*dt*exp(-B*dt));

        Q(5,2) = Q(2,5);
        Q(5,5) = usig^2*((1/B^2)*dt - (2/B^3)*(1 - exp(-B*dt)) + (1/(2*B^3))*(1 - exp(-2*B*dt)));
        Q(5,8) = usig^2*((1/(2*B^2))*(1 + exp(-2*B*dt)) - (1/B^2)*exp(-B*dt));

        Q(8,2) = Q(2,8);
        Q(8,5) = Q(5,8);
        Q(8,8) = usig^2*((1/(2*B^2))*(1 - exp(-2*B*dt)));

        %Z components
        B = Bz;
        usig = uz_sig;
        Q(3,3) = usig^2*((1/(3*B^2))*dt^3 - (1/B)^3*dt^2 + (1/B)^4*dt*(1 - 2*exp(-B*dt)) + (1/(2*B^5))*(1 - exp(-2*B*dt)));
        Q(3,6) = usig^2*((1/(2*B^2))*dt^2 - (1/B)^3*dt*(1 - exp(-B*dt)) + (1/B)^4*dt*(1 - exp(-B*dt)) - (1/(2*B^4))*(1 - exp(-2*B*dt)));
        Q(3,9) = usig^2*((1/(2*B^3))*(1 - exp(-2*B*dt)) - (1/B)^2*dt*exp(-B*dt));

        Q(6,3) = Q(3,6);
        Q(6,6) = usig^2*((1/B^2)*dt - (2/B^3)*(1 - exp(-B*dt)) + (1/(2*B^3))*(1 - exp(-2*B*dt)));
        Q(6,9) = usig^2*((1/(2*B^2))*(1 + exp(-2*B*dt)) - (1/B^2)*exp(-B*dt));

        Q(9,3) = Q(3,9);
        Q(9,6) = Q(6,9);
        Q(9,9) = usig^2*((1/(2*B^2))*(1 - exp(-2*B*dt)));     
        
    end
    
    %If needed, rotate from RIC to ECI frame
%     if Qrci_flag
%         rc_vect = Xref(1:3);
%         vc_vect = Xref(4:6);
%         Q = ric2eci(rc_vect,vc_vect,Q);
%     end
    
   
    xbar = phi*xhat_prior;
    Pbar = phi*P_prior*phi' + Q;

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
    
    ti;

end











