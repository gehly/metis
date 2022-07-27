function [Xref_mat,Xk_mat,P_mat,Pbk_mat,resids,resids_kl,Xkl_mat,Pkl_mat] = ls_ckf_snc_smoother(obs_data,intfcn,H_fcn,inputs)
%This script implements a Least Squares Conventional Kalman Filter
%Includes SNC
%Includes data smoothing

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
    delta_t = ti - t_prior;    
    
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
    %State Noise Compensation
    Q = [ux_sig^2     0             0;
         0         uy_sig^2         0;
         0            0      uz_sig^2];   
     
    
    %If needed, rotate from RIC to ECI frame
    if Qrci_flag
        rc_vect = Xref(1:3);
        vc_vect = Xref(4:6);
        Q = ric2eci(rc_vect,vc_vect,Q);
    end
    
    %Zero out SNC for big gaps
    Gamma = zeros(n,3);
    if delta_t <= 10
        Gamma(1:6,:) = delta_t*[eye(3)*delta_t/2;eye(3)];
    end
    
    xbar = phi*xhat_prior;
    Pbar = phi*P_prior*phi' + Gamma*Q*Gamma';

    %Step D: Measurement Update 
    [Hi_til,Gi] = feval(H_fcn,Xref,stati,inputs,ti);
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
    
    %Save data for smoothing algorithm
    xhat_mat(:,i) = xhat;
    phi_mat(:,:,i) = phi;
    Gamma_mat(:,:,i) = Gamma;
    Pbar_mat(:,:,i) = Pbar;
    P_mat(:,:,i) = P;
    
    %Save data to output
    Xref_mat(:,i) = Xref;
    Xk_mat(:,i) = Xk;    
    Pbk_mat(:,:,i) = P_bk;

end

%Smoothing Algorithm
%Run the filter backwards using final state and covariance as initial
%conditions
xhat_kl = xhat_mat(:,l);
P_kl = P_mat(:,:,l);
Xkl_mat = Xk_mat;
Pkl_mat = P_mat;
resids_kl = resids;

for k = l-1:-1:1    
    %Assign vectors and matrices for this iteration
    xhat_kk = xhat_mat(:,k);
    P_kk = P_mat(:,:,k);
    P_k1k = Pbar_mat(:,:,k+1);
    phi_k1k = phi_mat(:,:,k+1);   
    Gamma_k1k = Gamma_mat(:,:,k+1);
    Xref = Xref_mat(:,k);
    Yi = obs(k,:)';
    stati = stat(k);
    ti = t_obs(k);
    
    %Compute smoothing gain Sk
    Sk = P_kk*phi_k1k'*inv(P_k1k);
    
    %Compute alternate smoothing gain Sk    
    %P_k1k = phi_k1k*P_kk*phi_k1k' + Gamma_k1k*Q*Gamma_k1k';
    %Sk = P_kk*phi_k1k'*inv(P_k1k);
    
    %Compute best estimate and covariance
    xhat_kl = xhat_kk + Sk*(xhat_kl - phi_k1k*xhat_kk);
    P_kl = P_kk + Sk*(P_kl - P_k1k)*Sk';
    
    [R,p] = chol(P_kl);
    if p
        k
        fprintf('Error: P_kl is not positive definite!\n')
    end
    
    [Hi_til,Gi] = feval(H_fcn,Xref,stati,inputs,ti);
    yi = Yi - Gi;
    
    %Save data to output
    Xkl_mat(:,k) = Xref_mat(:,k) + xhat_kl;
    Pkl_mat(:,:,k) = P_kl;    
    resids_kl(:,k) = yi - Hi_til*xhat_kl;
    
end









