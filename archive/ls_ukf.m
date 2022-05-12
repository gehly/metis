function [Xref_mat,P_mat,resids] = ls_ukf(obs_data,intfcn,H_fcn,inputs)
%This script implements a Least Squares Unscented Kalman Filter

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
P = Po_bar;
Xref = Xo_ref;

%Step 1: Compute Weights
alpha = 10^(-3.8);
%alpha = 1;
beta = 2;
L = n;
kappa = 3 - L;
lambda = alpha^2*(L + kappa) - L;
gamma = sqrt(L + lambda);

Wm = 1/(2*(L + lambda))*ones(1,2*L);
Wc = Wm;
Wm = [lambda/(L + lambda) Wm];
Wc = [lambda/(L + lambda) + (1 - alpha^2 + beta) Wc];   
    
%Begin Kalman Filter
for i = 1:l
    
    %Step 2: Initialize the UKF
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
    P_prior = P;    
    tin = [t_prior ti];
    %int0 = [Xref_prior;phi0_v];
    
    
    %Step 3: Compute sigma points matrix
    chi = [Xref_prior (Xref_prior*ones(1,L) + gamma*sqrtm(P_prior)) (Xref_prior*ones(1,L) - gamma*sqrtm(P_prior))];
    chi_v = reshape(chi,(L*(2*L + 1)),1);
    
    %Step 4: Integrate Chi
    if ti == t_prior
        intout = chi_v';
    else
        [tout,intout] = ode45(intfcn,tin,chi_v,ode45_options,inputs);        
    end
        
    %Extract the values for use in later calculations
    chi_v = intout(end,:);
    chi = reshape(chi_v,L,(2*L + 1));    
    
    %Step 5: Time Update
    Xbar = chi*Wm';
    Q = zeros(L);
    chi_diff = chi - Xbar*ones(1,(2*L + 1));
    Pbar = Q + chi_diff*diag(Wc)*chi_diff';
    %Pbar = 0.5*(Pbar + Pbar');
    Pbar = (sqrtm(Pbar))'*(sqrtm(Pbar));
    
    %Step 6: Recompute Sigma Points
    chi = [Xbar (Xbar*ones(1,L) + gamma*sqrtm(Pbar)) (Xbar*ones(1,L) - gamma*sqrtm(Pbar))];
    
    %Step 7: Calculated Measurements
    Gi = feval(H_fcn,chi,stati,inputs,ti);
    ybar = Gi*Wm';
    
    %Step 8: Innovation and Cross-Correlation
    y_diff = Gi - ybar*ones(1,(2*L + 1));
    Pyy = Rk + y_diff*diag(Wc)*y_diff';
    Pxy = chi_diff*diag(Wc)*y_diff';
    
    %Step 9: Measurement Update
    K = Pxy*inv(Pyy);
    Xref = Xbar + K*(Yi - ybar);
    P = Pbar - K*Pyy*K';    
    %P = 0.5*(P + P');
    P = (sqrtm(P))'*(sqrtm(P));
    
    if ~isreal(Xref)
        ti
    end
    
    %Calculate post-fit residuals
    chi_post = [Xref (Xref*ones(1,L) + gamma*sqrtm(P)) (Xref*ones(1,L) - gamma*sqrtm(P))];
    Gi_post = feval(H_fcn,chi,stati,inputs,ti);
    ybar_post = Gi_post*Wm';
    resids(:,i) = Yi - ybar_post;
    residsi = Yi - ybar_post;

    %Save data to output
    Xref_mat(:,i) = Xref;
    P_mat(:,:,i) = P;

end











