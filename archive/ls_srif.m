function [Xref_mat,Xk_mat,P_mat,resids] = ls_srif(obs_data,intfcn,H_fcn,inputs)
%This script implements a Least Squares Square Root Information Filter

%Inputs:
% obs_data = matrix of [t_obs stat obs]
% intfcn = handle, integration routine function name
% H_fcn = handle, observation mapping matrix function name
% inputs = structure of parameters, initial conditions

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
R_meas = inputs.Rk;
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

%Initialize SRIF values
Ro_bar = chol(invPo_bar);
R = Ro_bar;
bo_bar = Ro_bar*xo_bar;

%Prewhiten measurements
V = sqrtm(R_meas);
%obs = obs*inv(V);

%Begin SRIF
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
    R_prior = R;
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
    Rbar = R_prior*inv(phi);
    
    %Use Householder to make R upper triangular
    %Rbar = householder(Rbar);
    
    bbar = Rbar*xbar;    

    %Step D: Measurement Update 
    [Hi_til,Gi] = feval(H_fcn,Xref,stati,inputs.s101,inputs.s337,inputs.s394,inputs.theta0,inputs.dtheta,ti);
    yi = Yi - Gi;
    
    %Whiten observations
    yi = inv(V)*yi;
    Hi_til = inv(V)*Hi_til;
    
    %Form the matrix A and use householder algorithm
    A = [Rbar bbar;Hi_til yi];
    A = householder(A);
    
    %Extract components of A
    R = A(1:n,1:n);
    b = A(1:n,end);
    e = A(n+1:end,end);

    %Step E: Compute best estimate state and covariance at ti
    xhat = backsub(R,b);
    P = inv(R)*inv(R)';
    
    %Calculate post-fit residuals
    %Note, yi are pre-fit residuals
    Xk = Xref + xhat;
    yi = V*yi;
    Hi_til = V*Hi_til;
    resids(:,i) = yi - Hi_til*xhat;

    %Save data to output
    Xref_mat(:,i) = Xref;
    Xk_mat(:,i) = Xk;
    P_mat(:,:,i) = P;

end











