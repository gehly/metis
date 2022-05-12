function [Xref_mat,P_mat,resids] = ls_batch_consider(obs_data,intfcn,H_fcn,inputs)
%This program implements the batch processing algorithm to determine the 
%state deviation vector based on given input observations.  It includes
%computations for consider covariance.

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
    
%Integrator Options
%ode45
ode45_error = 1e-12;
ode45_options = odeset('RelTol',ode45_error,'AbsTol',ode45_error);


%Batch Processing Algorithm
%Initialize loop parameters
maxiters = 6;
z = 0;
xo_hat_mag = 1;
conv_crit = 0.00003;
%conv_crit = 0;

while xo_hat_mag > conv_crit
    
    %Increment loop counter and exit if too many
    z = z + 1;  
    if z > maxiters
        z = z - 1;        
        fprintf('Solution did not converge in %2.0f iterations. Exiting...\n',z)
        xo_hat_mag
        break
    end
    
    %Step A: Initialize values for iteration
    Mxx = Mxx_bar;
    Mxc = Mxc_bar;
    Mcc = Mcc_bar;
    Nx = invPo_bar*xo_bar;
    
  
    %Step B: Read the next observation
    for i = 1:length(t_obs)
        t = t_obs(i);
        stati = stat(i);
        Y = obs(i,:)';
        
        %Don't include R at this point (same for each iteration)        
        
        %Calculate state transition matrix and new X_ref
        if i == 1
            %Form initial values for Xref and phi
            phi = eye(n);
            phi_v = reshape(phi,n^2,1);
            theta = zeros(n,1);
            theta_v = theta;
            
            Xref = Xo_ref;
            Po = Po_bar;
           
        else           
            %Form the integration routine
            tin = [t_obs(i-1) t];
            int0 = [Xref;phi_v;theta_v];
            [tout,intout] = ode45(intfcn,tin,int0,ode45_options,inputs);
            
            %Extract the values for use in later calculations
            xout = intout(end,:);
            Xref = xout(1:n)';
            phi_v = xout(n+1:n+n^2)';
            phi = reshape(phi_v,n,n);  
            theta_v = xout(n+n^2+1:end)';
            theta = theta_v;
        end    
        
        %Accumulate the current observation
        Xref_mat(:,i) = Xref;
        P_mat(:,:,i) = phi*Po*phi';
        [Hx_til,Hc_til,Gi] = feval(H_fcn,Xref,stati,inputs,t);
        yi = Y - Gi;
        Hx = Hx_til*phi;
        Hc = Hx_til*theta + Hc_til;
        Mxx = Mxx + Hx'*inv(Rk)*Hx;
        Mxc = Mxc + Hx'*inv(Rk)*Hc;
        Mcc = Mcc + Hc'*inv(Rk)*Hc;
        Nx = Nx + Hx'*inv(Rk)*yi;
        
        %N = N + Hi'*inv(Rk)*yi;        
        
        %Form residuals matrix for output
        resids(:,i) = yi;
       
    end
           
    %Step C: Solve normal equations 
    %Use Least Squares solution
    Mcx = Mxc';
    Px = inv(Mxx);
    Sxc = -Px*Mxc
    xc_hat = Px*Nx - Px*Mxc*cbar;
    Pxx = Px + Sxc*Pcc_bar*Sxc';
    Pxc = Sxc*Pcc_bar;
    Pcx = Pxc';
    
    %What should be used for xhat?
    xo_hat = xc_hat
    xo_hat_mag = norm(xo_hat);
    
   
    
    
%     GammaR = chol(Gamma);
%     GammaR_inv = inv(GammaR);
%     Po = GammaR_inv*GammaR_inv';
%     P_mat(:,:,1) = Po;
%     xo_hat = Po*N;
%     xo_hat_mag = norm(xo_hat);
%     
    %Update for next iteration
    Xo_ref = Xo_ref + xo_hat;
    xo_bar = xo_bar - xo_hat;   

end

P_mat(:,:,1) = Px;
