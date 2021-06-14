function [Lhat]=flipflop_Laplacian_est(Lhat0,alp,bet,ytrain,K,Niter,xi)
% Arun Venkitaraman 2018-04-01

% This function estimates the graph Laplacian along with the optimal kernel 
% regression coefficients assuming an initial estimate of the graph-Laplacian Lhat0


%% Inputs:
% Niter: No of flip-flop iterations
% Lhat0: initial value of Lhat
% alp: ridge regularization parameter
% bet: graph regularization parameter
% ytrain: training target matrix
% K: kernel matrix of input observations
% xi: regularization for Laplacian Frobenius norm; set to 0.1 typically in
% our experiments

%% Outputs:
% Lhat: estimated eigenvalue normalized Laplacian

m=size(Lhat0,1);
Lhat=Lhat0; % Initialization for the Laplacian estimation, we have used Lhat0=0 in our experiments.


[u,d_thet]=eig(K);
whos d_thet
d_thet=diag(d_thet);

for iter=1:Niter
            [v,d_lam]=eig(Lhat);
            d_lam=diag(d_lam);
            Z=kron(v,u);
            Psi_lin_g=KerRegGraph_fast(alp,bet,ytrain,d_lam,d_thet,Z);
   
            y_lin_g_train=K*Psi_lin_g;
            
            cvx_begin sdp quiet
            variable Lhat(m,m) symmetric
            minimize((1)*trace(y_lin_g_train'*y_lin_g_train*Lhat)+xi*norm(Lhat-0*diag(diag(Lhat)),'fro'))
            subject to
            %diag(Lhat)==1
            Lhat*ones(m,1)==zeros(m,1)
            %Lhat>=0;
            for i=1:m
                for j=1:m
                    if(i~=j)
                        Lhat(i,j)<=0;
                    end
                end
            end
            
            %trace(Lhat)==m;
            
            cvx_end
            
            Lhat=Lhat/max(abs(eig(Lhat))); % Normalizing
            
        end