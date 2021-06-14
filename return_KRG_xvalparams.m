% Arun Venkitaraman 2018-01-01

function [all_alpha,all_beta,all_mse]=return_KRG_xvalparams(X_train, Y_train,T_train,n,L, R, alpvec,betvec,sig_ker)
%% INPUTS
% X_train : input data matrix, samples along rows
% Y_train : output data matrix, samples along rows
% n: no of observation samples
% m : graph size
% L: graph Laplacian
% R: no of partitions for xvalidation
% alpvec: range of alpha values for grid search
% betvec: range of beta values for search

% OUTPUT
%This function returns \alpha and \beta parameters for 
%1. LR 2. LRG 3. KR, and 4. KRG

m=size(L,2);


La=length(alpvec);
Lb=length(betvec);
indices=crossvalind('Kfold',n,R);

for r=1:R
test=(indices==r);
train=~test;
    
    Phi_train=X_train(train,:);
    Phi_test=X_train(test,:);
    
    
    % Separating training data into training and validation sets for
    % crossvalidation
    
    ytrain=T_train(train,:);
    ytrain0=Y_train(train,:);
    ytest=Y_train(test,:); 
    
    K1=Phi_train*Phi_train';
    k1=(Phi_train*Phi_test')';
    
    K2=pdist2(Phi_train,Phi_train).^2;
    sig_rbf=sig_ker*mean(K2(:));
    K2=exp(-K2/sig_rbf);
    k2=pdist2(Phi_test,Phi_train).^2;
    k2=exp(-k2/sig_rbf);
      
  
    
    [u1,d_thet1]=eig(K1);
    d_thet1=diag(d_thet1);
    [u2,d_thet2]=eig(K2);
    d_thet2=diag(d_thet2);
    [v,d_lam]=eig(L);
    d_lam=diag(d_lam);
    
    
    Z1=kron(v,u1);
    Z2=kron(v,u2);
    
    % Internal loops for both alpha and beta parameters
    for b=1:Lb
        for a=1:La
            
            bet=betvec(b);
            alp=alpvec(a);
            %% Kernel
            Psi_lin=KerRegGraph_fast(alp,0,ytrain,d_lam,d_thet1,Z1);
            Psi_lin_g=KerRegGraph_fast(alp,bet,ytrain,d_lam,d_thet1,Z1);
            
            Psi_ker=KerRegGraph_fast(alp,0,ytrain,d_lam,d_thet2,Z2);
            Psi_ker_g=KerRegGraph_fast(alp,bet,ytrain,d_lam,d_thet2,Z2);
            
            
            % Training predictions
            y_lin_train=K1*Psi_lin;
            y_lin_g_train=K1*Psi_lin_g;
            
            y_ker_train=K2*Psi_ker;
            y_ker_g_train=K2*Psi_ker_g;
            
            % Test predictions
            
            y_lin_test=k1*Psi_lin;
            y_lin_g_test=k1*Psi_lin_g;
            
            y_ker_test=k2*Psi_ker;
            y_ker_g_test=k2*Psi_ker_g;
            

             mse_lin_test(a,b, r)=(norm(ytest(:)-y_lin_test(:),2)^2);
             mse_lin_g_test(a,b, r)=(norm(ytest(:)-y_lin_g_test(:),2)^2);

            mse_ker_test(a,b,r)=(norm(ytest(:)-y_ker_test(:),2)^2);
            mse_ker_g_test(a,b,r)=(norm(ytest(:)-y_ker_g_test(:),2)^2);
        end    
    end
    
            e_test(r)=norm(ytest(:),2)^2;
            
end


for a=1:La
    for b=1:Lb
  
        
        Mse_lin_test(a,b)=(mean(mse_lin_test(a,b,:))./mean(e_test));
        [a1,b1,val1]=matrix_max_loc(Mse_lin_test);
        
        Mse_lin_g_test(a,b)=(mean(mse_lin_g_test(a,b,:))./mean(e_test));
        [a2,b2,val2]=matrix_max_loc(Mse_lin_g_test);
        
        
        Mse_ker_test(a,b)=(mean(mse_ker_test(a,b,:))./mean(e_test));
        [a3,b3,val3]=matrix_max_loc(Mse_ker_test);
        
        Mse_ker_g_test(a,b)=(mean( mse_ker_g_test(a,b,:))./mean(e_test));
        [a4,b4,val4]=matrix_max_loc(Mse_ker_g_test);
        
        
    end
end
all_mse=[val1,val2,val3,val4];
all_alpha=alpvec([a1(1) a2(1) a3(1) a4(1)]);
all_beta=betvec([b2(1) b4(1)]);