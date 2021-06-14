%% Kernel Regression for signals over graphs
% Arun Venkitaraman 2018-01-01

close all
clear all
clc
tic;
n=5;  % No of training samples to be used for KRG


R=4; % No of folds for R-fold crossvalidation
SNR=0; % Signal to noise ratio of the additive noise

dataset='temp2';

[D,L,alpvec,betvec, Ntrain,Ntest,m,offset,city_ip,city_op]=get_dataset(dataset);

    
perturb=0; % To study large perturbations/missing data in training set
perturb=1; % Corresponds to additive noise of signal-to-noise-ratio given by SNR to training data


% D: Data matrix
%L: graph-Laplacian
%Ntrain: subset of indices of D corresponding to training set
%Ntest: data indices corresponding to test data
%m: size of graph
%offset: offset of days to be used in the case of temperature data
% city_ip: is the portion of the entrie data used for input (for example
% some cities in the ETEX data)
% city_op: is the portion of the entrie data used for output that lies over associated graph with Laplacian L

% In order to run synthesized small-world graphs, gspbox toolbox needs to be compiled.
% The information regarding compiling the associated files are provided in folder 'gspbox'

%% Mask to simulate random large perturbations in training data
Mask=ones(n,m);
for i=1:n
    Mask(i,randperm(m,5))=5; % This value set to 0 simulates missing samples, 1 means no noise, >1 implies large perturbation
end



%% Cross validation to find parameters
% Finding parameters alp, bet, sigma^2 for each of the four cases as
% applicable: LR, LRG, KR, and KRG


S=10; % Grid size
Sig_ker=logspace(0,2,S); % Grid range for $\sigma^2$
%Sig_ker=linspace(1.50e3,1.6e3,S); % For Cere sig=1.58e3
Sig_ker=linspace(30,40,S); % For temp17 sig=35
%Sig_ker=logspace(0,1,S); % For eeg sig=5
%Sig_ker=linspace(5,15,S); % For etex sig=5.5
% Sig_ker=

%% Sigma^2 values found by prior experiments
% Sig_ker=1.58e3; % Cere
% Sig_ker=35;% temp17
% Sig_ker=5; % EEG
% Sig_ker=5.5;%ETEX


All_alpha_s=zeros(S,4);
All_beta_s=zeros(S,2);
All_mse_s=zeros(S,4);
Run=10;
for  r=1:Run
    r
    ns=length(Ntrain);
    ntrain=Ntrain(randperm(ns,n));
    ntest=Ntest;
    
    
    ntest=ntest;
    X_train=(D((ntrain)+offset,city_ip));
    Y_train=(D((ntrain),city_op))*pinv(eye(m)+0*L);
    X_test=(D((ntest)+offset,city_ip));
    Y_test=(D((ntest),city_op))*pinv(eye(m)+0*L);
    
   
    
    % Generating noisy data
    sig_train=1*sqrt((norm(Y_train,'fro')^2/(length(Y_train(:))))*10^(-SNR/10));  % computing the variance for additive noise of given SNR
    
    %% Use for large perturbations
    if perturb==1
    T_train=Mask.*Y_train;
    end
    %% Use for additive noise
    if perturb==0
    T_train=(Y_train+1*sig_train*randn(size(Y_train))); %
    end
    
    for ss=1:S
        sig_ker=Sig_ker(ss);
        %%  alpha, beta parameters obtained from the crossvalidation step for a given sigma^2 value
        
        [all_alpha_s(ss,:),all_beta_s(ss,:),all_mse_s(ss,:)]=return_KRG_xvalparams(X_train, Y_train,T_train,n,L,R,alpvec,betvec,sig_ker);
        
        
    end
        All_alpha_s=All_alpha_s+all_alpha_s;
        All_beta_s=All_beta_s+all_beta_s;
        All_mse_s=All_mse_s+all_mse_s;
end

[mm,ii]=min(All_mse_s(:,3)/Run);

all_alpha=All_alpha_s(ii,:)/Run;
all_beta=All_beta_s(ii,:)/Run;
sig_ker=Sig_ker(ii);  % Finding the sig that results in minimum MSE for training set 

a1=all_alpha(1); % alpha for LR
a2=all_alpha(2);% alpha for LRG
a3=all_alpha(3);% alpha for KR
a4=all_alpha(4);% Alpha for KRG
b2=all_beta(1);% beta for LRG
b4=all_beta(2);% beta for KRG



%% At this point we know sigma^2 (given by sig_ker), alpha and beta for all four cases.



%% KRG on test data

% Now running the experiment using estimated hyperparameters for many
% different partitions of training and testing data

for r=1:100
    
    ns=length(Ntrain);
    ntrain=Ntrain(randperm(ns,n));
    ntest=Ntest;
    
    
    ntest=ntest;
    X_train=(D((ntrain)+offset,city_ip));
    Y_train=(D((ntrain),city_op))*pinv(eye(m)+0*L);
    X_test=(D((ntest)+offset,city_ip));
    Y_test=(D((ntest),city_op))*pinv(eye(m)+0*L);
    
    % Generating noisy data
    sig_train=1*sqrt((norm(Y_train,'fro')^2/(length(Y_train(:))))*10^(-SNR/10));  % computing the variance for additive noise of given SNR
    
  
    if perturb==1
         % Use for large perturbations
    T_train=Mask.*Y_train;
    end

    if perturb==0
            % Use for additive noise
    T_train=(Y_train+1*sig_train*randn(size(Y_train))); %
    end
   
    ytrain0=Y_train(1:n,:);
    ytrain=T_train(1:n,:);
    ytest=Y_test;
    
    
    clear train;
    
    Phi_train=X_train;
    Phi_test=X_test;
    
    ytrain=T_train;
    
    K1=Phi_train*Phi_train';
    k1=(Phi_train*Phi_test')';
    
    K2=pdist2(Phi_train,Phi_train).^2;
    sig_rbf=sig_ker*mean(K2(:));
    K2=exp(-K2/sig_rbf);
    k2=pdist2(Phi_test,Phi_train).^2;
    k2=exp(-k2/sig_rbf);
    
    
    %% Computing the eigenvectors of kernel and graph Laplacian matrices for fast computation of KRG
    [u1,d_thet1]=eig(K1);
    d_thet1=diag(d_thet1);
    [u2,d_thet2]=eig(K2);
    d_thet2=diag(d_thet2);
    
    
    %% Here one may insert code for estimating Laplacian
    %   Lhat0=zeros(m); % Initialization for the flip-flop algo
    %    Niter=5; % No of flip flop iterations for the flip-flop algo
    %    
    %   xi=0.1;  % Laplacian Frobenius norm regularization parameter
    %    Lhat=flipflop_Laplacian_est(Lhat0,a1(1),b2(1),ytrain,K1,Niter,xi);
    %    Lhat=flipflop_Laplacian_est(Lhat0,a2(1),b4(1),ytrain,K2,Niter,xi);
        
    
    
    [v,d_lam]=eig(L);
    d_lam=diag(d_lam);
    
    
    Z1=kron(v,u1);
    Z2=kron(v,u2);
    
    %% Kernel
    Psi_lin=KerRegGraph_fast((a1(1)),0,ytrain,d_lam,d_thet1,Z1);
    Psi_lin_g=KerRegGraph_fast((a2),(b2(1)),ytrain,d_lam,d_thet1,Z1);
    
    Psi_ker=KerRegGraph_fast((a3(1)),0,ytrain,d_lam,d_thet2,Z2);
    Psi_ker_g=KerRegGraph_fast((a4(1)),(b4(1)),ytrain,d_lam,d_thet2,Z2);
    
    
    % Training prediction
    y_lin_train=K1*Psi_lin;
    y_lin_g_train=K1*Psi_lin_g;
    y_ker_train=K2*Psi_ker;
    y_ker_g_train=K2*Psi_ker_g;
    
    % Test prediction
    y_lin_test=k1*Psi_lin;
    y_lin_g_test=k1*Psi_lin_g;
    y_ker_test=k2*Psi_ker;
    y_ker_g_test=k2*Psi_ker_g;
    
    % MSE
    mse_lin_train_f(r)=(norm(ytrain0(:)-y_lin_train(:),2)^2);
    mse_lin_g_train_f(r)=(norm(ytrain0(:)-y_lin_g_train(:),2)^2);
    e_train_f(r)=norm(ytrain0(:),2)^2;
    mse_lin_test_f(r)=(norm(ytest(:)-y_lin_test(:),2)^2);
    mse_lin_g_test_f(r)=(norm(ytest(:)-y_lin_g_test(:),2)^2);
    e_test_f(r)=norm(ytest(:),2)^2;
    
    mse_ker_train_f(r)=(norm(ytrain0(:)-y_ker_train(:),2)^2);
    mse_ker_g_train_f(r)=(norm(ytrain0(:)-y_ker_g_train(:),2)^2);
    mse_ker_test_f(r)=(norm(ytest(:)-y_ker_test(:),2)^2);
    mse_ker_g_test_f(r)=(norm(ytest(:)-y_ker_g_test(:),2)^2);
end

%% Final MSE (in dB) values of four approches:
% 1. LR, 2. LRG, 3. KR, and 4. KRG
allmse=10*log10([mean(mse_lin_test_f)/mean(e_test_f) mean(mse_lin_g_test_f)/mean(e_test_f) mean(mse_ker_test_f)/mean(e_test_f) mean(mse_ker_g_test_f)/mean(e_test_f) ])


%% Plotting an example realization
figure, plot(ytrain0(:),'k','Linewidth',2), hold on,plot(ytrain(:),'g'), hold on
hold on, plot(y_lin_train(:),'b'),plot(y_lin_g_train(:),'r'),
plot(y_ker_train(:),'bO-'),plot(y_ker_g_train(:),'rO-')

figure, plot(ytest(:),'k','Linewidth',2),  hold on,
plot(y_lin_test(:),'b'),plot(y_lin_g_test(:),'r'),
plot(ytest(:),'k','Linewidth',2),
plot(y_ker_test(:),'bO-'),plot(y_ker_g_test(:),'rO-')
%
