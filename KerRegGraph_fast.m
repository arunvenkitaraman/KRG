%% Computes the regression coeffiient matrix \Psi using Kronecker product
% Arun Venkitaraman 2018-01-01
%simplifications
% Takes as inputs:
% alp: alpha regularization
% bet: beta regularization
% ytrain: training output
% d_lam: eigenvalues of the laplacian
% d_thet: eigenvalues of the KErnel matrix
% Z: kron(V,U) where V and U are the eigenvector matrices of Laplacian L and
% kErnel matrix K, repsectively.



function [Psi]=KerRegGraph_fast(alp,bet,ytrain,d_lam,d_thet,Z)



[n,m]=size(ytrain);

J=diag(1./(kron(ones(m,1),alp+d_thet)+kron(bet*d_lam,d_thet)));

vec_a=Z*J*Z'*vec(ytrain);

Psi=reshape(vec_a,n,m);

