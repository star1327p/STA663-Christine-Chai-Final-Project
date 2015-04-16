function [Inv] = calcInverse(Z, M, i, k, val)

% Effective inverse calculation

M_i = M - (M*Z(i,:)'*Z(i,:)*M)/(Z(i,:)*M*Z(i,:)'-1);

Z(i,k) = val;

M = M_i - (M_i*Z(i,:)'*Z(i,:)*M_i)/(Z(i,:)*M_i*Z(i,:)'+1);

Inv = M;



