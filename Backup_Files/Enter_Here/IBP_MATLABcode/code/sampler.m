% This is an implementation of the algorithm described in the Computational
% cognition cheat sheet titied "The Indian Buffet Process."
% Written by Ilker Yildirim, September 2012.

% fix the random seed for replicability.
start = cputime;

randn('seed', 1); rand('seed', 1);

% Generate the synthetic data.

randn('seed', 1); rand('seed', 1);
A = [[0 1 0 0 0 0 1 1 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
     [0 0 0 1 1 1 0 0 0 1 0 1 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 1 0 0 0 0 1 1 1 0 0 0];
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 1 0 0 0 0 0 1 0];];

num_objects = 100; 
object_dim = 36;
sigma_x_orig = .5;

I = sigma_x_orig*diag(ones(object_dim,1));
Z_orig = zeros(num_objects, 4);
X = zeros(num_objects, object_dim);
for i=1:num_objects
    Z_orig(i,:) = (rand(1,4) > .5);
    while (sum(Z_orig(i,:)) == 0)
        Z_orig(i,:) = (rand(1,4) > .5);
    end
    X(i,:) = randn(1, object_dim)*I+Z_orig(i,:)*A;
end;


% The sampler

% Compute Harmonic number for N.
HN = 0;
for i=1:num_objects HN = HN + 1/i; end;

E = 1000;
BURN_IN = 0;
SAMPLE_SIZE = 1000;

% Initialize the chain.
sigma_A = 1;
sigma_X = 1;
alpha = 1;
K_inf = 10;
[Z K_plus] = sampleIBP(alpha, num_objects);
chain.Z = zeros(SAMPLE_SIZE,num_objects,K_inf);
chain.K = zeros(SAMPLE_SIZE, 1);
chain.sigma_X = zeros(SAMPLE_SIZE, 1);
chain.sigma_A = zeros(SAMPLE_SIZE, 1);
chain.alpha = zeros(SAMPLE_SIZE,1);

s_counter = 0;
for e=1:E
    % Store samples after the BURN-IN period.
    if (e > BURN_IN)
        s_counter = s_counter+1;
        chain.Z(s_counter,:,1:K_plus) = Z(:,1:K_plus);
        chain.K(s_counter) = K_plus;
        chain.sigma_X(s_counter) = sigma_X;
        chain.sigma_A(s_counter) = sigma_A;
        chain.alpha(s_counter) = alpha;
    end;
    disp(['At iteration ', num2str(e), ': K_plus is ', num2str(K_plus), ', alpha is ', num2str(alpha)]);

    for i=1:num_objects
        % The matrix M will be handy for future likelihood and matrix
        % inverse computations.
        M = (Z(:,1:K_plus)'*Z(:,1:K_plus) + (sigma_X^2/sigma_A^2)*diag(ones(K_plus,1)))^-1;
        for k=1:K_plus
            % That can happen, since we may decrease K_plus inside.
            if (k>K_plus)
                break;
            end;
            if Z(i,k) > 0
                % Take care of singular features
                if sum(Z(:,k)) - Z(i,k) <= 0
                    Z(i,k) = 0;
                    Z(:,k:K_plus-1) = Z(:,k+1:K_plus);
                    K_plus = K_plus-1;
                    M = (Z(:,1:K_plus)'*Z(:,1:K_plus) + ...
                        (sigma_X^2/sigma_A^2)*diag(ones(K_plus,1)))^-1;
                    continue;
                end;
            end;
            
            % This equations are for computing the inverse efficiently.
            % It is an implementation of the trick from Griffiths and
            % Ghahramani (2005; Equations 51 to 54). 
            M1 = calcInverse(Z(:,1:K_plus), M, i, k, 1);
            M2 = calcInverse(Z(:,1:K_plus), M, i, k, 0);

            % Compute conditional distributions for the current cell in Z.
            Z(i,k) = 1;
            P(1) = likelihood(X, Z(:,1:K_plus), M1, sigma_A, sigma_X, K_plus, num_objects, ...
                object_dim) + log(sum(Z(:,k))- Z(i,k)) -log(num_objects);

            Z(i,k) = 0;
            P(2) = likelihood(X, Z(:,1:K_plus), M2, sigma_A, sigma_X, K_plus, num_objects, ...
                object_dim) + log(num_objects - sum(Z(:,k))) - log(num_objects);
            P = exp(P - max(P));
            
            % Sample from the conditional.
            if rand < P(1)/(P(1)+P(2))
                Z(i,k) = 1;
                M = M1;
            else
                Z(i,k) = 0;
                M = M2;
            end;
        end;
        % Sample the number of new dishes for the current object.
        trun = zeros(1,5);
        alpha_N = alpha / num_objects;
        for k_i=0:4
            Z(i,K_plus+1:K_plus+k_i) = 1;
            M = (Z(:,1:K_plus+k_i)'*Z(:,1:K_plus+k_i) + (sigma_X^2/sigma_A^2)*diag(ones(K_plus+k_i,1)))^-1;
            trun(k_i+1) = k_i*log(alpha_N) - alpha_N - log(factorial(k_i)) + ...
                likelihood(X, Z(:,1:K_plus+k_i), M, sigma_A, sigma_X, K_plus+k_i, num_objects, object_dim);
        end;
        Z(i,K_plus+1:K_plus+4) = 0;
        trun = exp(trun - max(trun));
        trun = trun/sum(trun);
        p = rand;
        t = 0;
        for k_i=0:4
            t = t+trun(k_i+1);
            if p < t
                new_dishes = k_i;
                break;
            end;
        end;
        Z(i,K_plus+1:K_plus+new_dishes) = 1;
        K_plus = K_plus + new_dishes;
    end;

    % Metropolis steps for sampling sigma_X and sigma_A
    M = (Z(:,1:K_plus+new_dishes)'*Z(:,1:K_plus+new_dishes) + ...
        (sigma_X^2/sigma_A^2)*diag(ones(K_plus+new_dishes,1)))^-1;
    l_curr = likelihood(X, Z(:,1:K_plus+new_dishes), M, sigma_A, sigma_X, ...
        K_plus+new_dishes, num_objects, object_dim);

    if rand < .5
        pr_sigma_X = sigma_X - rand/20;
    else
        pr_sigma_X = sigma_X + rand/20;
    end;
    
    M = (Z(:,1:K_plus+new_dishes)'*Z(:,1:K_plus+new_dishes) + ...
        (pr_sigma_X^2/sigma_A^2)*diag(ones(K_plus+new_dishes,1)))^-1;
    l_new_X = likelihood(X, Z(:,1:K_plus+new_dishes), M, sigma_A, pr_sigma_X, ...
        K_plus+new_dishes, num_objects, object_dim);
    acc_X = exp(min(0, l_new_X - l_curr));

    if rand < .5
        pr_sigma_A = sigma_A - rand/20;
    else
        pr_sigma_A = sigma_A + rand/20;
    end;
    M = (Z(:,1:K_plus+new_dishes)'*Z(:,1:K_plus+new_dishes) + ...
        (sigma_X^2/pr_sigma_A^2)*diag(ones(K_plus+new_dishes,1)))^-1;
    l_new_A = likelihood(X, Z(:,1:K_plus+new_dishes), M, pr_sigma_A, sigma_X, ...
        K_plus+new_dishes, num_objects, object_dim);
    acc_A = exp(min(0, l_new_A - l_curr));

    if rand < acc_X
        sigma_X = pr_sigma_X;
    end;
    if rand < acc_A
        sigma_A = pr_sigma_A;
    end;

    % Sample alpha from its conditional posterior.
    alpha = mygamrnd(1+K_plus, 1/(1+HN),1);
    
    % Save the chain at every 1000th iteration.
    if mod(e,1000) == 0
        s = strcat('chain_ibp_',num2str(e));
        save(s, 'chain');
    end;
end;

elapsed = cputime - start  % CPU time spent in seconds

% Make figures.
% Figures for the Computational cognition cheat sheet.


subplot(1,4,1); imagesc(reshape(A(1,:),6,6)); colormap(gray); axis off
subplot(1,4,2); imagesc(reshape(A(2,:),6,6)); colormap(gray); axis off
subplot(1,4,3); imagesc(reshape(A(3,:),6,6)); colormap(gray); axis off
subplot(1,4,4); imagesc(reshape(A(4,:),6,6)); colormap(gray); axis off

%Z_orig

%     1     0     0     0
%     0     1     0     0
%     0     1     1     0
%     0     1     1     1

figure
subplot(1,4,1); imagesc(reshape(X(1,:),6,6)); colormap(gray); axis off
subplot(1,4,2); imagesc(reshape(X(2,:),6,6)); colormap(gray); axis off
subplot(1,4,3); imagesc(reshape(X(3,:),6,6)); colormap(gray); axis off
subplot(1,4,4); imagesc(reshape(X(4,:),6,6)); colormap(gray); axis off


figure
hist(chain.K(201:10:1000),3); colormap(gray)
xlabel('K')
ylabel('Count')
title('Histogram of K values')

figure
Z=reshape(chain.Z(1000,:,1:4),100,4);
sigma_x=chain.sigma_X(1000);
sigma_A=chain.sigma_A(1000);
A_inf=(Z'*Z+(sigma_X/sigma_A)*eye(4))^-1*Z'*X;

subplot(1,4,1); imagesc(reshape(A_inf(1,:),6,6)); colormap(gray); axis off
subplot(1,4,2); imagesc(reshape(A_inf(2,:),6,6)); colormap(gray); axis off
subplot(1,4,3); imagesc(reshape(A_inf(3,:),6,6)); colormap(gray); axis off
subplot(1,4,4); imagesc(reshape(A_inf(4,:),6,6)); colormap(gray); axis off



