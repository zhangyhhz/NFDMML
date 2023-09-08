function [f,g] = optn(x,Data)

[n, ~] = size(Data.Dt);
[d, ~] = size(Data.Wt);
Wt = Data.Wt;
Dt = Data.Dt;
Nt = reshape(x, n, d);

E = ones(size(Data.train_target));
HingeL = max((E - Data.train_target .* (Dt * Wt)) , 0);
term1 = 1/2 * norm((HingeL.^2),1);

% term1 = 1/2 * norm(Data.train_target - Dt * Wt, 'fro')^2;

term2 = Data.lambda1 * 1/2 * norm(Wt, 'fro')^2;

term3 = Data.lambda2 * sum(abs(svd(Nt, 'econ')));
% term3 = Data.lambda2 * norm(Nt, 1);

L1 = diag(sum(Data.Rx, 2)) - Data.Rx;
% term4 = Data.lambda3 * trace(Wt * L1 * Wt');
term4 = Data.lambda3 * trace((Dt * Wt) * L1 * (Dt * Wt)');


% L2 = diag(sum(Data.Rl, 2)) - Data.Rl;
% term5 = Data.lambda4 * trace(Dt' * L2 * Dt);

term5 = Data.lambda4 * 1/2 * norm(Dt - Data.train_target * Wt', 'fro')^2;

term7 = Data.lambda5 * 1/2 * norm(Data.train_data - Dt - Nt, 'fro')^2;


f = term1 + term2 + term3 + term4 + term5 + term7;

% ��
gterm3 = Data.lambda2 * subgradient_nuclearnorm(Nt);
% gterm3 = Data.lambda2 * sign(Nt);

gterm7 = -Data.lambda5 * (Data.train_data - Dt - Nt);

g = gterm3 + gterm7;

g=reshape(g, n*d, 1);

end