function[Wt, F_list]=model(train_data, train_target, lambda1, lambda2, lambda3, lambda4, lambda5)
% m ��ǩ����
[n, m]=size(train_target);
% d ����ά��
[~, d]=size(train_data);


% ��ʼ��W, D, N
XTX = train_data' * train_data;
XTY = train_data' * train_target;
Wt = (XTX + eye(d)) \ (XTY);


Dt = rand(n, d);
Nt = train_data - Dt;

% �����ǩ�����       ---Ҫ��
% Rx = corr(train_target);

Rx = zeros(m, m);
sigma = 1;
mix = train_target';
for i = 1 : m
    for j = 1 : m
        norm2 = norm(mix(:, i) - mix(:, j));
        Rx(i, j) = exp(-1 *  norm2 * norm2 / (2 * sigma * sigma));
    end
end

% �������ƶ�
% Rx = squareform(1-pdist(train_target','cosine')) + eye(size(train_target',1));


% �������������
% ֻ�����������ƶȣ����ڵ����и���
Rl = zeros(n, n);
% mix = [train_data, train_target];
% mix = train_data;
% mix = train_target;
% for i = 1 : n
%     for j = 1 : n
%         norm2 = norm(mix(i, :) - mix(j, :));
%         Rl(i, j) = exp(-1 *  norm2 * norm2 / (2 * sigma * sigma));
%     end
% end


% FunctionValue�洢
F_list = zeros(2, 100);



% ����
Flag = true;
iteration = 1;
while Flag && iteration <= 25
% while Flag && iteration <= 100

    [Wtplus, Dtplus, Ntplus, ~] = opt(train_data, train_target, Rx, Rl, Wt, Dt, Nt, lambda1, lambda2, lambda3, lambda4, lambda5);
    
    % �����Է���
%     [Wtplus, Dtplus, Ntplus, Fvalue] = opt(train_data, train_target, Rx, Rl, Wt, Dt, Nt, lambda1, lambda2, lambda3, lambda4, lambda5);
%     F_list(1, iteration) = iteration;
%     F_list(2, iteration) = Fvalue;
%     if norm(Wtplus - Wt, 'fro')/norm(Wt, 'fro') < 0


    if norm(Wtplus - Wt, 'fro')/norm(Wt, 'fro') < 10^-5
        Flag=false;
    else
        iteration = iteration + 1; 
        Wt = Wtplus;
        Dt = Dtplus;
        Nt = Ntplus;
    end
%     fprintf("iteration ---- %d\r", iteration);
end

end