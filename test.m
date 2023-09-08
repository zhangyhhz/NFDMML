clear;
clc;
addpath(genpath('.'));

mis = 7;

load('medical.mat');


% ²ÎÊý
cv_num = 5;

lambda1   = 1;
lambda2   = 0.01;
lambda3   = 0.01;
lambda4   = 0.0001;
lambda5   = 10;


if exist('train_data','var')==1
    data    = [train_data;test_data];
    target  = [train_target,test_target];  
end
clear train_data test_data train_target test_target
target(target == 0) = -1;

target      = double (target);
data      = double (data);

num_data  = size(data,1);
temp_data = data + eps;
temp_data = temp_data./repmat(sqrt(sum(temp_data.^2,2)),1,size(temp_data,2));
if sum(sum(isnan(temp_data)))>0
    temp_data = data+eps;
    temp_data = temp_data./repmat(sqrt(sum(temp_data.^2,2)),1,size(temp_data,2));
end
temp_data = [temp_data,ones(num_data,1)];


randorder = randperm(num_data);

cvResult  = zeros(6, cv_num);


for j = 1:5
    fprintf('- Cross Validation - %d/%d \r', j, cv_num);
    [cv_train_data,cv_train_target,cv_test_data,cv_test_target] = generateCVSet( temp_data,target',randorder,j,cv_num );

    IncompleteTarget = getIncompleteTarget_lsml(cv_train_target, mis * 0.1, 1);

    [out, ~] = model(cv_train_data, IncompleteTarget, lambda1, lambda2, lambda3, lambda4, lambda5);
    Outputs = (cv_test_data * out)';
    
    thr = TuneThreshold(Outputs, cv_test_target');
    Pre_Labels             = Predict(Outputs,thr);

    tmpResult(1, 1) = Hamming_loss(Pre_Labels, cv_test_target');
    tmpResult(2, 1) = Average_precision(Outputs, cv_test_target');
    tmpResult(3, 1) = coverage(Outputs, cv_test_target');
    tmpResult(4, 1) = One_error(Outputs, cv_test_target');
    tmpResult(5, 1) = Ranking_loss(Outputs, cv_test_target');
    tmpResult(6, 1) = avgauc(Outputs,cv_test_target');
    

    cvResult(:, j) = cvResult(:, j) + tmpResult;
end
Avg_Result      = zeros(6, 2);
Avg_Result(:, 1) = mean(cvResult, 2);
Avg_Result(:, 2) = std(cvResult, 1, 2);
fprintf('\nEvaluation Metric                 \n');
fprintf('-----------------avg----------------\n');
fprintf('HammingLoss           %.4f  %.4f\r', Avg_Result(1,1), Avg_Result(1,2));
fprintf('Averge Prec           %.4f  %.4f\r', Avg_Result(2,1), Avg_Result(2,2));
fprintf('Coverage           %.4f  %.4f\r', Avg_Result(3,1), Avg_Result(3,2));
fprintf('One_error           %.4f  %.4f\r', Avg_Result(4,1), Avg_Result(4,2));
fprintf('Ranking_loss           %.4f  %.4f\r', Avg_Result(5,1), Avg_Result(5,2));
fprintf('AUC           %.4f  %.4f\r', Avg_Result(6,1), Avg_Result(6,2));


fprintf('end.\n');
