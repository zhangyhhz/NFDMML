function[Wtplus, Dtplus, Ntplus, Fvalue] = opt(train_data, train_target, Rx, Rl, Wt, Dt, Nt, lambda1, lambda2, lambda3, lambda4, lambda5)
Data.train_data = train_data;
Data.train_target = train_target;
Data.Rx = Rx;
Data.Rl = Rl;
Data.Wt = Wt;
Data.Dt = Dt;
Data.Nt = Nt;
Data.lambda1 = lambda1;
Data.lambda2 = lambda2;
Data.lambda3 = lambda3;
Data.lambda4 = lambda4;
Data.lambda5 = lambda5;

[d, m] = size(Wt);
x0 = reshape(Wt, d*m, 1);
out=ncg(@(x) optw(x, Data), x0, 'MaxIters', 1, 'Display', 'off');
Wtplus = reshape(out.X, d, m);
Data.Wt = Wtplus;

[n, m] = size(Dt);
x1=reshape(Dt, m*n ,1);
out=ncg(@(x) optd(x, Data), x1, 'MaxIters', 1, 'Display', 'off');
Dtplus=reshape(out.X, n, m);
Data.Dt = Dtplus;

x2=reshape(Nt, m*n ,1);
out=ncg(@(x) optn(x, Data), x2, 'MaxIters', 1, 'Display', 'off');
Ntplus=reshape(out.X, n, m);

Fvalue = out.F;

end