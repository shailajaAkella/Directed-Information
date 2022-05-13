function DI = CalcDI(CR,ER,D)
% CR - rate of causal neuron/LFP band - size num_trials X time
% ER - rate of effect neuron/LFP band - size num_trials X time
% D - maximum delay
[N,~] = size(CR);
CR = parallel.pool.Constant(CR);
ER = parallel.pool.Constant(ER);
DI = zeros(1, N);
% tic
parfor n = 1:N
    if sum(CR.Value(n,~isnan(CR.Value(n,:))))*...
            sum(ER.Value(n,~isnan(ER.Value(n,:))))~= 0
        DI_d = 0;
        for d = 1:D
            T = length(CR.Value(n,~isnan(CR.Value(n,:))));
            cpa = zeros(T-d,d+1); epr = zeros(T-d, 1);
            epa = zeros(T-d,d);
            for i = 1:T-D
                cpa(i,:) = CR.Value(n, i:i + d);
                epr(i,:) = ER.Value(n, i + d);
                epa(i,:) = ER.Value(n, i:i + d - 1);
            end
            cmi = CMIestim(cpa,epr,epa,1.01);
            DI_d = DI_d + cmi;
        end
        DI(n) = DI_d;
    end
end
% toc
end


