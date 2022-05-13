function CMI = CMIestim(Xp,Y,Yp,alpha)

if sum(sum(Xp))*sum(sum(Yp))*sum(sum(Y)) ~= 0
    [N,T] = size(Xp);
    sig_xp = N^(-1/(4+T));
    
    [N,T] = size(Y);
    sig_y = N^(-1/(4+T));
    
    [N,T] = size(Yp);
    sig_yp = N^(-1/(4+T));
    
    Ky = GaussMat(Y,sig_xp)/N;
    Kxp = GaussMat(Xp,sig_xp)/N;
    Kyp = GaussMat(Yp,sig_xp)/N;   
    
    KYYp = Ky.*Kyp*N;
    LYYp = eig(KYYp);
    absLYYp = abs(LYYp);
    HYYp = (1/(1 - alpha)) * log2(sum((absLYYp).^(alpha)));
    
    LYp = eig(Kyp);
    absLYp = abs(LYp);
    HYp = (1/(1 - alpha)) * log2(sum((absLYp).^(alpha)));
    
    KXpYp = Kxp.*Kyp*N;
    LXpYp = eig(KXpYp);
    absLXpYp = abs(LXpYp);
    HXpYp = (1/(1 - alpha)) * log2(sum((absLXpYp).^(alpha)));
    
    KYXpYp = Ky.*Kxp.*Kyp*N^2;
    LYXpYp = eig(KYXpYp);
    absLYXpYp = abs(LYXpYp);
    HYXpYp = (1/(1 - alpha)) * log2(sum((absLYXpYp).^(alpha)));
    
    CMI = HYYp - HYp + HXpYp - HYXpYp;
else
    CMI = 0;
end
end

function K = GaussMat(X,sig)

if isempty(sig)
    [N,m] = size(X);
    sig = (1.06*nanstd(X(:))*(N*m)^(-1/5));
end
G = X*X';
K = 2*G - reshape(diag(G),[1,size(G,1)]);
K = (1/(2*(sig)^2)) * (K - reshape(diag(G),[size(G,1),1]));
K = exp(K);
end
