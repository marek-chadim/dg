/* This is a subroutine to generate static bertrand */
/* profits given demand parameters alpha and beta */
proc statprofs;
local profit, w2, egw, i, p, profstar, w;

sigmatemp = zeros(rlnfirms,1);  /* temp variable for market shares */
profit = zeros(wmax,rlnfirms);
p = 5.5*ones(rlnfirms,1);
i = 1;
do while i <= wmax;
  w = rescale(decode2(i,rlnfirms+1));
  p = newton(p,w,&funk);
  profstar = M*p.*sigmatemp - M*estmc*sigmatemp;
  profit[i,.]=profstar';
  i = i+1;
endo;

retp(profit);

endp;


/* This is the first order condition for profits. */
proc funk(p,w);
  local n,denom,sigmaprime,egw;

  egw = eg(w);
  n = egw.*(income-p)^alphain;
  denom = 1.0 + sumc(n);
  sigmatemp = n./denom;
  sigmaprime = zeros(rows(w),1);
  sigmaprime = -alphain/(income-p).*(1-sigmatemp);

  retp((p-estmc).*sigmaprime + 1);
endp;


@ Calculates e^g(w) @
proc eg(w);
  local i,wret;

  wret = zeros(rows(w),1);
  i=1;
  do while i <= rows(w);
    wret[i] = exp(betain*(wstar - ( (wstar - w[i])^4 )/(5*wstar)^3) );
    i=i+1;
  endo;
  retp(wret);
endp;



@ This procedure performs a simple Newton-Raphson search to find the root
  of the function objfunk @
proc newton(p,w,&objfunk);
  local objfunk:proc,deriv,pnew,epsilon,norm,tol,x,i,id,iter,maxiter;

  id = eye(rows(p));
  epsilon=0.0001; tol = 1e-8;
  maxiter = 100; iter = 0;
  norm = tol+1;
  deriv = zeros(rows(p),rows(p));
  do while (norm > tol) and (iter < maxiter);
    iter=iter+1;
    @ Calculate function at p @
    x = objfunk(p,w);
    @ Calculate derivative matrix @
    i=1;
    do while i <= rows(p);
      deriv[.,i] = (objfunk(p+(epsilon.*id[.,i]),w)-x)/epsilon;
      i=i+1;
    endo;
    pnew = p - 0.6*inv(deriv)*x;
    norm = maxc(abs(pnew - p));
    p = pnew; 
  endo;
  if norm > tol;
    "Error: Newton method could not solve this problem.";
  endif;
  retp(pnew);
endp;
