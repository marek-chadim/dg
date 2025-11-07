/* Do OLS regression */
/* Takes in y, x and returns coefficients in b and covariance in covm */
/* and r-squared in r2 */
proc (3) = reg(y,x);
  local b, covm, sse, r2;

  b = invpd(x'*x)*(x'*y);
  sse = ( ( (sumc((y - x*b).*(y - x*b)))^2 )/rows(x) );
  covm = sse*invpd(x'*x);
  r2 = 1 - sse/(sumc(y^2)- (sumc(y)^2)/rows(x));

  retp(b,covm,r2);

endp;


/* Local linear nonparametric regression 

  Estimates y=g(x)+epsilon and returns yhat at point x0.

  NOTE: x should NOT contain a constant vector because program inserts
        one autmatically.

  Inputs: 
    x0  -- point at which you want to evaluate g(x) (kx1)
    gam -- bandwidth parameter (kx1)

    xdata -- matrix of observations on x (nxk)
    ydata -- matrix of observations on y (nx1)

   outputs:
     yhat -- predicted y at point tx

*/


wmin=0;   /* Minimum weight to include in regression */


proc llreg2(x0,gam,xdata,ydata);
local w,y,x,z,yhat;

/* calculate kernel weights */
w=prodc(pdfn((x0'-xdata)./gam')');

/* select only data points with positive weight in regression */
x=selif(ones(rows(xdata),1)~xdata,w.>wmin);
y=selif(ydata,w.>wmin);
w=selif(w,w.>wmin);

z=(x.*w)'*x;

/* If enough observations do local linear */
if rank(z)>=cols(x);
 
  /* predict value of y associated with x0 (=x*betahat) */
  yhat=(1~(x0'))*invpd(z)*(x.*w)'*y;

/* otherwise just do local weighted average */
else;
  yhat=sumc(y.*w)/sumc(w);
endif;

retp(yhat);

endp;



/* Does weighted local linear regression where weights are given by "wghts" */
proc wllreg(x0,gam,xdata,ydata,wghts);
local w,y,x,yhat;

/* calculate kernel weights */
w=prodc(pdfn((x0'-xdata)./gam')');

/* select only data points with positive weight in regression */
x=selif(ones(rows(xdata),1)~xdata,w.>0);
y=selif(ydata,w.>0);
w=selif(w.*wghts,w.>0).*eye(rows(x));

/* predict value of y associated with x0 (=x*betahat) */
yhat=(1~(x0'))*invpd(x'w*x)*x'w*y;
retp(yhat);

endp;

