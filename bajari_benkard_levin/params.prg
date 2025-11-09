
@ simulation parameters @
numtimes = 100;  @ No. of time periods to simulate (size of data set) @
ni = 500;     /* Number of alternative  policies */
nsim = 2000;  /* Number of simulated paths per W calculation */
nbs=400;   /* Number of monte carlo runs */
nsubsamples = 20;    /* Number of inner bootsraps (resampling generated data) */
nper = 80;     /* length of paths (after which prof=0) */
subsamplen = numtimes/2;

maxbdiff=5|30;   /* max diff allowed for subsample run */

@ Parameters for alternative policies @
sd1 = 0.3;    /* SD of inv alternatives */
sd2 = 0.75;    /* SD of exit alternatives */

@ Profit function params @
mc = 3;
M = 5;
wstar = 5;
sigmatemp=0;

/*Matt: income level, parameter*/
income = 6; @Consumer income@
alpha = 1.5;  @parameter of ln(y-p)@
beta = 0.1;  @parameter on w@

prefix="b";

@ For the Bertrand model, set the following parameters. @
kmax = 20;  @ max value of omega; min. value is 0 @
phi = 6;  @ Scrap value @ 
x_entryl = 7;  @ Sunk cost of entry - lowerbound @
x_entryh = 11;  @ Sunk cost of entry - upperbound @
trueb=-1|phi;   @ True value of dynamic params @

@ Define some constants @
entry_k = 7; @ omega at which firms enter @
rlnfirms = 3; @ Real max. # of firms @
stfirm = 1;     @ Firm to start at @
encfirm = 5;  @ Max. number of firms to encode in table @

@ dynamic params @
betad = 0.925;  @ Discounting factor @
betadp = betad^seqa(0,1,nper);
delta = 0.7; @ prob. that outside world moves up @
a = 1.25; @ From p(x) = ax / (1 + ax) @
tol = 0.0001;  @ Tolerance for convergence @

@ Starting state for simulations @
eyenfirms=eye(rlnfirms);

@ Value function variables @
v=0;       /* Value function */
x=0;       /* Investment */
p=0;       /* Probability of state rising */
isentry=0; /* Probability of entry */
profit=0;  /* Profit */
csurplus=0;/* Consumer Surplus */
share=0;   /* Market share */
pmcmargm=0;/* P/MC */
concentm=0;/* One Firm Concentration Ratio */
squant=0;  /* Quantity */
sprice=0;  /* Price */

@ Format masks @
let mask[1,3] = 0 1 1;
let fmt[3,3] = "-*.*s" 8 8
               "*.*lf" 10 3
               "*.*lf" 10 3;
let mask3[1,4] = 0 1 1 1;
let fmt3[4,3] = "-*.*s" 8 8
               "*.*lf" 10 3
               "*.*lf" 10 3
               "*.*lf" 10 3;

@ Entry bandwidth @
entrybw=1.4|1.4;
invbw=1.4|1.4|1.4;
exitbw=1.4|1.4|1.4;

@ Entry Costs @
nentrysim=1000;
quants=seqa(4,0.5,20);
entrycostbw=2;

@ True value of entry quantiles @
truee=(quants-x_entryl)/(x_entryh-x_entryl);
truee=minc(truee'|ones(1,rows(quants)));
truee=maxc(truee'|zeros(1,rows(quants)));

@ amoeba params @
ftol = 1e-04; ftol2=1e-04; amoebaprn = 10; itmax=250;_zerocheck=0;

@ Set up binomial coefficients for decoding/encoding of n-tuples @
binom = eye(rlnfirms+kmax+1);
binom = zeros(rlnfirms+kmax+1,1)~binom;
i=2;
do while i <= rlnfirms+kmax+1;
  binom[i,2:i] = binom[i-1,2:i] + binom[i-1,1:i-1];
  i=i+1;
endo;

@ Number of possible industry structures @
wmax = binom[rlnfirms+1+kmax,kmax+2];


/* Rescales integer states into true state values */
proc rescale(ntuple);
retp(ntuple*1.5-25);
endp;


/* Encodes a state "tuple" into a numeric integer code, 1..wmax */
proc encode(ntuple);
@ This procedure takes a weakly descending n-tuple (n = rlnfirms), with @
@ min. elt. 0, max. elt. kmax, and encodes it into an integer @
 
  local code,digit,i;
  code = 1; @ Coding is from 1 to wmax @
  i = 1;
  do while i <= rlnfirms;
    digit = ntuple[i];
    code = code + binom[digit+rlnfirms+1-i,digit+1];
    i=i+1;
  endo;
  retp(code);
endp;
 

/* Following are two different versions of the state decode procedure */
proc decode(code);
@ This procedure takes a previously encoded number, and decodes it into @
@ a weakly descending n-tuple (n = nfirms)                              @

  local ntuple,digit,i;
  code = code-1;
  ntuple = zeros(rlnfirms,1);
  i = 1;
  do while i <= rlnfirms;
    digit = 0;
    do while binom[digit+rlnfirms-i+2,digit+2] <= code;
      digit=digit+1;
    endo;
    ntuple[i] = digit;
    code = code-binom[digit+rlnfirms-i+1,digit+1];
    i = i+1;
  endo;
  retp(ntuple);
endp;


proc decode2(code,nfirms);
@ This procedure takes a previously encoded number, and decodes it into @
@ a weakly descending n-tuple (n = nfirms - 1)                          @

  local ntuple,digit,i;
  code = code-1;
  ntuple = zeros(nfirms-1,1);
  i = 1;
  do while i <= nfirms - 1;
    digit = 0;
    do while binom[digit+nfirms-i+1,digit+2] <= code;
      digit=digit+1;
    endo;
    ntuple[i] = digit;
    code = code-binom[digit+nfirms-i,digit+1];
    i = i+1;
  endo;

  retp(ntuple);
endp;
