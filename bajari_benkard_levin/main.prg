new;
outwidth 160;
output file=main.out reset;
 
format /rd 12,6;
rndseed 123456;

library pgraph optmum;
graphset;

/* Loads C simulation (for simulating W's) procedures */
/* EDIT THIS LINE TO GIVE CORRECT PATH ON YOUR MACHINE */
dlibrary -a /afs/ir.stanford.edu/data/gsb/lanierb/data3/matthewo/pm9fixed/webfiles/libwproc.so;

#include params.prg       /* File with parameters in it */
#include gmopt.arc        /* Nelder-Meade optimization procedure */
#include llreg.prg        /* File with local linear regression procedures */

@CLB #include series.prg @
@CLB entryorder=2;
     exitorder=3;
     investorder=3;  @

@ Load in all the data stored by the equilibrium generation program @ 
loadequil;

/*************************************************************/
/* Beginning of outer monte carlo loop                       */

paramvec=zeros(nbs,2);                    /* Parameter vectors for outer loops */
paramvec2=zeros(nbs,rows(quants));
sesubsamples=zeros(nbs,2);                /* Save avg cov from resampling data */
paramvecsample=zeros(nsubsamples,2);      /* Parameter vectors for inner loops */

/* Outer Monte Carlo Loop */
bs=1;
do while bs<=nbs;

  redo=0;
  seed=floor(1000000*rndu(1,1));
  output on;"Seed " seed;output off;
  rndseed seed;

  /* Draw seeds for inner subsamples here */
  /* Otherwise when you hold simulation draws constant in W's,
     it turns out you fix the subsample draws too -- bad  */
  subsampleseeds=floor(1000000*rndu(nsubsamples+1,1));

  /* Data from one Monte Carlo is quant, price, investment, state, exit */
  dquant=zeros(numtimes,rlnfirms);
  dprice=zeros(numtimes,rlnfirms);
  dinvest=zeros(numtimes,rlnfirms);
  dstate=zeros(numtimes,rlnfirms);
  dexit=zeros(numtimes,rlnfirms);
  dfirmns=zeros(numtimes,rlnfirms);


  /***********************************************************/
  /* Draw starting state at random and simulate one data set */
  /***********************************************************/

  wstart=(floor(8*rndu(1,1))+6)|(floor(8*rndu(1,1))+6)|(floor(8*rndu(1,1))+6);
  wstart=rev(sortc(wstart,1));
  firmns=seqa(1,1,rlnfirms).*(wstart.>0);
  firmn=maxc(firmns);
  output on;"Starting state: " wstart';output off;
  simdata;

  output off;

  /* Set up matrix for investment, exit regressions */
  invmat = zeros(numtimes*rlnfirms,2+rlnfirms);
  makeinvmat;

  /* Set up matrix needed for entry regressions */
  wmax2 = binom[rlnfirms-1+1+kmax,kmax+2];  /* wmax for nfirms-1 */
  frequencies1 = zeros(wmax2,1); @Keep track of how many obs we have@
  frequencies2 = zeros(wmax2,1); @Keep track of how many obs we have@
  entrymat = zeros(numtimes,1);          /* Matrix that is 1 if entry occurs, 0 if not */
  makeentrymat;          /* Fill in entrymat */

  /* Make copies of generated data for resampling procedure (for se's) */
  invmatorig = invmat;
  entrymatorig = entrymat;
  dstateorig = dstate;
  dpriceorig = dprice;
  dquantorig = dquant;
  dexitorig = dexit;

  /*************************************************************/
  /* Beginning of inner subsample loop                         */

  /* Run 0 is estimates.  Runs 1..nsubsamples are resample runs */
  subsample = 0;
  do while (subsample <= nsubsamples) and (redo == 0);

    /*************************************************/
    /*Do shares regression to get demand coefficients*/
    {b, vcovb, rsq} = sharereg;
    demnames = "Beta"|"Alpha";
    if (subsample==0);output on;endif;
    print;"demand coeff, std err.";
    d = printfm(demnames~(beta|alpha)~b~sqrt(diag(vcovb)),mask3,fmt3);
    print;"Demand R-Squared:" rsq;print ;

    /******************************/
    /* Now estimate marginal cost */
    alphain=b[2];   /* alpha parameter for profits */
    betain=b[1];    /* beta parameter for profits */
    estmc = estimatemc;      /* Get estimated marginal cost */
    "Real, Estimated Marginal Cost:" mc estmc;
    output off;

    /********************************************************************/
    /* Estimate probability of investment success, outside good success */
    freq1=0;
    freq2=0;
    {invparams, h} = doprinvlogit;
    invpnames = "a"|"Delta";
    if (subsample==0);output on;endif;
    "Prob inv coeffs, std error:";
    d=printfm(invpnames~(a|delta)~invparams~sqrt(diag(h)),mask3,fmt3);
    print ;
    output off;

    /*********************************************/
    /*   Investment local linear                 */
    if (subsample==0);output on;endif;
    "Doing investment regressions...";
    predinvest=firststinvll(invbw);
    predinvest=predinvest.*(predinvest.>0);
    "Mean Absolute Error: " meanc(meanc(abs(predinvest-x)));
    output off;

    /*********************************************/
    /* Estimate entry function */
    if (subsample==0);output on;endif;
    print;"Doing entry regressions...";
    { entryllrg, maxnormdiff,stentryrsq, entryobs } = doentryest(entrybw);
    entryllrg=minc(entryllrg'|ones(1,rows(entryllrg)));   /* Make sure entryllrg is in [0,1] */
    entryllrg=maxc(entryllrg'|zeros(1,rows(entryllrg)));
    "Entry nobs:" entryobs;
    "Mean absolute error over states w/pos prob of entry: " meanc(abs(selif(entryllrg-isentry,isentry.>0)));print;
    output off;

    /*********************************************/
    /*   Exit local linear                       */
    if (subsample==0);output on;endif;
    "Doing exit regressions...";
    predexit=doexitestll(exitbw);
    "Percent Correct Overall: " 1-sumc(sumc(abs((predexit.>0.5) - (v.==phi))))/(rows(predexit)*cols(predexit));
    output off;

    /************************************************************/
    /* Generate new static profits using the estimated parameters*/
    /* (used in W simulations) */
    newprofit = statprofs;  /* calc profits.  uses alphin, betain */

    /*************************************************************/
    /* Draw random Alternative Policies and Starting states      */
    /* (inequalities to use) */
    altinvest=sd1*rndn(ni,1);
    altexit=sd2*rndn(ni,1);

    /* randomly choose starting states from data but never choose a zero */
    startingstates=floor(numtimes*rndu(ni,1))+1;
    startingme=floor(rlnfirms*rndu(ni,1))+1;
    i=1;
    do while i<=ni;
      do while dstateorig[startingstates[i],startingme[i]]==0;
        startingstates[i]=floor(numtimes*rndu(1,1))+1;
        startingme[i]=floor(rlnfirms*rndu(1,1))+1;
      endo;
      i=i+1;
    endo;

    /**********************************/
    /* Now Calculate the ws and whats */
    wvector=zeros(ni,3);
    sdvec=zeros(ni,9);
    simsigma=zeros(3,3);

    if (subsample==0);output on;endif;
    print;
    "Calculating w's.";
    i=1;
    do while i <= ni;
      {wvector[i,.], sdw} = wmaker_c(altinvest[i],altexit[i],dstateorig[startingstates[i],.]',startingme[i]);
      sdvec[i,.]=vecr(sdw)';
      simsigma=simsigma+sdw;
      i=i+1;
    endo;
    simsigma=(simsigma/(ni))/nsim;

    /******************************************************/
    /* Now do Minimum Distance for dynamic params         */

    /* Select out flat zeros since they contribute nothing to likelihood */
    sdvec=selif(sdvec,maxc(abs(wvector)') .ne 0);
    wvector=selif(wvector,maxc(abs(wvector)') .ne 0);
    "Number of nonzero w's: " rows(wvector);

    b0 = -1|phi;
    add=0.5|3;
    simp=b0~(b0+add.*eye(rows(b0)));
    output off;
    b=gmamoeba(simp,&wmd);             /* Nelder-Meade procedure */
    if (subsample==0);output on;endif;

    /* Only use this run if results within max diff */
    /* (Otherwise something screwy happened.) */
    if (maxc(abs(b-trueb))<maxbdiff);  
  
      if subsample == 0;

        paramvec[bs,.]=b';

        /* Now estimate entry cost distribution */
        output on;
        print;
        "Now estimating entry distribution quantiles...";
        print;
        output off;
        entrydist=estentryqs(quants,entrycostbw,b);
        paramvec2[bs,.]=entrydist';

      else;  
        paramvecsample[subsample,.]=b';
      endif;

      output on;
      "Estimates for Inner Loop # " subsample " : " b' "     Next Seed " subsampleseeds[subsample+1];
      output off;

      subsample = subsample + 1;

    else;
  
      /* If this run screwed up, redo this run */
      redo=1;

    endif;

    /* Resample the data */
    if (subsample<=nsubsamples);
      rndseed subsampleseeds[subsample+1];
      bstrap;
    endif;

  endo;  /* End of inner subsample loop */

  /* Do this if the last monte carlo run wasn't messed up */
  if (redo==0);

    /* Compute booststrap stats */
    msubsample = meanc(paramvecsample);
    sesubsample=sqrt(diag((paramvecsample-msubsample')'(paramvecsample-msubsample')/(nsubsamples-1)));
    sesubsample=sesubsample*sqrt(subsamplen)/sqrt(numtimes);

    /* Record bootstrap stats and save mc stats */
    sesubsamples[bs,.] = sesubsample';
    save paramvec, paramvec2, sesubsamples;

    /* Compute mc stats to print out */
    mb=meanc(paramvec[1:bs,.]);
    seb=sqrt(diag((paramvec[1:bs,.]-mb')'(paramvec[1:bs,.]-mb')/(bs-1)));
    rmseb=sqrt(diag((paramvec[1:bs,.]-trueb')'(paramvec[1:bs,.]-trueb')/bs));
    bssesamp=meanc(sesubsamples[1:bs,.]);
    me=meanc(paramvec2[1:bs,.]);
    see=sqrt(diag((paramvec2[1:bs,.]-me')'(paramvec2[1:bs,.]-me')/(bs-1)));
    rmsee=sqrt(diag((paramvec2[1:bs,.]-truee')'(paramvec2[1:bs,.]-truee')/bs));

    output on;
    "***************************************************************";
    secondstlabs = "Investment"|"Exit Value";
    "Estimated Values for Monte Carlo Outer Loop Run " bs ":";
    d=printfm(secondstlabs~trueb~(paramvec[bs,.]')~sesubsample,mask3,fmt3);

    print;
    "Estimated Entry Distribution for Run " bs ":";
    quants~truee~entrydist;

    print;
    "Summary of MC Runs thus far: (" bs ")";
    "TrueVal Mean, SE,  SE(Mean), RMSE, Mean BSSE(SAMP)";
    trueb~mb~seb~(seb/sqrt(bs-1))~rmseb~bssesamp;

    print;
    "Summary of MC Runs for entry cost distribution: ";
    truee~me~see~(see/sqrt(bs-1))~rmsee;
  
    "***************************************************************";
    output off;

    bs=bs+1;
  endif;


endo;  /* end of outer monte carlo loop */


end;


#include bertprfproc.prg; /* File with static profit function equilibrium calculations in it. */
#include estproc.prg;     /* Estimation procedures */
#include wproc_c.prg;     /* Procedures to simulate W's */
#include estentry.prg     /* Procedures to simulate V and then estimate entry distribution. */


proc (0)=loadequil;
local filename,bigread;

@ This data is: v (value), x (investment), p (probability of state rising), 
                isentry (probability of entry) @

filename = prefix $+ "markov." $+ ftos(rlnfirms,"%*.*lf",1,0) $+ "ot";
load bigread[] = ^filename;
v = (reshape(bigread[1:wmax*rlnfirms],wmax,rlnfirms));
bigread = bigread[wmax*rlnfirms+1:rows(bigread)];
x = (reshape(bigread[1:wmax*rlnfirms],wmax,rlnfirms));
bigread = bigread[wmax*rlnfirms+1:rows(bigread)];
p = (reshape(bigread[1:wmax*rlnfirms],wmax,rlnfirms));
bigread = bigread[wmax*rlnfirms+1:rows(bigread)];
isentry = bigread[1:wmax];

@ Load in all data from the static profit calculation @
@ The data is: firm profits, consumer surplus, market shares at each state,
price/cost margins, one-firm concentration ratios, quantity and price @
filename = prefix $+ "consur." $+ ftos(rlnfirms,"%*.*lf",1,0) $+ "f";
load bigread[] = ^filename;
profit = reshape(bigread[1:wmax*rlnfirms],wmax,rlnfirms);
bigread = bigread[wmax*rlnfirms+1:rows(bigread)];
csurplus = reshape(bigread[1:wmax],wmax,1);
bigread = bigread[wmax+1:rows(bigread)];
share = reshape(bigread[1:wmax*rlnfirms],wmax,rlnfirms);
bigread = bigread[wmax*rlnfirms+1:rows(bigread)];
pmcmargm = reshape(bigread[1:wmax],wmax,1);
bigread = bigread[wmax+1:rows(bigread)];
concentm = reshape(bigread[1:wmax],wmax,1);

filename = prefix $+ "qp." $+ ftos(rlnfirms,"%*.*lf",1,0) $+ "f";
load bigread[] = ^filename;
squant = reshape(bigread[1:wmax*rlnfirms],wmax,rlnfirms);
bigread = bigread[wmax*rlnfirms+1:rows(bigread)];
sprice = reshape(bigread[1:wmax*rlnfirms],wmax,rlnfirms);

endp;
