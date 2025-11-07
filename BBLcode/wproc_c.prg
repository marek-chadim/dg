/* This program creates the ws and whats.  Note that there are two versions: a gauss version 
   called wmaker, and a C version called wmaker_c that calls a much faster C subroutine for the 
   forward simulations. */

/* altinvest     -- is alternative investment policy, which is estimated policy plus
                    a constant equal to altinvest.
   altexit       -- is alternative exit policy, which is estimated policy plus
                    a constant equal to altexit.
   startingstate -- is state to start at.
   me            -- is firm of interest in startingstate. */


proc (2)=wmaker_c(altinvest,altexit,startingstate,startme);

  local investdraws,outsidedraws,entrydraws,wret,meanw,sdw;

  investdraws = rndu(nper*nsim,rlnfirms);
  outsidedraws = rndu(nper*nsim,1);       /* draws for outside good (delta) */
  entrydraws = rndu(nper*nsim,1);

  wret = zeros(nsim,3);           /* returned w's */
  dllcall wsim(investdraws,outsidedraws,entrydraws,startingstate,startme,
               altinvest,altexit,nsim,nper,predinvest,invparams,predexit,
               wret,newprofit,betadp,entryllrg,binom);
  meanw=meanc(wret)';
  sdw=(wret-meanw)'(wret-meanw)/nsim;

  retp(meanw,sdw);
endp;


/* This is the minimum distance objective function. */
proc wmd(b);

  local b1, t;

  b1 = 1|b;
  t=wvector*b1;

  retp(sumc((t.<0).*t^2));

endp;



/* gauss version of simulation procedure */
proc (2)=wmaker(altinvest,altexit,startingstate,startme);

  local wret,i,wthis,w,s,labsin,k,selecter,statepoly,templabels,
        xx,wtrans,profitw,profitwhat,statevec,exitmat,policy,w2,
        investdraws,outsidedraws,entrydraws,wnext,me,meanw,sdw,
        binvestdraws,boutsidedraws,bentrydraws;

  labsin = "Constant"|"myst"|"otst1"|"otst2"|"otst3"|"otst4";

  xx=zeros(rlnfirms,1);           /* going to be predicted investment */
  profitw = zeros(nper,3);        /* vector of w at each period -- real policy */
  profitwhat = zeros(nper,3);     /* vector of w at each period -- alt. policy */
  wret = zeros(nsim,3);           /* returned w's */
  selecter = zeros(rlnfirms,1);   /* temp vector */
  statevec = zeros(rlnfirms,2);   /* current state in the simulation */
  wtrans = zeros(rlnfirms,1);     /* matrix of changes in state */

  binvestdraws = rndu(nper*nsim,rlnfirms);
  boutsidedraws = rndu(nper*nsim,1);       /* draws for outside good (delta) */
  bentrydraws = rndu(nper*nsim,1);

  s=1;
  do while s <= nsim;

    investdraws = binvestdraws[(s-1)*nper+1:s*nper,.];
    outsidedraws = boutsidedraws[(s-1)*nper+1:s*nper];
    entrydraws = bentrydraws[(s-1)*nper+1:s*nper];

    policy = 0;                        /* policy=1 is alternative policy */

    do while policy <= 1;

      i=1;

      wthis = startingstate;
      me = startme;

      do while i <= nper;

        if me > -1;                /* firm exited */

          w = encode(wthis);

          @Find Predicted Investment@

          xx = predinvest[w,.]';
          xx[me] = xx[me] + policy*altinvest;
          xx=maxc((xx')|zeros(1,rlnfirms));   /* investment can't be negative */

          wtrans = (wthis + (investdraws[i,.]' .< (invparams[1].*xx)./(1+invparams[1].*xx) ) 
                    - (outsidedraws[i] < invparams[2]).*ones(rlnfirms,1)).*(wthis .> 0);    /* Implements investment draws */

          wtrans = wtrans.*(wtrans .<= kmax) + kmax*(wtrans .> kmax);   /* Cut off firms that go above kmax */
          wtrans = wtrans.*(wtrans .>= 1) + (wtrans.<1).*(wthis.>0);                              /* Cut off firms that go below 0 */

          @Figure Out Exit@

          /* Implement exit */
          wtrans = wtrans.*((predexit[w,.]' + altexit*policy*eyenfirms[.,me]) .<= 0.5);

          /* me=-1 if me exits */
          me = (predexit[w,me]+altexit*policy<=0.5)*me - (predexit[w,me]+altexit*policy>0.5);  

          if policy == 0;
            if me > -1;
              profitw[i,1] = newprofit[w,me]*betad^(i-1);
              profitw[i,2] = xx[me]*betad^(i-1);
              profitw[i,3] = 0;
            else;
              profitw[i,.] = 0~0~betad^(i-1);   
            endif;
          else;
            if me > -1;
              profitwhat[i,1] = newprofit[w,me]*betad^(i-1);
              profitwhat[i,2] = xx[me]*betad^(i-1);
              profitwhat[i,3] = 0;
            else;
              profitwhat[i,.] = 0~0~betad^(i-1);
            endif;
          endif;

          @Figure Out Entry@

          if wtrans[rlnfirms] == 0;
            rlnfirms = rlnfirms-1;
            w2=encode(wthis[1:rlnfirms]);
            rlnfirms = rlnfirms+1;
            if entrydraws[i] < entryllrg[w2];
              wtrans[rlnfirms] = entry_k - (outsidedraws[i] < invparams[2]);
            endif;
          endif;

          if me > -1;
            selecter = zeros(rlnfirms,1);
            selecter[me] = 1;
            wnext = rev(sortc(wtrans~selecter,1));
            wthis = wnext[.,1];
            me = maxindc(wnext[.,2]);
          endif;


        else;
          if policy == 0;
            profitw[i,.] = 0~0~0;
          else;
            profitwhat[i,.] = 0~0~0;
          endif;
        endif;

        i=i+1;
      endo;

      policy = policy + 1;

    endo;

    wret[s,.] = sumc(profitw-profitwhat)';

    s=s+1;

  endo;

  meanw=meanc(wret)';
  sdw=(wret-meanw)'(wret-meanw)/nsim;

  retp(meanw,sdw);

endp;


