/* This procedure estimates the quantiles of the entry distrbution specified in 
   quants using local linear regression on the simulated value functions. */
proc estentryqs(quants,entrycostbw,dynparams);

local estentryprob,edv,tempstate,w,w2,i,out,gam;

/* Make necessary data */
estentryprob=zeros(wmax2,1);
edv=-999999*ones(wmax2,1);
w=1;
do while w<=wmax2;

  /* Check if any observations */
  if (frequencies1[w]>0);

    /* Simulate EDV of entering at w */
    rlnfirms=rlnfirms-1;
    tempstate=decode(w)|0;
    rlnfirms=rlnfirms+1;
    edv[w]=edventry(tempstate,dynparams);

    /* Record empirical estimate of entry probability */
    w2=encode(tempstate);
    estentryprob[w]=entryllrg[w2];

  endif;

  w=w+1;
endo;
estentryprob=selif(estentryprob,edv.ne -999999);
edv=selif(edv,edv.ne -999999);

/* Now estimate quantiles using local linear */
out=zeros(rows(quants),1);
i=1;
do while i<=rows(quants);
  out[i]=llreg2(quants[i],entrycostbw,edv,estentryprob);
  i=i+1;
endo;

retp(out);

endp;



/* This procedure simulates the EDV of entering at a certain state, given the estimates
   of the dynamic parameters.  Note that this could be done in C as
   well, but since it's not done as often as the W's we didn't bother.  */ 
proc edventry(startingstate,dynparams);

  local wret,i,wthis,w,s,labsin,k,selecter,statepoly,templabels,
        xx,wtrans,profitw,statevec,exitmat,w2,
        investdraws,outsidedraws,entrydraws,wnext,me,meanw,sdw,
        binvestdraws,boutsidedraws,bentrydraws;

  labsin = "Constant"|"myst"|"otst1"|"otst2"|"otst3"|"otst4";

  xx=zeros(rlnfirms,1);           /* going to be predicted investment */
  wret = zeros(nentrysim,1);           /* returned w's */
  selecter = zeros(rlnfirms,1);   /* temp vector */
  statevec = zeros(rlnfirms,2);   /* current state in the simulation */
  wtrans = zeros(rlnfirms,1);     /* matrix of changes in state */

  binvestdraws = rndu(nper*nentrysim,rlnfirms);
  boutsidedraws = rndu(nper*nentrysim,1);       /* draws for outside good (delta) */
  bentrydraws = rndu(nper*nentrysim,1);

  s=1;
  do while s <= nentrysim;

    investdraws = binvestdraws[(s-1)*nper+1:s*nper,.];
    outsidedraws = boutsidedraws[(s-1)*nper+1:s*nper];
    entrydraws = bentrydraws[(s-1)*nper+1:s*nper];

    i=1;

    wthis = startingstate;
    me = rlnfirms;
    profitw = zeros(nper,1);        /* vector of w at each period -- real policy */

    do while i <= nper;

      if me > -1;                /* firm exited */

        w = encode(wthis);

        @Find Predicted Investment@

        xx = maxc(predinvest[w,.]|zeros(1,rlnfirms));

        wtrans = (wthis + (investdraws[i,.]' .< (invparams[1].*xx)./(1+invparams[1].*xx) ) 
                  - (outsidedraws[i] < invparams[2]).*ones(rlnfirms,1)).*(wthis .> 0);    /* Implements investment draws */

        wtrans = wtrans.*(wtrans .<= kmax) + kmax*(wtrans .> kmax);   /* Cut off firms that go above kmax */
        wtrans = wtrans.*(wtrans .>= 1) + (wtrans.<1).*(wthis.>0);                              /* Cut off firms that go below 1 */

        @Figure Out Exit@

        if (i>1);

          /* Implement exit */
          wtrans = wtrans.*(predexit[w,.]' .<= 0.5);

          /* me=-1 if me exits */
          me = (predexit[w,me]<=0.5)*me - (predexit[w,me]>0.5);  
          if me > -1;
            profitw[i] = newprofit[w,me]*betad^(i-1) + 
                         dynparams[1]*xx[me]*betad^(i-1);
          else;
            profitw[i] = dynparams[2]*betad^(i-1);   
          endif;
       
        else;

          /* Implement exit for only the incumbents */
          wtrans[1:rlnfirms-1] = wtrans[1:rlnfirms-1].*
                    (predexit[w,1:rlnfirms-1]' .<= 0.5);

        endif;


        @Figure Out Entry@

        if (i>1) and (wtrans[rlnfirms] == 0);
          wthis[rlnfirms]=0;
          w2=encode(wthis[1:rlnfirms]);
          if entrydraws[i] < entryllrg[w2];
            wtrans[rlnfirms] = entry_k - (outsidedraws[i] < invparams[2]);
          endif;
        endif;

        @ insert entrant in start period @
        if (i==1); 
           wtrans[rlnfirms]=entry_k - (outsidedraws[i] < invparams[2]);
        endif;

        if me > -1;
          selecter = zeros(rlnfirms,1);
          selecter[me] = 1;
          wnext = rev(sortc(wtrans~selecter,1));
          wthis = wnext[.,1];
          me = maxindc(wnext[.,2]);
        endif;

      endif;

      i=i+1;
    endo;

    wret[s] = sumc(profitw);

    s=s+1;

  endo;

  retp(meanc(wret));

endp;
