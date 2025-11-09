/* Subprograms for main.prg */

/*******************************************************************************/
/*                                                                             */
/* Data generation and organization procedures.                                */
/*                                                                             */
/*******************************************************************************/

/* Uses equilibrium output to simulate one dataset */
proc (0)=simdata; 

local wthis,lastsize,t,codew,codew2,pp,xx,vv,wtrans,i,wnext,
      temp,yesentry,entrypr;

wthis = wstart; 
lastsize = share[encode(wthis),.]';  @ Shares of firms last iteration @
t = 1;
do while t <= numtimes;
  @ Get probabilities of investment causing a rise in eff level, as well
  as actual investment and value function for this state tuple @
  codew2 = encode(wthis);
  pp = p[codew2,.]'; 
  xx = x[codew2,.]'; 
  vv = v[codew2,.]'; 

  dstate[t,.]=wthis';
  dfirmns[t,.]=firmns';

  @ Figure out exit @
  wtrans = zeros(rlnfirms,1);
  i = (minc(vv) == phi)*(minindc(vv)-1) +
     (minc(vv) > phi)*rlnfirms;
  if i > 0;
    wtrans[1:i] = wthis[1:i];   
  endif;

  firmns = (wtrans.>0) .* firmns;
  dexit[t,.] = (wtrans .== 0)' .and (wthis .> 0)';
  dinvest[t,.]=xx';

  codew = encode(wtrans);
  dquant[t,.]=squant[codew,.].*(wtrans.>0)'; 
  dprice[t,.]=sprice[codew,.].*(wtrans.>0)'; 

  @ Now figure out entry @
  yesentry = 0;
  entrypr = rndu(1,1);
  yesentry = (isentry[codew]>entrypr);
  if yesentry;
    wtrans[rlnfirms] = entry_k;
    firmn=firmn+1;
    firmns[rlnfirms] = firmn;
  endif;

  wnext = wtrans+(pp.>=rndu(rlnfirms,1)) - (rndu(1,1) <= delta); 
  wnext = maxc((wnext~zeros(rlnfirms,1))');
  
  if t % 100 == 0;
    "Completed " t " time periods of simulation draws, state is " wthis';
  endif;
 
  @ Now re-sort all firm level data, to reflect the fact that firms
  must be in descending order next year @

  temp = rev(sortc(wnext~firmns,1)); 
  wthis = temp[.,1]; firmns = temp[.,2]; 

  t = t+1;
endo;

endp;




/* The next procedure makes the matrix invmat. */
/* invmat is a matrix of whether my state changed in the first column, */
/* If a firm drops out next period, the first column is -10 */
/* if no firm there, first colum is -1000 */
/* my investment in the second column */
/* my state in the third column, and other guys states */
/* in the rest of the columns in order from greatest to least */
proc (0) = makeinvmat;
  local i, j, k, firmnumber;

  i=1;
  do while i <= numtimes-1;
    j=1;
    do while j <= rlnfirms;

      /* if firm is there in period i */
      if dfirmns[i,j] > 0;  
        k=1;

        /* Looks to find where firm is next period */
        firmnumber = 0;
        do while k <= rlnfirms;
          if dfirmns[i+1,k] == dfirmns[i,j];
            firmnumber = k;
          endif;
          k=k+1;
        endo;

        /* if firm is there next period, find out what happened to it */
        if firmnumber > 0;
          invmat[rlnfirms*(i-1)+j,1] = dstate[i+1,firmnumber] - dstate[i,j];
        else;   /* if firm exited, put -10 in first position */
          invmat[rlnfirms*(i-1)+j,1] = -10;
        endif;
        invmat[rlnfirms*(i-1)+j,2] = dinvest[i,j];   /* firm's investment */
        invmat[rlnfirms*(i-1)+j,3] = dstate[i,j];    /* firm's own state */

        /* Puts rival states into invmat in order from biggest to smallest */
        k=1;
        do while k <= rlnfirms;
          if k < j;
            invmat[rlnfirms*(i-1)+j,3+k] = dstate[i,k];
          elseif k > j;
            invmat[rlnfirms*(i-1)+j,3+k-1] = dstate[i,k];
          endif;
          k=k+1;
        endo;

      /* firm not there in period i at all */
      else;
        invmat[rlnfirms*(i-1)+j,1] = -1000;
      endif;
      j=j+1;
    endo;
    i=i+1;
  endo;

endp;


/* This procedure fills in the entry data matrix, entrymat */
proc (0) = makeentrymat;
  local i, entryflag, state, w, j;

  i=1;
  do while i <= numtimes-1;
    j=1;
    if dexit[i,3] == 1 or dstate[i,3] == 0;
      entryflag = 0;
      state=dstate[i,1:2];
      rlnfirms=rlnfirms-1;
      w=encode(state);
      rlnfirms=rlnfirms+1;
      frequencies2[w]=frequencies2[w]+1;
      if (dstate[i+1,3] > 0) or 
         ((dexit[i,2]==1 or dstate[i,2]==0) and dstate[i+1,2]>0) or
         ((dexit[i,1]==1 or dstate[i,1]==0) and dstate[i+1,1]>0);
        entryflag = 1;
      endif;
      if entryflag == 1;
        entrymat[i]=1;
        frequencies1[w] = frequencies1[w] + 1;
      endif;
    else;
      entrymat[i] = -1;
    endif;
    i=i+1;
  endo;

  frequencies1 = frequencies1./frequencies2;  /* emp. prob of entry */

endp;


/*******************************************************************************/
/*                                                                             */
/* First stage regression procedures.                                          */
/*                                                                             */
/*******************************************************************************/


/* This procedure does the logit shares regression to get demand parameters. */
proc (3) = sharereg;
  local b, vcovb, rsq, mktquant, regmat, xmat, lnshdiff, i, j, n;

  n=rows(dstate); /* Number of observations */
  mktquant = sumc(dquant');
  regmat = zeros(n*rlnfirms,3);

  i=1;
  do while i <= n;
    j=1;
    do while j <= rlnfirms;
      if dquant[i,j] > 0;
        regmat[rlnfirms*(i-1)+j,1] = ln(dquant[i,j]/M) - ln( (M-mktquant[i])/M );
        regmat[rlnfirms*(i-1)+j,2] = wstar - ( (wstar - rescale(dstate[i,j]))^4 )/(5*wstar)^3;
        regmat[rlnfirms*(i-1)+j,3] = ln( income - dprice[i,j] );
      else;
        regmat[rlnfirms*(i-1)+j,1] = -1;
      endif;
      j=j+1;
    endo;
    i=i+1;
  endo;

  regmat = rev(sortc(regmat,1));

  lnshdiff = selif(regmat[.,1],regmat[.,1].ne -1);
  xmat = selif(regmat[.,2:3],regmat[.,1].ne -1);

  {b, vcovb, rsq} = reg(lnshdiff, xmat);

  retp(b, vcovb, rsq);

endp;


/* This procedure estimates marginal cost conditional on demand parameters. */
/* Note that this could have been done jointly with the demand equation above
   but we didn't bother as there was no need. */
proc estimatemc;
  local predsigma, temp;

  predsigma = (exp(betain*(wstar - ((wstar - rescale(dstate))^4)/(5*wstar)^3)).*(income-dprice)^alphain).*((dstate .> 0) .and (dexit .ne 1));
  predsigma = predsigma./((1.0 + sumc(predsigma'))*ones(1,rlnfirms));
  temp=(dprice - (income-dprice)./(alphain*(1-predsigma)));
  retp( sumc(sumc(temp.*(temp.>0)))./(sumc(sumc(temp.>0))));
endp;



/* Likelihood function for investment transitions. */
/* Procedure that maximizes this function is below. */
proc invlik(b);
  local i,delta,a,pr,invoutcomes,likel,t,t2,n;

  n=rows(invmat)/rlnfirms;  /* number of observations */
  a = abs(b[1]);
  delta=exp(b[2])/(1+exp(b[2]));

  likel=0;t=0;t2=0;
  i=1;
  do while i<=n-1;

    /* Get investment outcomes for all firms in period i */
    invoutcomes=invmat[rlnfirms*(i-1)+1:rlnfirms*i,1];

    /* Don't use top and bottom states because cutoffs confuse things */
    /* Also make sure there are observations */
    if (minc(invmat[rlnfirms*(i-1)+1,3:3+rlnfirms-1]')==1 or
        maxc(invmat[rlnfirms*(i-1)+1,3:3+rlnfirms-1]')==kmax or
        sumc(invoutcomes.>-10)==0);
        pr=1;
    else;

    /* if all firms that we observe are at zero, then we don't 
       know what og did */
    if (sumc(invoutcomes.==0) == sumc(invoutcomes.>-10));
       pr=delta*prodc( (invoutcomes.<=-10)+
          (invmat[rlnfirms*(i-1)+1:rlnfirms*i,1].==0).*
           (a*invmat[rlnfirms*(i-1)+1:rlnfirms*i,2]./
           (1+a*invmat[rlnfirms*(i-1)+1:rlnfirms*i,2])))
          +(1-delta)*prodc( (invoutcomes.<=-10)+
          (invmat[rlnfirms*(i-1)+1:rlnfirms*i,1].==0).*
           (1-(a*invmat[rlnfirms*(i-1)+1:rlnfirms*i,2]./
           (1+a*invmat[rlnfirms*(i-1)+1:rlnfirms*i,2]))));
    else;
      /* if any firm moves down, then we know og moved up */
      if (sumc(invoutcomes.==-1)>0);
         pr=delta*prodc( (invoutcomes.<=-10)+
            (invmat[rlnfirms*(i-1)+1:rlnfirms*i,1].==0).*
             (a*invmat[rlnfirms*(i-1)+1:rlnfirms*i,2]./
             (1+a*invmat[rlnfirms*(i-1)+1:rlnfirms*i,2]))+
            (invmat[rlnfirms*(i-1)+1:rlnfirms*i,1].==-1).*
             (1-(a*invmat[rlnfirms*(i-1)+1:rlnfirms*i,2]./
             (1+a*invmat[rlnfirms*(i-1)+1:rlnfirms*i,2]))));
       /* otherwise we know og stayed put */
      else;
         pr=(1-delta)*prodc( (invoutcomes.<=-10)+
            (invmat[rlnfirms*(i-1)+1:rlnfirms*i,1].==1).*
             (a*invmat[rlnfirms*(i-1)+1:rlnfirms*i,2]./
             (1+a*invmat[rlnfirms*(i-1)+1:rlnfirms*i,2]))+
            (invmat[rlnfirms*(i-1)+1:rlnfirms*i,1].==0).*
             (1-(a*invmat[rlnfirms*(i-1)+1:rlnfirms*i,2]./
             (1+a*invmat[rlnfirms*(i-1)+1:rlnfirms*i,2]))));
      endif;
    endif;       

    endif;  /* endif that cuts out obs at 1 and kmax */

    if pr == 0;
      invmat[rlnfirms*(i-1)+1:rlnfirms*i,.];
    endif;

    likel=likel-ln(pr);

    i=i+1;
  endo;

retp(likel);

endp;


/* Estimate probability of successful investment (objective function above). */
proc (2) = doprinvlogit;
  local b, h, f, g, i, j, retcode, b0;

  b0 = {1.0,0.2};
  _opalgr=2;
  _opgtol=1e-4;
  "Obj Func at starting values";
  invlik(b0);
  {b,f,g,retcode} = optmum( &invlik, b0);
  h = invpd(_opfhess);

  b[1]=abs(b[1]);
  b[2]=exp(b[2])/(1+exp(b[2]));

  retp(b,h);

endp;


/* This function projects investment onto states using local linear regression to
   get investment policy function. */
proc (1) = firststinvll(bandwidth);
  local i, j, k, statemat, nobs, statevec, tempstate, predinvest;

  predinvest=zeros(wmax,rlnfirms);
  tempstate=zeros(rlnfirms,1);

  /* Pick out investment for all observed firms (even if they exit) */
  statemat = selif(invmat[.,2:2+rlnfirms],invmat[.,1].>=-10);
  nobs=rows(statemat);
  "Number of observations in investment regression: " nobs;

  i=1;
  do while i <= wmax;
    statevec = decode(i);
    j=1;
    do while j<=rlnfirms;
      if (statevec[j]>0);
        tempstate[1]=statevec[j];
        k=1;
        do while k <= rlnfirms;
          if k < j;
            tempstate[k+1] = statevec[k];
          elseif k > j;
            tempstate[k] = statevec[k];
          endif;
          k=k+1;
        endo;
        predinvest[i,j] = llreg2(tempstate[1:rlnfirms],bandwidth,statemat[.,2:cols(statemat)],statemat[.,1]);
      endif;
      j=j+1;
    endo;
    i=i+1;
  endo; 

retp(predinvest);

endp;


/* This procedure projects entry onto state variables using a local linear to get
   entry probabilities (policy function).  */
proc (4) = doentryest(bandwidth);
  local i,j,k,firmnumber,labsin,labsout,zout,indstatepoly,entryparams,entryh,entryrsq,tempstate,
        actualentry, predictedentry, maxnormdiff, entryllrg, statevec, entryflag, pentryrsq, statepoly,
        rsqmat, stentryrsq, templabels, w, state, statespace, y, x, wghts, entryobs, n;

  n=rows(entrymat); /* Number of observations */
  tempstate = rev(sortc(entrymat~dstate,1));
  entryobs = maxindc(tempstate[.,1].==-1)-1;
  statevec = zeros(rlnfirms,1);

  rlnfirms=rlnfirms-1;
  statespace=zeros(wmax2,rlnfirms);
  i=1;
  do while i <= wmax2;
    statespace[i,.] = decode(i)';
    i=i+1;
  endo;
  rlnfirms=rlnfirms+1;


  @Do local linear regression 1@
  i=1;
  entryllrg = zeros(wmax,1);
    do while i <= wmax;
    statevec = decode(i);
    if statevec[rlnfirms] == 0;
      entryllrg[i] = llreg2(statevec[1:rlnfirms-1],bandwidth,tempstate[1:entryobs,2:cols(tempstate)-1],tempstate[1:entryobs,1]);
    endif;
    i=i+1;
  endo; 

  /* The rest of this procedure is for data reporting purposes.  It compares actual and predicted and 
     computes some norms and R2's in how well the local linear is working. */

  /* Get max difference between actual and predicted entry prob. */
  actualentry = zeros(wmax2,1);
  predictedentry = zeros(wmax2,1);
  i=1;
  do while i <= wmax2;
    actualentry[i] = isentry[encode(statespace[i,.]~0)];
    predictedentry[i] = entryllrg[encode(statespace[i,.]~0)];
    i=i+1;
  endo;
  predictedentry = predictedentry.*(predictedentry .>= 0 .and predictedentry .<= 1) + 
                   (predictedentry .> 1);
  maxnormdiff = maxc(abs(actualentry-predictedentry));

  /* Get R-squared over all states */
  pentryrsq = zeros(wmax,1);
  i=1;
  do while i <= wmax;
    statevec=decode(i);
    if statevec[rlnfirms] == 0;
      pentryrsq[i] = entryllrg[i]; 
      pentryrsq[i] = pentryrsq[i]*(pentryrsq[i] >= 0 and pentryrsq[i] <= 1) + 
                   (pentryrsq[i] > 1);
    else;
      pentryrsq[i] = -1;
    endif;
    i=i+1;
  endo;
  rsqmat = rev(sortc(pentryrsq~isentry,1));
  i=1;
  j=0;
  do while j == 0;
    if rsqmat[i,1] == -1;
      j=1;
    endif;
    i=i+1;
  endo;
  stentryrsq = 1 - ( sumc( (rsqmat[1:i-2,1] - rsqmat[1:i-2,2]).^2 ) )/
               ( sumc( (rsqmat[1:i-2,2] - (sumc(rsqmat[1:i-2,2])/(i-2)).*ones(i-2,1)).^2) );

  retp(entryllrg, maxnormdiff,stentryrsq,entryobs);

endp;


/* This procedure projects exit onto state variables to get predicted exit policy */
proc (1) = doexitestll(bandwidth);

  /* Get first stage investment function */

  local i, j, k, statemat, nobs, statevec, tempstate, predexit, exited;

  predexit=zeros(wmax,rlnfirms);
  tempstate=zeros(rlnfirms,1);

  /* Pick out investment for all observed firms (even if they exit) */
  statemat = selif(invmat[.,3:2+rlnfirms],invmat[.,1].>=-10);
  exited = selif(invmat[.,1].==-10,invmat[.,1].>=-10);
  nobs=rows(statemat);
  "Number of obs in exit reg: " nobs " of which " sumc(exited) " exited.";

  i=1;
  do while i <= wmax;
    statevec = decode(i);
    j=1;
    do while j<=rlnfirms;
      if (statevec[j]>0);
        tempstate[1]=statevec[j];
        k=1;
        do while k <= rlnfirms;
          if k < j;
            tempstate[k+1] = statevec[k];
          elseif k > j;
            tempstate[k] = statevec[k];
          endif;
          k=k+1;
        endo;
        predexit[i,j] = llreg2(tempstate[1:rlnfirms],bandwidth,statemat,exited);
      endif;
      j=j+1;
    endo;
    i=i+1;
  endo; 

retp(predexit);

endp;


/*******************************************************************************/
/*                                                                             */
/* Procedure to subsample data. (Even though it is called bstrap it actually   */
/* does subsampling.)                                                          */
/*                                                                             */
/*******************************************************************************/


/* This procedure bootstraps new data set */
proc (0) = bstrap;
local bstrapdata, data, row, out, i;

/* Choose starting row for a sequence of length ssn */
row=floor((numtimes-subsamplen+1)*rndu(1,1))+1;
out=seqa(row,1,subsamplen);

  /* Now assign new data matrices */
  invmat = zeros(subsamplen*rlnfirms,2+rlnfirms);
  i=1;
  do while i<=subsamplen;
    invmat[3*(i-1)+1:3*i,.] = invmatorig[3*(out[i]-1)+1:3*out[i],.];
    i=i+1;
  endo;
  entrymat = entrymatorig[out,.];
  dstate = dstateorig[out,.];
  dprice = dpriceorig[out,.];
  dquant = dquantorig[out,.];
  dexit  = dexitorig[out,.];

endp;
