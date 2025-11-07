#define rlnfirms 3
#define kmax 20
#define entry_k 7
#define NUMTHREADS 1
#define MAXTHREADS 16

#define _REENTRANT


#include <math.h>
#include <pthread.h>
#include <stdio.h>

struct thrpass {
   int start,end;
   double *investdraws,*outsidedraws,*entrydraws,*startingstate,
          *startme,*altinvest,*altexit,*nsimin,*nperin,*predinvest,
          *invparams,*predexit,*wret,*newprofit,*betadp,*entryllrg,*binom;
};

int encode();
int bubblesort2();

void thrpass_init(thr,start,end,
          investdraws,outsidedraws,entrydraws,startingstate,startme,
          altinvest,altexit,nsimin,nperin,predinvest,invparams,predexit,
          wret,newprofit,betadp,entryllrg,binom)
struct thrpass *thr;
int start,end;
double *investdraws,*outsidedraws,*entrydraws,*startingstate,
          *startme,*altinvest,*altexit,*nsimin,*nperin,*predinvest,
          *invparams,*predexit,*wret,*newprofit,*betadp,*entryllrg,*binom;
{
thr->start=start;
thr->end=end;
thr->investdraws=investdraws;
thr->outsidedraws=outsidedraws;
thr->entrydraws=entrydraws;
thr->startingstate=startingstate;
thr->startme=startme;
thr->altinvest=altinvest;
thr->altexit=altexit;
thr->nsimin=nsimin;
thr->nperin=nperin;
thr->predinvest=predinvest;
thr->invparams=invparams;
thr->predexit=predexit;
thr->wret=wret;
thr->newprofit=newprofit;
thr->betadp=betadp;
thr->entryllrg=entryllrg;
thr->binom=binom;
}

void thrpass_getarg(arg,start,end,
          investdraws,outsidedraws,entrydraws,startingstate,startme,
          altinvest,altexit,nsimin,nperin,predinvest,invparams,predexit,
          wret,newprofit,betadp,entryllrg,binom)
void *arg;
int *start,*end;
double **investdraws,**outsidedraws,**entrydraws,**startingstate,
          **startme,**altinvest,**altexit,**nsimin,**nperin,**predinvest,
          **invparams,**predexit,**wret,**newprofit,**betadp,**entryllrg,**binom;
{
struct thrpass *thr=arg;

*start=thr->start;
*end=thr->end;
*investdraws=thr->investdraws;
*outsidedraws=thr->outsidedraws;
*entrydraws=thr->entrydraws;
*startingstate=thr->startingstate;
*startme=thr->startme;
*altinvest=thr->altinvest;
*altexit=thr->altexit;
*nsimin=thr->nsimin;
*nperin=thr->nperin;
*predinvest=thr->predinvest;
*invparams=thr->invparams;
*predexit=thr->predexit;
*wret=thr->wret;
*newprofit=thr->newprofit;
*betadp=thr->betadp;
*entryllrg=thr->entryllrg;
*binom=thr->binom;
}





void *wsimbody(void *arg)
{

  /* Function Arguments */
  int start,end;
  double *investdraws,*outsidedraws,*entrydraws,*startingstate,
         *startme,*altinvest,*altexit,*nsimin,*nperin,*predinvest,
         *invparams,*predexit,*wret,*newprofit,*betadp,*entryllrg,*binom;

  /* Local vars */
  int i, j, s, policy, me, og, exited;
  int nsim,nper;
  int w, w2, wthis[rlnfirms], wtrans[rlnfirms], wentry[rlnfirms];
  double xx[rlnfirms];

  thrpass_getarg(arg,&start,&end,
	&investdraws,&outsidedraws,&entrydraws,&startingstate,&startme,
	&altinvest,&altexit,&nsimin,&nperin,&predinvest,&invparams,&predexit,
        &wret,&newprofit,&betadp,&entryllrg,&binom);

  nsim=(int)nsimin[0];
  nper=(int)nperin[0];

  for(s=start;s<end;s++) {

  /* do this twice, once for est and once for alt policy */
  policy=0;
  while (policy<=1) {

    /* starting state */
    me=(int)startme[0]-1;
    for(i=0;i<rlnfirms;i++) wthis[i]=(int)startingstate[i];

    for(i=0;i<nper;i++) {
   
      if (me>-1)  {  /* firm exited */  

        /* Assign wentry */
        for(j=0;j<rlnfirms;j++) wentry[j]=wthis[j];
        exited=0;

        w=encode(wthis,rlnfirms,binom);

        /* Find predicted investment */
        for(j=0;j<rlnfirms;j++) xx[j]=predinvest[w*rlnfirms+j];
        xx[me]+=policy*altinvest[0];
        for(j=0;j<rlnfirms;j++) 
          if (xx[j]<0.0) xx[j]=0.0;

        /* Implement investment and exit draws */
	og=(outsidedraws[s*nper+i]<invparams[1]);
        for(j=0;j<rlnfirms;j++) {

          if (wthis[j]) {

            /* investment */
            wtrans[j]=wthis[j]+(investdraws[(s*nper+i)*rlnfirms+j] < 
	  	     (invparams[0]*xx[j])/(1+invparams[0]*xx[j]))-og;
            if (wtrans[j]>kmax) wtrans[j]=kmax;
            if (wtrans[j]<1 && wthis[j]==1) wtrans[j]=1;
          
            /* exit */
            if (predexit[w*rlnfirms+j]+altexit[0]*policy*(j==me) > 0.5 || exited==1) {
	       wtrans[j]=0;
               wentry[j]=0;
               exited=1;
               if (j==me) me=-1;
            }

          } /* if (wthis) */
          else wtrans[j]=0;

        }  /* for (j) */

        /* Assign profits */
        if (policy==0) 
	  if (me>-1) {
  	    wret[s*rlnfirms+0]+=newprofit[w*rlnfirms+me]*betadp[i];
            wret[s*rlnfirms+1]+=xx[me]*betadp[i];
          }
          else wret[s*rlnfirms+2]+=betadp[i];
        else 
	  if (me>-1) {
  	    wret[s*rlnfirms+0]-=newprofit[w*rlnfirms+me]*betadp[i];
            wret[s*rlnfirms+1]-=xx[me]*betadp[i];
          }
          else wret[s*rlnfirms+2]-=betadp[i];

        /* Now do entry */
        if (me>-1 && wtrans[rlnfirms-1]==0) {
	     w2=encode(wentry,rlnfirms,binom);
             if (entrydraws[s*nper+i]<entryllrg[w2])
  	        wtrans[rlnfirms-1] = entry_k-og;
           }

	/* Now reorder the state */
        if (me>-1) {
          for(j=0;j<rlnfirms;j++) wthis[j]=wtrans[j];
	  me=bubblesort2(wthis,0,rlnfirms-1,me);
        }


      }  /* if (me>-1) */

    }  /* for (i) */

    policy +=1;

  }  /* while policy<=1 */

} /* for(s) */

}  /* procedure */




void wsim(investdraws,outsidedraws,entrydraws,startingstate,startme,
          altinvest,altexit,nsimin,nperin,predinvest,invparams,predexit,
          wret,newprofit,betadp,entryllrg,binom)
     double *investdraws,*outsidedraws,*entrydraws,*startingstate,
            *startme,*altinvest,*altexit,*nsimin,*nperin,*predinvest,
            *invparams,*predexit,*wret,*newprofit,*betadp,*entryllrg,*binom;
{
int i,numthreads;
int nsim=(int)nsimin[0];
struct thrpass in[MAXTHREADS];
pthread_t thrs[MAXTHREADS];
pthread_attr_t attr;

numthreads=NUMTHREADS;

/* Initialize thread attributes to be SYSTEM wide:
   This is unnecessary in Linux since it is the default, but in Solaris 
   the default is PTHREAD_SCOPE_PROCESS. */
pthread_attr_init(&attr);
pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);  

/* Create n-1 threads each doing one part of the state space */
for(i=0;i<numthreads-1;i++) {
  thrpass_init(&in[i],i*(nsim/numthreads),(i+1)*nsim/numthreads,
          investdraws,outsidedraws,entrydraws,startingstate,startme,
          altinvest,altexit,nsimin,nperin,predinvest,invparams,predexit,
          wret,newprofit,betadp,entryllrg,binom);
  pthread_create(&thrs[i],&attr,wsimbody,&in[i]);
}


/* nth part of state space is done in last thread -- end is set to wmax to avoid
   integer rounding problems */
thrpass_init(&in[i],i*(nsim/numthreads),nsim,
          investdraws,outsidedraws,entrydraws,startingstate,startme,
          altinvest,altexit,nsimin,nperin,predinvest,invparams,predexit,
          wret,newprofit,betadp,entryllrg,binom);
pthread_create(&thrs[i],&attr,wsimbody,&in[i]);

/* Join all threads when done */
for(i=0;i<numthreads;i++) pthread_join(thrs[i],NULL);

}  /* procedure */


int encode(ntuple,nfirms,binom)
/* Code a weakly descending n-tuple on a number from 0 to wmax-1 */
int *ntuple, nfirms;
double *binom;
{ int i,out;
  double code=0.0;
    
  for(i=0;i<nfirms;i++)
    code += binom[(ntuple[i]+nfirms-i-1)*(rlnfirms+kmax+2) + ntuple[i]];
  
  out=(int)code;
  return code;
}




int bubblesort2(data,start,finish,place)

/* This algorithm uses bubblesort and also indicates the new place of the
placeth element */

int start,finish,*data,place;
{
int i,j,temp,switched=1,*p;

for(i=finish;(i>start)&&switched;i--)
  for(switched=0,j=start,p=&data[j];j<i;j++,p++)
    if (*p<*(p+1)) {
       switched++;
       temp=*p;
       *p=*(p+1);
       *(p+1)=temp;
       if (j==place) place++; 
       else if ((j+1)==place) place--;
    }
return place;
}


