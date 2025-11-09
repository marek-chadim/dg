@ this version of amoeba was originally adapted from Numerical
  Recipes by D. Mark Kennet @

@ 
  you might want to check out the "test for done" and modify it for
  your problem ftol is the tolerance for the simplex of points
  and ytol is the tolerance for the function values 
@


proc gmamoeba(p,&f);               @ Procedure takes matrix p as input,  @
                                 @ where p is an initial NDIMx1 point  @
                                 @ augmented with a matrix of vertices @
                                 @ so that p defines an n-dimensional  @
                                 @ simplex.  Y contains the function   @
                                 @ values for the n+1 vertices for the @
                                 @ function f which is to be minimized @
                                 @ to a tolerance ftol.                @
                                 @ The procedure returns the simplex   @
                                 @ containing the minimizer vertically @
                                 @ concatenated with the vector of     @
                                 @ function values above it; i.e.,     @
                                 @ y'|p.                               @

/* MUST SET ftol and itmax and amoebaprn before running!!! */

         local nmax,aleph,bet,gimel,mpts,ilo,ihi,inhi,tol,
                 pbar,pr,ypr,yprr,prr,iter,ndim,i,ytemp,mp,y,tol2;

         local f:proc;

         nmax=90; aleph=1.0; bet=0.5; gimel=2.0;

	 ndim=rows(p);
	 mpts=ndim+1;
	 y=zeros(mpts,1);
	 i=1;
	 do while i <= mpts;
	   y[i] = f(p[.,i]);
	   i=i+1;
	 endo;

         "****BEGIN NELDER & MEAD MINIMIZATION****";
         "ftol: " ftol " ftol2: " ftol2;
         iter=0;

do until iter>itmax;             @ BEGIN MAIN LOOP. @

@----------------------------------------------------------------------@
@
         The following finds the point with the highest function value,
         the next highest, and the lowest.
@

         ilo=minindc(y); ihi=maxindc(y);
         ytemp=miss(y,maxc(y));
         inhi=maxindc( ytemp );

         tol = maxc(maxc(abs(p[.,ilo]-p))) ;
         tol2= maxc(maxc(abs(p[.,ilo]-p)./(abs(p[.,ilo])+1e-4)));

@----------------------------------------------------------------------@
@
         Print current results.
@
/*         format 8,4 ; 
         " " ; "Simplex/Fvec " ;
         p ; " " ; y' ; " " ; */
         if ((amoebaprn > 0) and ((iter % amoebaprn) eq 0));
         "Amoeba Iteration: ";; format /rd 3,0; iter;
         "***Low function value: ";; format 10,8 ; minc(y);
         format 6,5 ; "Low estimate:  ";; (p[.,ilo])';
         "Current tolerances (tol/tol2): ";; tol~tol2; " " ;
         save simplex2=p;
         endif;

@----------------------------------------------------------------------@

@
         TEST FOR DONE:
@

         if ((tol<ftol) or (tol2<ftol2)) ;
           retp( p[.,ilo] );
           iter=itmax;
         endif;  @ Convergence.    @

         if iter==itmax;
           save fvec=y;             @ Save values @
           save simplex=p;
           print "AMOEBA exceeding maximum iterations. ";
           retp( p[.,ilo] );
           iter=itmax;
         endif;
         iter=iter+1;

@----------------------------------------------------------------------@
@
         The following computes the vector average of all vertices
         in the simplex except the one with the high function value.
@
         if ihi/=1 and ihi/=mpts;
                 pbar=meanc( (p[.,1:ihi-1]~p[.,ihi+1:mpts])' );
         elseif ihi==1;
                 pbar=meanc( p[.,2:mpts]' );
         elseif ihi==mpts;
                 pbar=meanc( p[.,1:mpts-1]' );
         endif;

@----------------------------------------------------------------------@

         pr = (1+aleph)*pbar - aleph*p[.,ihi];    @ Reflect high point @
                                                  @ through average.   @

         ypr=f(pr);                               @ Evaluate new point.@

         if ypr<=y[ilo,1];                        @ If new point is the@
                 prr=gimel*pr + (1-gimel)*pbar;   @ best so far, try to@
                 yprr=f(prr);                     @ extend reflection. @

                 if ( yprr < minc(y[ilo,1]|ypr) ); @ Extension succeeds,@
                         p[.,ihi]=prr;             @ so replace high    @
                         y[ihi,1]=yprr;            @ point with new one.@

                 else;                            @ Extension fails,   @
                         p[.,ihi]=pr;             @ but can still use  @
                         y[ihi,1]=ypr;            @ reflected point.   @

                 endif;

         elseif ypr>=y[inhi,1];                   @ If reflected point is   @
                                                  @ worse than 2nd highest  @
                 if ypr<y[ihi,1];                 @ but better than highest,@
                         p[.,ihi]=pr;             @ replace highest...      @
                         y[ihi,1]=ypr;
                 endif;
                                                  @...but look for better.  @
                 prr=bet*p[.,ihi] + (1-bet)*pbar;
                 yprr=f(prr);

                 if yprr<y[ihi,1];                @ Contraction improves.   @
                         p[.,ihi]=prr;
                         y[ihi,1]=yprr;

                 else ;                           @ Can't get rid of high pt@
                         i=1;                     @ so contract the simplex @
                         do until i>mpts;
                            if i/=ilo;
                                 pr=0.5*(p[.,i]+p[.,ilo]);
                                 y[i,1]=f(pr);
                                 p[.,i] = pr ;
                            endif;
                            i=i+1;
                         endo;
                 endif;

         else;                                    @ Arrive here with   @
                 p[.,ihi]=pr;                     @ middling point.    @
                 y[ihi,1]=ypr;
         endif;


endo;
endp;


