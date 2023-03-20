#include <math.h>
#include <float.h>
#include "../comms.h"
#include "drivers.h"
#include "../kernel_interface.h"
#include "../chunk.h"

void tqli(double* d, double* e, int n);

// Calculates the eigenvalues from cg_alphas and cg_betas
void eigenvalue_driver_initialise(
    Chunk* chunks, Settings* settings, int num_cg_iters)
{
  START_PROFILING(settings->kernel_profile);

  for(int cc = 0; cc < settings->num_chunks_per_rank; ++cc)
  {
    double diag[num_cg_iters];
    double offdiag[num_cg_iters];
    memset(diag, 0, sizeof(diag));
    memset(offdiag, 0, sizeof(offdiag));

    // Prepare matrix
    for(int ii = 0; ii < num_cg_iters; ++ii)
    {
      diag[ii] = 1.0 / chunks[cc].cg_alphas[ii];

      if(ii > 0)
      {
        diag[ii] += chunks[cc].cg_betas[ii-1] / chunks[cc].cg_alphas[ii-1];
      }
      if(ii < num_cg_iters-1)
      {
        offdiag[ii+1] = sqrt(chunks[cc].cg_betas[ii]) / chunks[cc].cg_alphas[ii];
      }
    }

    // Calculate the eigenvalues (ignore eigenvectors)
    tqli(diag, offdiag, num_cg_iters);

    chunks[cc].eigmin = DBL_MAX;
    chunks[cc].eigmax = DBL_MIN;

    // Get minimum and maximum eigenvalues
    for(int ii = 0; ii < num_cg_iters; ++ii)
    {
      chunks[cc].eigmin = MIN(chunks[cc].eigmin, diag[ii]);
      chunks[cc].eigmax = MAX(chunks[cc].eigmax, diag[ii]);
    }

    if(chunks[cc].eigmin < 0.0 || chunks[cc].eigmax < 0.0)
    {
      die(__LINE__, __FILE__, "Calculated negative eigenvalues.\n");
    }

    // TODO: Find out the reasoning behind this!?
    // Adds some buffer for precision maybe?
    chunks[cc].eigmin *= 0.95;
    chunks[cc].eigmax *= 1.05;

    print_and_log(settings, 
        "Min. eigenvalue: \t%.12e\nMax. eigenvalue: \t%.12e\n", 
        chunks[cc].eigmin, chunks[cc].eigmax);
  }

  STOP_PROFILING(settings->kernel_profile, __func__);
}

// Adapted from
// http://ftp.cs.stanford.edu/cs/robotics/scohen/nr/tqli.c
void tqli(double* d, double* e, int n)
{
  int m,l,iter,i;
  double s,r,p,g,f,dd,c,b;

  for (i=0;i<n-1;i++) e[i]=e[i+1];
  e[n-1]=0.0;
  for (l=0;l<n;l++) {
    iter=0;
    do {
      for (m=l;m<n-1;m++) {
        dd=fabs(d[m])+fabs(d[m+1]);
        if (fabs(e[m])+dd == dd) break;
      }

      if (m == l) break;

      if (iter++ == 30){
        die(__LINE__, __FILE__,
            "Too many iterations in TQLI routine\n");
      }
      g=(d[l+1]-d[l])/(2.0*e[l]);
      r=sqrt((g*g)+1.0);
      g=d[m]-d[l]+e[l]/(g+sign(r,g));
      s=c=1.0;
      p=0.0;
      for (i=m-1;i>=l;i--) {
        f=s*e[i];
        b=c*e[i];
        r=sqrt(f*f+g*g);
        e[i+1]=r;
        if(r == 0.0)
        {
          d[i+1]-=p;
          e[m]=0.0;
          continue;
        }
        s=f/r;
        c=g/r;
        g=d[i+1]-p;
        r=(d[i]-g)*s+2.0*c*b;
        p=s*r;
        d[i+1]=g+p;
        g=c*r-b;
      }
      d[l]=d[l]-p;
      e[l]=g;
      e[m]=0.0;
    } while (m != l);
  }
}
