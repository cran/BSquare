#include "math.h"
#include <stdlib.h>
#include "R.h"
#include "Rmath.h"
#include <Rinternals.h>
#include <R_ext/Applic.h>
#include <R_ext/Lapack.h>
#include <assert.h>
#include <stdio.h>
#include <R_ext/Utils.h>
#include <R_ext/Lapack.h>

#include "helpers_np.h"

/************************** matrix multiplication *************************


function that returns C = alpha * A * B + beta * C


TRANSA = indicator if A should be transposed
    'N', 'n' indicate no
    'T', 't', 'C', 'c' indicate yes
TRANSB = should B be tranposed
    'N', 'n' indicate no
    'T', 't', 'C', 'c' indicate yes
M = nrow(C) = nrow(A) or ncol(A')
N = ncol(C) = ncol(B) or nrow(B')
K = ncol(A) or nrow(A') = nrow(B) or ncol(B')
alpha is scalar
A is double precision array of dimension (LDA, ka),
    where ka = k if TRANSA = 'N' or 'n'
             = m otherwise
LDA informs BLAS how large a "stride" to take w/r/t A
    C language is row major so stride is the number of columns in A
    if TRANSA = 'N' or 'n' then LDA must be at least max(1,M)
    o/w LDA must be at least max(1,k)
B is double precision array of dimension (LDB, kb),
    kb = n if TRANSB = 'N' or 'n'
       = k o/w
LDB informs BLAS how large a "stride" to take w/r/t B
    C language is row major so stride is the number of columns in B
    if TRANSB = 'N' or 'n' then LDB must be at least max(1,n)
    o/w LDA must be at least max(1,k)
beta is scalar
C is double precision array of dimension (LDC,n)
LDC informs BLAS how large a stride to take w/r/t matrix C
    must be no smaller than max(1,m)

*/
void MM (int *T_A, int *T_B, int *M, int *N, int *K, double *alpha, double *A, double *B, double *beta, double *C){
    int LDA = *M;
    char TRANSA = 'N';
    if (*T_A == 1){
        LDA = *K;
        TRANSA = 'T';
        }
    int LDB = *K;
    char TRANSB = 'N';
    if (*T_B == 1){
        LDB = *N;
        TRANSB = 'T';
        }
    int LDC = *M;
    F77_CALL(dgemm)(&TRANSA, &TRANSB, M, N, K, alpha, A, &LDA, B, &LDB, beta, C, &LDC);
}

void dot_product(int *D_vector, double *v1, double *v2, double *dp){
    int p;
    *dp = 0;
    for(p = 0; p < *D_vector; p++){*dp += v1[p] * v2[p];}
}

/*function that creates a G x G AR-1 precision matrix with correlation parameter rho*/
void make_prec(double *rho, int *G, double *OMEGA){
    int g;
    double rho2 = *rho**rho;
    double a = 1/(1-rho2);
    double b =     1+2*rho2/(1-rho2);
    double c = rho2/(1-2*rho2 +pow(rho2,2));

    for(g = 0; g < (*G-1);g++){
        OMEGA[g + *G*g]= b;
        OMEGA[g +*G*(g+1)]= c;
    OMEGA[g+1 +*G*g]=c;
    }
  OMEGA[0] = a;
  OMEGA[*G * *G -1] = a;
}

void chol2inv(int *G, double *OMEGA, double *OMEGA1){
    int g, l, info;
    for(g = 0; g < *G; g++){
        for(l = 0; l < *G; l++){
            OMEGA1[g+*G*l] = OMEGA[g+*G*l];
        }
    }
    char uplo = 'L';
	F77_CALL(dpotrf)(&uplo, G, OMEGA1, G, &info);
	if (info) {
		Rprintf("Error with chol(OMEGA): info = %d\n", info);
	}
	// complete inverse
	F77_CALL(dpotri)(&uplo, G, OMEGA1, G, &info);
	if (info) {
		Rprintf("Error with inv(chol(OMEGA)): info = %d\n", info);
	}
}
/***********************************************
thanks to Ryan Parker for teaching my Fortran calls
************************************************/
/*function that returns the log of the determinant of a symmetric positive definite matrix*/
void symm_log_det(int*G, double *OMEGA, double *logdet){
    int g, l, info;
    *logdet = 0;
    double G2 = *G**G;
    double *OMEGA1 = (double *)Calloc(G2, double);

    for (int g = 0; g < *G; g++){
        for(l = 0; l < *G; l++){
            OMEGA1[g+*G*l] = OMEGA[g+*G*l];
            }
        }
    char uplo = 'L';
	// compute chol(OMEGA)
	F77_CALL(dpotrf)(&uplo, G, OMEGA1, G, &info);
	if (info) {
		Rprintf("rmvnorm: Error with chol(sigma1): info = %d\n", info);
	}
    for(g = 0; g < *G; g++){
        *logdet +=   log(OMEGA1[g + g**G]);
        }
    *logdet = 2**logdet;
 	Free(OMEGA1);
}

void sum_theta_IP(double *rho, int *M, int *P, int *p, double *theta, double *sumtheta, double *sumprec){
    int m;
    double rho2 = *rho * *rho;
    double a = 1/(1-rho2);
    double b =     1+2*rho2/(1-rho2);
    double c = -*rho/(1-rho2);
    *sumprec = (2 * 1/(1-rho2)+((*M - 1)-2)*(1+2*rho2/(1-rho2))+2*((*M - 1)-1)*(-*rho/(1-rho2)));
    *sumtheta = 0;
    /*  sum the m = 2 to m = (M - 2) cases*/
    for(m = 2; m < *M - 1; m++){
        *sumtheta += theta[m + *M * *p];
    }
    *sumtheta *= (b + 2 * c);
    /*sum the m = 1 and m = M-1 cases*/
    *sumtheta += (a + c) * (theta[1 + *M * *p] + theta[(*M - 1) + *M * *p]);
    }

 void rho_quad_IP(double *rho, int *M, int *P, int *p, double *theta, double *mu, double *rhoquad){
    int m;
    double rho2 = *rho * *rho;
    double a = 1 / (1 - rho2);
    double b =     1 + 2 * rho2 / (1 - rho2);
    double c = -*rho / (1 - rho2);
    double x[(*M - 1)];
    *rhoquad = 0;
    for(m = 1; m < *M; m++){
        x[m - 1] = theta[m + *M * *p] - mu[*p];
        }
    for(m = 0; m < (*M - 2); m++){
        *rhoquad += x[m] * x[m+1];
    }
    *rhoquad *= (2 * c);
    for(m = 1; m < (*M - 2); m++){
        *rhoquad += b * x[m] * x[m];
    }
    *rhoquad += a * (x[0] * x[0] + x[*M-2] * x[*M-2]);
    }

void rho_quad(double *rho, int *M, int *G, int *P, int *m, int *p, double *theta, double *mu, double *rhoquad){
    int g;
    double rho2 = *rho * *rho;
    double a = 1/(1-rho2);
    double b =     1+2*rho2/(1-rho2);
    double c = -*rho/(1-rho2);
    double x[*G];
    *rhoquad = 0;
    for(g = 0; g < *G; g++){
        x[g]=theta[*m + *M * g + *M * *G * *p]-mu[*m + *M * *p];
        }
    for(g = 0; g < *G-1; g++){
        *rhoquad += x[g] * x[g+1];
    }
    *rhoquad *= (2 * c);
    for(g = 1; g < (*G - 1); g++){
        *rhoquad += b * x[g] * x[g];
    }
    *rhoquad = *rhoquad + a * (x[0] * x[0] + x[*G-1] * x[*G-1]);
}
void sum_theta(double *rho, int *M, int *G, int *P, int *m, int *p, double *theta, double *sumtheta, double *sumprec){
    int g;

    double rho2 = *rho * *rho;
    double a = 1/(1-rho2);
    double b =     1+2*rho2/(1-rho2);
    double c = -*rho/(1-rho2);
    *sumprec = (2*1/(1-rho2)+(*G-2)*(1+2*rho2/(1-rho2))+2*(*G-1)*(-*rho/(1-rho2)));
    *sumtheta = 0;
    for(g = 1; g < *G-1; g++){
        *sumtheta += theta[*m + *M * g +*M * *G * *p];
    }
    *sumtheta *= (b+2*c);
    *sumtheta += (a+c)*(theta[*m + *M * 0  + *M * *G * *p] + theta[*m + *M * (*G-1)+*M * *G * *p]);
    }

/*use for preds that don't change across g in CP*/
void sum_theta_constant(double *rho, int *M, int *G, int *P, int *g, int *p, double *theta, double *sumtheta, double *sumprec){
    int m;
    double rho2 = *rho * *rho;
    double a = 1/(1-rho2);
    double b =     1+2*rho2/(1-rho2);
    double c = -*rho/(1-rho2);
    *sumprec = (2 * 1/(1-rho2)+((*M - 1)-2)*(1+2*rho2/(1-rho2))+2*((*M - 1)-1)*(-*rho/(1-rho2)));
    *sumtheta = 0;
    /*  sum the m = 2 to m = (M - 2) cases*/
    for(m = 2; m < *M - 1; m++){
        *sumtheta += theta[m + *M * *g + *M * *G * *p];
    }
    *sumtheta *= (b + 2 * c);
    /*sum the m = 1 and m = M-1 cases*/
    *sumtheta += (a + c) * (theta[1 + *M * *g  + *M * *G * *p] + theta[(*M - 1) + *M * *g + *M * *G * *p]);
}

/*use for preds that don't change across g in CP*/
void rho_quad_constant(double *rho, int *M, int *G, int *P, int *g, int *p, double *theta, double *mu, double *rhoquad){
    int m;
    double rho2 = *rho * *rho;
    double a = 1/(1-rho2);
    double b =     1+2*rho2/(1-rho2);
    double c = -*rho/(1-rho2);
    double x[(*M - 1)];
    *rhoquad = 0;
    for(m = 1; m < *M; m++){
        x[m-1] = theta[m + *M * *g + *M * *G * *p] - mu[*g + *G * *p];
        }
    for(m = 0; m < (*M - 2); m++){
        *rhoquad += x[m] * x[m+1];
    }
    *rhoquad *= (2 * c);
    for(m = 1; m < (*M - 2); m++){
        *rhoquad += b * x[m] * x[m];
    }
    *rhoquad += a * (x[0] * x[0] + x[*M-2] * x[*M-2]);
}

void log_det_AR(double *rho, int *G, double *logdet){
    int g;
    double rho2 = *rho * *rho;
    double a = 1 / (1 - rho2);
    double b =     1 + 2 * rho2/(1 - rho2);
    double c2 = rho2 / (1 - 2 * rho2 + pow(rho2,2));
    double x[*G - 1]; /*go through recursion G -1 times*/

    x[0] = a;
    x[1] = b * a - c2;
    for(g = 2; g < *G - 1; g++){
        x[g]= b * x[g-1] - c2 * x[g-2];
        }
    *logdet = -1 * log(a * x[*G - 2] - c2 * x[*G - 3]);
}


/***********************************************
this function is based on the source code for the
"findInterval" function in R
************************************************/

void find_int(int *n, double *xt, double *tau, int *bin) {
    int istep;

    int middle = 0;
    int ihi = *bin + 1;

    if (*tau <  xt[ihi]) {
        if (*tau>=  xt[* bin]) {} /* `lucky': same interval as last time */
        else  {
 /* **** now *x< * xt[* bin] . decrease * bin to capture *x*/
            for(istep = 1; ; istep *= 2) {
            ihi = * bin;
            * bin = ihi - istep;
            if (* bin <= 1)
            break;
            if (*tau>=  xt[* bin]) goto L50;
            }
        * bin = 1;
        }
    }
    else {
  /* **** now *tau>= * xt[ihi] . increase ihi to capture *x*/
    for(istep = 1; ; istep *= 2) {
        * bin = ihi;
        ihi = * bin + istep;
        if (ihi >= *n)
            break;
        if (*tau< xt[ihi]) goto L50;
    }
    ihi = *n;
    }
    L50:
 /* **** now * xt[* bin] <= *x< * xt[ihi] . narrow the interval. */
    while(middle != *bin) {
    /* note. it is assumed that middle = * bin in case ihi = * bin+1 . */
        if (*tau>= xt[middle])
        * bin = middle;
        else
        ihi = middle;
        middle = (* bin + ihi) / 2;
    }
}

double ispline3(double tau, int spline_df, int  m, double  I_knots[], int bin, double IKM[], int M_1){
    double v = 0; /*if bin < m*/
    if ((bin - spline_df + 1) > m){v = 1;}
    else if (bin == m){v = IKM[m] * pow((tau - I_knots[m]),3);}
    else if(bin == (m + 1)){
        v = IKM[m + M_1] * (tau - I_knots[m]) + IKM[m + 2 * M_1] * (pow(tau - I_knots[m+1],2) - pow(I_knots[m+3] - tau,2)) + IKM[m + 3 * M_1] * pow(I_knots[m+3] - tau, 3) + IKM[m + 4 * M_1] * pow(tau-I_knots[m],3);
    }
    else if(bin == (m + 2)){v = 1 + IKM[m + 5 * M_1] * pow(I_knots[m + 3] - tau,3);}
    return(v);
}

double mspline3(double tau, int spline_df, int  m, double  I_knots[], int bin, double MKM[], int M_1){
    double v = 0;
    if (bin == m){v = MKM[m] * pow(tau - I_knots[m],2);}
    else if(bin == (m+1)){v = MKM[m + M_1] + MKM[m + 3 * M_1] * pow(tau - I_knots[m+3], 2) + MKM[m + 4 * M_1] * pow(tau - I_knots[m],2);}
    else if(bin == (m+2)){v = MKM[m + 5 * M_1] * pow(tau - I_knots[m + 3], 2);}
    return(v);
}

double Q_3 (int M, double tau, double w[], int spline_df, double I_knots[], int bin, double IKM[], int M_1){
    int m;
    double r_Q = w[0];
    for(m = 0; m < (M - 1); m++){
        r_Q += w[m + 1] * ispline3(tau, spline_df, m, I_knots, bin, IKM, M_1);
                            }
    return r_Q;
}

double q_3 (int M, double tau, double w[], int  spline_df, double  I_knots[], int bin, double MKM[], int M_1){
    int m;
    double r_q = 0;
    for(m = 0; m < (M - 1); m++){
        r_q += w[m + 1] * mspline3(tau, spline_df, m, I_knots, bin, MKM, M_1);
                        }
    return r_q;
}

void rootfind_GPU (int *M_knots_length, int *I_knots_length, int *M, int *spline_df, double *tau_scalar, double *w,
               double *y_scalar,  double  *M_knots, double  *I_knots, int *bin, double *q_low, double *q_high,
               double *IKM, double *MKM, int *M_1, double *reset_value){
                   if(*tau_scalar < *q_low){*tau_scalar = 0.1;}
                   else if(*tau_scalar > *q_high){*tau_scalar = 0.9;}
        find_int( I_knots_length, I_knots, tau_scalar, bin);
        double f =        Q_3 (*M, *tau_scalar, w, *spline_df, I_knots, *bin, IKM, *M_1);
        double f_prime =  q_3 (*M, *tau_scalar, w, *spline_df, I_knots, *bin, MKM, *M_1);
        int iter = 0;
        int index = 0;
            while(fabsf(*y_scalar - f) > 0.00001 && iter < 1000 && index < 29){
                iter++;
                *tau_scalar += (*y_scalar - f)/f_prime;
                if(*tau_scalar < 0 || *tau_scalar > 1){
                    *tau_scalar =   reset_value[index];
                    index ++;
                }
                find_int(I_knots_length, I_knots,  tau_scalar, bin);
                f = Q_3(*M, *tau_scalar, w, *spline_df, I_knots, *bin, IKM, *M_1);
                f_prime =  q_3(*M, *tau_scalar, w, *spline_df, I_knots, *bin, MKM, *M_1);
            }
            if(index == 29 || iter == 1000){
                    *tau_scalar = 0.5;
            }

}
/*log densities*/
void exponential_low(double *sigma, double *xi, double *v, double *q_level, double *y_scalar){
    *y_scalar =  log(*q_level) - log(*sigma) - (*v / *sigma);
}

void exponential_high(double *sigma, double *xi, double *v, double *q_level, double *y_scalar){
    *y_scalar =   log(1 - *q_level) - log(*sigma) - (*v / *sigma);
}

void pareto_low(double *sigma, double *xi, double *v, double *q_level, double *y_scalar){
    if(*xi < pow(10,-10)){*xi = pow(10,-10);}
    *y_scalar =   (log(*q_level) - log(*sigma) - (1 / *xi + 1) * log(1 + *xi * *v/ *sigma));
}

void pareto_high(double *sigma, double *xi, double *v, double *q_level, double *y_scalar){
    if(*xi < pow(10,-10)){*xi = pow(10,-10);}
    *y_scalar = (log(1 - *q_level) - log(*sigma) -(1 / *xi + 1) * log(1 + *xi * *v / *sigma));
}

/*quantile functions*/

void exponential_tau_low(double *sigma, double *xi, double *v, double *q_level, double *tau_scalar){
    *tau_scalar =    (*q_level - *q_level * (1- exp(-*v / *sigma)));
}

void exponential_tau_high(double *sigma, double *xi, double *v, double *q_level, double *tau_scalar){
    *tau_scalar =    (*q_level+ (1 - *q_level) * (1- exp(-*v / *sigma)));
}

void pareto_tau_low(double *sigma, double *xi, double *v, double *q_level, double *tau_scalar){
    if(*xi < pow(10,-10)){*xi = pow(10,-10);}
    *tau_scalar =    (*q_level- *q_level * ( 1- pow(1+*xi* *v / *sigma, -1 / *xi)));
}

void pareto_tau_high(double *sigma, double *xi, double *v, double *q_level, double *tau_scalar){
    if(*xi < pow(10,-10)){*xi = pow(10,-10);}
    *tau_scalar =    (*q_level) + (1-*q_level) * ( 1- pow(1 + *xi * *v / *sigma, -1 / *xi));
}

void ll (int *M, int *P,
        int *N, int *n1, int *n2, int *n3, int *n4, int *n5,
        double *X, double *y, double *y_low, double *y_high,
        int *M_knots_length, int *I_knots_length, int *spline_df, double *M_knots, double *I_knots,
        double *M_low, double *M_high, double *I_low, double *I_high,
        double *q_low, double *q_high, int *xi_zero,
        double *theta,double *xi_low, double *xi_high,
        double *tau, double *tau_low, double *tau_high, double *log_like, double *ll_sum,
        int *bin, int *bin_low, int *bin_high,
        void ld_low_ptr(double *, double *, double *, double *, double*), /*pointer to the log density for low obs*/
        void ld_high_ptr(double *, double *, double *, double *, double*), /*pointer to the log density for high obs*/
        void tau_low_ptr(double *, double *, double *, double *, double*), /*pointer to the cdf*/
        void tau_high_ptr(double *, double *, double *, double *, double*),
        double *IKM, double *MKM, int *M_1, double *reset_value
        ){
        int k, m, p;
        double low_thresh, high_thresh, sig_low, sig_high, v;
        double w[*M];
        *ll_sum = 0;

        if(*n1 > 0){/*uncensored data*/
            for (k = 0; k < *n1; k++){
                low_thresh = 0;
                high_thresh = 0;
                sig_low = 0;
                sig_high = 0;
                for(m = 0; m < *M; m++){
                    w[m] = theta[m];
                    for(p = 1; p < *P; p++){
                        w[m] += X[k + *N * p] * theta[m + *M * p];
                    }
                    low_thresh += w[m] * I_low[m];
                    high_thresh += w[m] * I_high[m];
                    sig_low += w[m] * M_low[m];
                    sig_high += w[m] * M_high[m];
                }
                if(y[k] < low_thresh){
                    v = low_thresh - y[k];
                    ld_low_ptr(&sig_low, xi_low, &v, q_low, &log_like[k]);
                    tau_low_ptr(&sig_low, xi_low, &v, q_low, &tau[k]);
                }
                else if(y[k] > high_thresh){
                        v = y[k] - high_thresh;
                        ld_high_ptr(&sig_high, xi_high, &v, q_high, &log_like[k]);
                        tau_high_ptr(&sig_high, xi_high, &v, q_high, &tau[k]);
                }
                else{
                    rootfind_GPU (M_knots_length, I_knots_length, M, spline_df, &tau[k], w, &y[k],  M_knots, I_knots, &bin[k], q_low, q_high, IKM, MKM, M_1, reset_value); /*pass to void*/
                    log_like[k] = -1 * log(q_3 (*M, tau[k], w, *spline_df, I_knots, bin[k], MKM, *M_1));
                }
                *ll_sum += log_like[k];
            }
        }
        if(*n2 > 0){ /*obsevations censored below*/
            for(k = *n1; k < (*n1 + *n2); k++){
                low_thresh = 0;
                high_thresh = 0;
                sig_low = 0;
                sig_high = 0;
                for(m = 0; m < *M; m++){
                    w[m] = theta[m]; /*create weight vector*/
                    for(p = 1; p < *P; p++){
                        w[m] += X[k + *N * p] * theta[m + *M * p];
                    }
                    low_thresh += w[m] * I_low[m];
                    high_thresh += w[m] * I_high[m];
                    sig_low += w[m] * M_low[m];
                    sig_high += w[m] * M_high[m];
                }
                if(y_high[k] < low_thresh){
                    v = low_thresh - y_high[k];
                    tau_low_ptr(&sig_low, xi_low, &v, q_low, &tau_high[k]);
                }
                else if(y_high[k] > high_thresh){
                    v = y_high[k] - high_thresh;
                    tau_high_ptr(&sig_high, xi_high, &v, q_high, &tau_high[k]);
                }
                else{
                    rootfind_GPU (M_knots_length, I_knots_length, M, spline_df, &tau_high[k], w, &y_high[k],  M_knots, I_knots, &bin_high[k], q_low, q_high, IKM, MKM, M_1, reset_value);
                }
                if(tau_high[k] == 0){tau_high[k] = 0.0000000001;} /*crummy fix*/
                log_like[k] =  log(tau_high[k]);
                *ll_sum += log_like[k];
            }
        }
        if(*n3 > 0){ /*observations censored above*/
            for(k = (*n1 + *n2); k < (*n1 + *n2 + *n3); k++){
                low_thresh = 0;
                high_thresh = 0;
                sig_low = 0;
                sig_high = 0;
                for(m = 0; m < *M; m++){
                    w[m] = theta[m]; /*create weight vector*/
                    for(p = 1; p < *P; p++){
                        w[m] += X[k + *N * p] * theta[m + *M * p];
                    }
                    low_thresh += w[m] * I_low[m];
                    high_thresh += w[m] * I_high[m];
                    sig_low += w[m] * M_low[m];
                    sig_high += w[m] * M_high[m];
                }
                if(y_low[k] < low_thresh){
                    v = low_thresh - y_low[k];
                    tau_low_ptr(&sig_low, xi_low, &v, q_low, &tau_low[k]);
                    log_like[k] = log(1 - tau_low[k]);
                }
                else if(y_low[k] > high_thresh){
                        v = y_low[k] - high_thresh;
                        tau_high_ptr(&sig_high, xi_high, &v, q_high, &tau_low[k]);
                    }
                else{
                    rootfind_GPU (M_knots_length, I_knots_length, M, spline_df, &tau_low[k], w, &y_low[k],  M_knots, I_knots, &bin_low[k], q_low, q_high,
                        IKM, MKM, M_1, reset_value);
                }
                if(tau_low[k] == 1){tau_low[k] = 0.999999999;} /*crummy fix*/
                log_like[k] = log(1 - tau_low[k]);
                *ll_sum += log_like[k];
            }
        }
        if(*n4 > 0 ){/*observations that are interval censored*/
            for(k = (*n1 + *n2 + *n3); k < (*n1 + *n2 + *n3 + *n4); k++){ /*interval censored observations*/
                low_thresh = 0;
                high_thresh = 0;
                sig_low = 0;
                sig_high = 0;
                for(m = 0; m < *M; m++){
                    w[m] = 0; /*create weight vector*/
                    for(p = 0; p < *P; p++){
                        w[m] += X[k + *N * p] * theta[m + *M * p];
                    }
                    low_thresh += w[m] * I_low[m];
                    high_thresh += w[m] * I_high[m];
                    sig_low += w[m] * M_low[m];
                    sig_high += w[m] * M_high[m];
                }
                if(y_high[k] < low_thresh){
                    v = low_thresh - y_low[k];
                    tau_low_ptr(&sig_low, xi_low, &v, q_low, &tau_low[k]);
                    v = low_thresh - y_high[k];
                    tau_low_ptr(&sig_low, xi_low, &v, q_low, &tau_high[k]);
                }
                else if (y_low[k] < low_thresh){
                    v = low_thresh - y_low[k];
                    tau_low_ptr(&sig_low, xi_low, &v, q_low, &tau_low[k]);
                    rootfind_GPU (M_knots_length, I_knots_length, M, spline_df, &tau_high[k], w, &y_high[k],  M_knots, I_knots, &bin_high[k], q_low, q_high, IKM, MKM, M_1, reset_value);
                }
                else if(y_low[k] > high_thresh){
                        v = y_low[k] - high_thresh;
                        tau_high_ptr(&sig_high, xi_high, &v, q_high, &tau_low[k]);
                        v = y_high[k] - high_thresh;
                        tau_high_ptr(&sig_high, xi_high, &v, q_high, &tau_high[k]);
                    }
                else if(y_high[k] > high_thresh){
                        rootfind_GPU (M_knots_length, I_knots_length, M, spline_df, &tau_low[k], w, &y_low[k],  M_knots, I_knots, &bin_low[k], q_low, q_high, IKM, MKM, M_1, reset_value);
                            v = y_high[k] - high_thresh;
                        tau_high_ptr(&sig_high, xi_high, &v, q_high, &tau_high[k]);
                }
                else{
                    rootfind_GPU (M_knots_length, I_knots_length, M, spline_df, &tau_low[k], w, &y_low[k],  M_knots, I_knots, &bin_low[k], q_low, q_high, IKM, MKM, M_1, reset_value);
                    rootfind_GPU (M_knots_length, I_knots_length, M, spline_df, &tau_high[k], w, &y_high[k],  M_knots, I_knots, &bin_high[k], q_low, q_high, IKM, MKM, M_1, reset_value);
                }
                if(tau_low[k] == 1){tau_low[k] = 0.999999999;} /*crummy fix*/
                if(tau_high[k] == 0){tau_high[k] = 0.0000000001;} /*crummy fix*/
                log_like[k] =  log(tau_high[k] - tau_low[k]);
                *ll_sum += log_like[k];
            }
        }
        if(*n5 > 0){/*binary data*/
            for(k = (*n1 + *n2 + *n3 + *n4); k < *N; k++){
                low_thresh = 0;
                high_thresh = 0;
                sig_low = 0;
                sig_high = 0;
                for(m = 0; m < *M; m++){
                    w[m] = 0;
                    for(p = 0; p < *P; p++){
                        w[m] += X[k + *N * p] * theta[m + *M * p];
                    }
                    low_thresh += w[m] * I_low[m];
                    high_thresh += w[m] * I_high[m];
                    sig_low += w[m] * M_low[m];
                    sig_high += w[m] * M_high[m];
                }
                if(y_low[k] < low_thresh){
                    v = low_thresh - y_low[k];
                    tau_low_ptr(&sig_low, xi_low, &v, q_low, &tau[k]); /*find tau s.t. Q(tau) = cutpoint */
                }
                else if(y_low[k] > high_thresh){
                        v = y_low[k] - high_thresh;
                        tau_high_ptr(&sig_high, xi_high, &v, q_high, &tau[k]);
                }
                else{
                    rootfind_GPU (M_knots_length, I_knots_length, M, spline_df, &tau[k], w, &y_low[k],  M_knots, I_knots, &bin[k], q_low, q_high, IKM, MKM, M_1, reset_value);
                }
                if(tau[k] == 1){tau[k] = 0.999999999;} /*crummy fix*/
                if(tau[k] == 0){tau[k] = 0.0000000001;} /*crummy fix*/
                if(y[k] == 0){log_like[k] = log(tau[k]);}
                else{log_like[k] = log(1 - tau[k]);}
                *ll_sum += log_like[k];
            }
        }
}

void ll_G (int *M, int *G, int *P, int *g_min, int *g_max,
        int *N, int *n1, int *n2, int *n3, int *n4, int *n5,
        double *X, double *y, double *y_low, double *y_high,
        int *M_knots_length, int *I_knots_length, int *spline_df, double *M_knots, double *I_knots,
        double *M_low, double *M_high, double *I_low, double *I_high,
        double *q_low, double *q_high, int *xi_zero,
        double *theta,double *xi_low, double *xi_high,
        double *tau, double *tau_low, double *tau_high, double *log_like, double *ll_sum,
        int *bin, int *bin_low, int *bin_high,
        void ld_low_ptr(double *, double *, double *, double *, double*), /*pointer to the log density for low obs*/
        void ld_high_ptr(double *, double *, double *, double *, double*), /*pointer to the log density for high obs*/
        void tau_low_ptr(double *, double *, double *, double *, double*), /*pointer to the cdf*/
        void tau_high_ptr(double *, double *, double *, double *, double*),
        double *IKM, double *MKM, int *M_1, double *reset_value
        ){
        int g, k, m, p;
        double low_thresh, high_thresh, sig_low, sig_high, v;
        double w[*M];


        for(g = *g_min; g < *g_max; g++){
            ll_sum[g] = 0;
            if(n1[g] > 0){/*uncensored data*/
                for (k = 0; k < n1[g]; k++){
                    low_thresh = 0;
                    high_thresh = 0;
                    sig_low = 0;
                    sig_high = 0;
                    for(m = 0; m < *M; m++){
                        w[m] = theta[m];
                        for(p = 1; p < *P; p++){
                            w[m] += X[k + *N * p] * theta[m + *M * p];
                        }
                        low_thresh += w[m] * I_low[m];
                        high_thresh += w[m] * I_high[m];
                        sig_low += w[m] * M_low[m];
                        sig_high += w[m] * M_high[m];
                    }
                    if(y[k] < low_thresh){
                        v = low_thresh - y[k];
                        ld_low_ptr(&sig_low, xi_low, &v, q_low, &log_like[k]);
                        tau_low_ptr(&sig_low, xi_low, &v, q_low, &tau[k]);
                    }
                    else if(y[k] > high_thresh){
                        v = y[k] - high_thresh;
                        ld_high_ptr(&sig_high, xi_high, &v, q_high, &log_like[k]);
                        tau_high_ptr(&sig_high, xi_high, &v, q_high, &tau[k]);
                    }
                    else{
                        rootfind_GPU (M_knots_length, I_knots_length, M, spline_df, &tau[k], w, &y[k],  M_knots, I_knots, &bin[k], q_low, q_high, IKM, MKM, M_1, reset_value); /*pass to void*/
                        log_like[k] = -1 * log(q_3 (*M, tau[k], w, *spline_df, I_knots, bin[k], MKM, *M_1));
                    }
                    ll_sum[g] += log_like[k];
                }
            }
            if(n2[g] > 0){ /*obsevations censored below*/
                for(k = n1[g]; k < (n1[g] + n2[g]); k++){
                    low_thresh = 0;
                    high_thresh = 0;
                    sig_low = 0;
                    sig_high = 0;
                    for(m = 0; m < *M; m++){
                        w[m] = theta[m]; /*create weight vector*/
                        for(p = 1; p < *P; p++){
                            w[m] += X[k + *N * p] * theta[m + *M * p];
                        }
                        low_thresh += w[m] * I_low[m];
                        high_thresh += w[m] * I_high[m];
                        sig_low += w[m] * M_low[m];
                        sig_high += w[m] * M_high[m];
                    }
                    if(y_high[k] < low_thresh){
                        v = low_thresh - y_high[k];
                        tau_low_ptr(&sig_low, xi_low, &v, q_low, &tau_high[k]);
                    }
                    else if(y_high[k] > high_thresh){
                        v = y_high[k] - high_thresh;
                        tau_high_ptr(&sig_high, xi_high, &v, q_high, &tau_high[k]);
                    }
                    else{
                        rootfind_GPU (M_knots_length, I_knots_length, M, spline_df, &tau_high[k], w, &y_high[k],  M_knots, I_knots, &bin_high[k], q_low, q_high, IKM, MKM, M_1, reset_value);
                    }
                    if(tau_high[k] == 0){tau_high[k] = 0.0000000001;} /*crummy fix*/
                    log_like[k] =  log(tau_high[k]);
                    ll_sum[g] += log_like[k];
                }
            }
            if(n3[g] > 0){ /*observations censored above*/
                for(k = (n1[g] + n2[g]); k < (n1[g] + n2[g] + n3[g]); k++){
                    low_thresh = 0;
                    high_thresh = 0;
                    sig_low = 0;
                    sig_high = 0;
                    for(m = 0; m < *M; m++){
                        w[m] = theta[m]; /*create weight vector*/
                        for(p = 1; p < *P; p++){
                            w[m] += X[k + *N * p] * theta[m + *M * p];
                        }
                        low_thresh += w[m] * I_low[m];
                        high_thresh += w[m] * I_high[m];
                        sig_low += w[m] * M_low[m];
                        sig_high += w[m] * M_high[m];
                    }
                    if(y_low[k] < low_thresh){
                        v = low_thresh - y_low[k];
                        tau_low_ptr(&sig_low, xi_low, &v, q_low, &tau_low[k]);
                        log_like[k] = log(1 - tau_low[k]);
                    }
                    else if(y_low[k] > high_thresh){
                        v = y_low[k] - high_thresh;
                        tau_high_ptr(&sig_high, xi_high, &v, q_high, &tau_low[k]);
                    }
                    else{
                        rootfind_GPU (M_knots_length, I_knots_length, M, spline_df, &tau_low[k], w, &y_low[k],  M_knots, I_knots, &bin_low[k], q_low, q_high,
                            IKM, MKM, M_1, reset_value);
                    }
                    if(tau_low[k] == 1){tau_low[k] = 0.999999999;} /*crummy fix*/
                    log_like[k] = log(1 - tau_low[k]);
                    ll_sum[g] += log_like[k];
                }
            }
            if(n4[g] > 0 ){/*observations that are interval censored*/
                for(k = (n1[g] + n2[g] + n3[g]); k < (n1[g] + n2[g] + n3[g] + n4[g]); k++){ /*interval censored observations*/
                    low_thresh = 0;
                    high_thresh = 0;
                    sig_low = 0;
                    sig_high = 0;
                    for(m = 0; m < *M; m++){
                        w[m] = 0; /*create weight vector*/
                        for(p = 0; p < *P; p++){
                            w[m] += X[k + *N * p] * theta[m + *M * p];
                        }
                        low_thresh += w[m] * I_low[m];
                        high_thresh += w[m] * I_high[m];
                        sig_low += w[m] * M_low[m];
                        sig_high += w[m] * M_high[m];
                    }
                    if(y_high[k] < low_thresh){
                        v = low_thresh - y_low[k];
                        tau_low_ptr(&sig_low, xi_low, &v, q_low, &tau_low[k]);
                        v = low_thresh - y_high[k];
                        tau_low_ptr(&sig_low, xi_low, &v, q_low, &tau_high[k]);
                    }
                    else if (y_low[k] < low_thresh){
                        v = low_thresh - y_low[k];
                        tau_low_ptr(&sig_low, xi_low, &v, q_low, &tau_low[k]);
                        rootfind_GPU (M_knots_length, I_knots_length, M, spline_df, &tau_high[k], w, &y_high[k],  M_knots, I_knots, &bin_high[k], q_low, q_high, IKM, MKM, M_1, reset_value);
                    }
                    else if(y_low[k] > high_thresh){
                        v = y_low[k] - high_thresh;
                        tau_high_ptr(&sig_high, xi_high, &v, q_high, &tau_low[k]);
                        v = y_high[k] - high_thresh;
                        tau_high_ptr(&sig_high, xi_high, &v, q_high, &tau_high[k]);
                    }
                    else if(y_high[k] > high_thresh){
                        rootfind_GPU (M_knots_length, I_knots_length, M, spline_df, &tau_low[k], w, &y_low[k],  M_knots, I_knots, &bin_low[k], q_low, q_high, IKM, MKM, M_1, reset_value);
                            v = y_high[k] - high_thresh;
                        tau_high_ptr(&sig_high, xi_high, &v, q_high, &tau_high[k]);
                    }
                    else{
                        rootfind_GPU (M_knots_length, I_knots_length, M, spline_df, &tau_low[k], w, &y_low[k],  M_knots, I_knots, &bin_low[k], q_low, q_high, IKM, MKM, M_1, reset_value);
                        rootfind_GPU (M_knots_length, I_knots_length, M, spline_df, &tau_high[k], w, &y_high[k],  M_knots, I_knots, &bin_high[k], q_low, q_high, IKM, MKM, M_1, reset_value);
                    }
                    if(tau_low[k] == 1){tau_low[k] = 0.999999999;} /*crummy fix*/
                    if(tau_high[k] == 0){tau_high[k] = 0.0000000001;} /*crummy fix*/
                    log_like[k] =  log(tau_high[k] - tau_low[k]);
                    ll_sum[g] += log_like[k];
                }
            }
            if(n5[g] > 0){/*binary data*/
                for(k = (n1[g] + n2[g] + n3[g] + n4[g]); k < (n1[g] + n2[g] + n3[g] + n4[g] + n5[g]); k++){
                    low_thresh = 0;
                    high_thresh = 0;
                    sig_low = 0;
                    sig_high = 0;
                    for(m = 0; m < *M; m++){
                        w[m] = 0;
                        for(p = 0; p < *P; p++){
                            w[m] += X[k + *N * p] * theta[m + *M * p];
                        }
                        low_thresh += w[m] * I_low[m];
                        high_thresh += w[m] * I_high[m];
                        sig_low += w[m] * M_low[m];
                        sig_high += w[m] * M_high[m];
                    }
                    if(y_low[k] < low_thresh){
                        v = low_thresh - y_low[k];
                        tau_low_ptr(&sig_low, xi_low, &v, q_low, &tau[k]); /*find tau s.t. Q(tau) = cutpoint */
                    }
                    else if(y_low[k] > high_thresh){
                        v = y_low[k] - high_thresh;
                        tau_high_ptr(&sig_high, xi_high, &v, q_high, &tau[k]);
                    }
                    else{
                        rootfind_GPU (M_knots_length, I_knots_length, M, spline_df, &tau[k], w, &y_low[k],  M_knots, I_knots, &bin[k], q_low, q_high, IKM, MKM, M_1, reset_value);
                    }
                    if(tau[k] == 1){tau[k] = 0.999999999;} /*crummy fix*/
                    if(tau[k] == 0){tau[k] = 0.0000000001;} /*crummy fix*/
                    if(y[k] == 0){log_like[k] = log(tau[k]);}
                    else{log_like[k] = log(1 - tau[k]);}
                    ll_sum[g] += log_like[k];
                }
            }
        }
}

 void no_update_tail(
        int *M, int *P,
        int *N, int *n1, int *n2, int *n3, int *n4, int *n5,
        double *X, double *y, double *y_low, double *y_high,
        int *M_knots_length, int *I_knots_length, int *spline_df, double *M_knots, double *I_knots,
        double *M_low, double *M_high, double *I_low, double *I_high,
        double *q_low, double *q_high, int *xi_zero,
        double *theta,double *xi_low, double *xi_high,
        double *tau, double *tau_low, double *tau_high, double *log_like, double *ll_sum,
        int *bin, int *bin_low, int *bin_high,
        void ld_low_ptr(double *, double *, double *, double *, double*), /*pointer to the log density for low obs*/
        void ld_high_ptr(double *, double *, double *, double *, double*), /*pointer to the log density for high obs*/
        void tau_low_ptr(double *, double *, double *, double *, double*), /*pointer to the cdf*/
        void tau_high_ptr(double *, double *, double *, double *, double*),
        double *IKM, double *MKM, int *M_1, double *reset_value,
        double *tuning_tail,
        double *tail_mean,
        double *tail_prec,
        double *E_vec,
        double *Z_vec,
        int *ACC_TAIL)
        {}

 /*function that updates the tails*/
 void update_tail(
        int *M, int *P,
        int *N, int *n1, int *n2, int *n3, int *n4, int *n5,
        double *X, double *y, double *y_low, double *y_high,
        int *M_knots_length, int *I_knots_length, int *spline_df, double *M_knots, double *I_knots,
        double *M_low, double *M_high, double *I_low, double *I_high,
        double *q_low, double *q_high, int *xi_zero,
        double *theta,double *xi_low, double* xi_high,
        double *tau, double *tau_low, double *tau_high, double *log_like, double *ll_sum,
        int *bin, int *bin_low, int *bin_high,
        void ld_low_ptr(double *, double *, double *, double *, double*), /*pointer to the log density for low obs*/
        void ld_high_ptr(double *, double *, double *, double *, double*), /*pointer to the log density for high obs*/
        void tau_low_ptr(double *, double *, double *, double *, double*), /*pointer to the cdf*/
        void tau_high_ptr(double *, double *, double *, double *, double*),
        double *IKM, double *MKM, int *M_1, double *reset_value,
        double *tuning_tail,
        double *tail_mean,
        double *tail_prec,
        double *E_vec,
        double *Z_vec,
        int *ACC_TAIL){
        double can_ll = 0;
        double can_ll_sum; /*dummy vector of candidate log likelihoods*/
        /*lower tail*/
        double can_xi_low =  exp(tuning_tail[0] * Z_vec[0] +  log(*xi_low)); /*do not need to call RNG function*/

        ll(M, P, N, n1, n2, n3, n4, n5,
        X, y, y_low, y_high,
        M_knots_length, I_knots_length, spline_df, M_knots, I_knots,
        M_low, M_high, I_low, I_high,
        q_low, q_high, xi_zero,
        theta, &can_xi_low, xi_high,
        tau, tau_low, tau_high, log_like, &can_ll_sum,
        bin, bin_low, bin_high,
        ld_low_ptr,
        ld_high_ptr,
        tau_low_ptr,
        tau_high_ptr,
        IKM, MKM, M_1, reset_value);

        if (E_vec[0] > (*tail_prec * (pow(log(can_xi_low) - *tail_mean,2) - pow(log(*xi_low) - *tail_mean,2)) + *ll_sum - can_ll_sum)){
            *xi_low = can_xi_low;
            *ll_sum = can_ll;
            ACC_TAIL[0]++;
        }
        /*upper tail*/
        double can_xi_high =  exp(tuning_tail[1] * Z_vec[1] +  log(*xi_high));

        ll(M, P, N, n1, n2, n3, n4, n5,
        X, y, y_low, y_high,
        M_knots_length, I_knots_length, spline_df, M_knots, I_knots,
        M_low, M_high, I_low, I_high,
        q_low, q_high, xi_zero,
        theta, xi_low, &can_xi_high,
        tau, tau_low, tau_high, log_like, &can_ll_sum,
        bin, bin_low, bin_high,
        ld_low_ptr,
        ld_high_ptr,
        tau_low_ptr,
        tau_high_ptr,
        IKM, MKM, M_1, reset_value);

        if (E_vec[1] > (*tail_prec * (pow(log(can_xi_high) - *tail_mean,2) - pow(log(*xi_high) - *tail_mean,2)) + *ll_sum - can_ll_sum)){
            *xi_high = can_xi_high;
            *ll_sum = can_ll;
            ACC_TAIL[1]++;
        }
}
 void no_update_tail_CP(
        int *M, int *G, int *P, int *g_min, int *g_max,
        int *N, int *n1, int *n2, int *n3, int *n4, int *n5,
        double *X, double *y, double *y_low, double *y_high,
        int *M_knots_length, int *I_knots_length, int *spline_df, double *M_knots, double *I_knots,
        double *M_low, double *M_high, double *I_low, double *I_high,
        double *q_low, double *q_high, int *xi_zero,
        double *theta,double *xi_low, double* xi_high,
        double *tau, double *tau_low, double *tau_high, double *log_like, double *ll_sum,
        int *bin, int *bin_low, int *bin_high,
        void ld_low_ptr(double *, double *, double *, double *, double*), /*pointer to the log density for low obs*/
        void ld_high_ptr(double *, double *, double *, double *, double*), /*pointer to the log density for high obs*/
        void tau_low_ptr(double *, double *, double *, double *, double*), /*pointer to the cdf*/
        void tau_high_ptr(double *, double *, double *, double *, double*),
        double *IKM, double *MKM, int *M_1, double *reset_value,
        double *tuning_tail,
        double *tail_mean,
        double *tail_prec,
        double *E_vec,
        double *Z_vec,
        int *ACC_TAIL){}


 /*function that updates the tails*/
 void update_tail_CP(
        int *M, int *G, int *P, int *g_min, int *g_max,
        int *N, int *n1, int *n2, int *n3, int *n4, int *n5,
        double *X, double *y, double *y_low, double *y_high,
        int *M_knots_length, int *I_knots_length, int *spline_df, double *M_knots, double *I_knots,
        double *M_low, double *M_high, double *I_low, double *I_high,
        double *q_low, double *q_high, int *xi_zero,
        double *theta,double *xi_low, double* xi_high,
        double *tau, double *tau_low, double *tau_high, double *log_like, double *ll_sum,
        int *bin, int *bin_low, int *bin_high,
        void ld_low_ptr(double *, double *, double *, double *, double*), /*pointer to the log density for low obs*/
        void ld_high_ptr(double *, double *, double *, double *, double*), /*pointer to the log density for high obs*/
        void tau_low_ptr(double *, double *, double *, double *, double*), /*pointer to the cdf*/
        void tau_high_ptr(double *, double *, double *, double *, double*),
        double *IKM, double *MKM, int *M_1, double *reset_value,
        double *tuning_tail,
        double *tail_mean,
        double *tail_prec,
        double *E_vec,
        double *Z_vec,
        int *ACC_TAIL){
        int g;
        double curr_ll = 0;
        double can_ll = 0;
        double *can_ll_sum = (double *)R_alloc(*G, sizeof(double*)); /*dummy vector of candidate log likelihoods*/
        /*lower tail*/
        double can_xi_low =  exp(tuning_tail[0] * Z_vec[0] +  log(*xi_low)); /*do not need to call RNG function*/

        ll_G(M, G, P, g_min, g_max,
            N, n1, n2, n3, n4, n5,
            X, y, y_low, y_high,
            M_knots_length, I_knots_length, spline_df, M_knots, I_knots,
            M_low, M_high, I_low, I_high,
            q_low, q_high, xi_zero,
            theta, &can_xi_low, xi_high,
            tau, tau_low, tau_high, log_like, can_ll_sum,
            bin, bin_low, bin_high,
            ld_low_ptr,
            ld_high_ptr,
            tau_low_ptr,
            tau_high_ptr,
            IKM, MKM, M_1, reset_value);

        for(g = 0; g < *G; g++){
            curr_ll += ll_sum[g];
            can_ll += can_ll_sum[g];
        }

        if (E_vec[0] > (*tail_prec * (pow(log(can_xi_low) - *tail_mean,2) - pow(log(*xi_low) - *tail_mean,2)) + curr_ll - can_ll)){
            *xi_low = can_xi_low;
            for(g = 0; g < *G; g++){ll_sum[g] = can_ll_sum[g];}
            curr_ll = can_ll;
            ACC_TAIL[0]++;
        }
        /*upper tail*/
        double can_xi_high =  exp(tuning_tail[1] * Z_vec[1] +  log(*xi_high));

        ll_G(M, G, P, g_min, g_max,
            N, n1, n2, n3, n4, n5,
            X, y, y_low, y_high,
            M_knots_length, I_knots_length, spline_df, M_knots, I_knots,
            M_low, M_high, I_low, I_high,
            q_low, q_high, xi_zero,
            theta, xi_low, &can_xi_high,
            tau, tau_low, tau_high, log_like, can_ll_sum,
            bin, bin_low, bin_high,
            ld_low_ptr,
            ld_high_ptr,
            tau_low_ptr,
            tau_high_ptr,
            IKM, MKM, M_1, reset_value);

        if (E_vec[1] > (*tail_prec * (pow(log(can_xi_high) - *tail_mean,2) - pow(log(*xi_high) - *tail_mean,2)) + curr_ll - can_ll)){
            *xi_high = can_xi_high;
            for(g = 0; g < *G; g++){ll_sum[g] = can_ll_sum[g];}
            ACC_TAIL[1]++;
        }
}

/*function that makes the lth row of theta proper*/
void threshold_CP(int *M, int *G, int *P, int *m, int *g, double *thetastar, double *theta, int *slope_update){
    int p;
    slope_update[*g] = 0;
    for(p = 0; p < *P; p++){theta[*m + *M * *g +  *M * *G * p] = thetastar[*m + *M * *g +  *M * *G * p];}
    if(theta[*m + *M * *g] < 0){theta[*m + *M * *g] = 0.001;}
    double neg_part = theta[*m + *M * *g];
    for(p = 1; p < *P; p++){
        if(thetastar[*m + *M * *g +  *M * *G * p] < 0){neg_part +=  thetastar[*m + *M * *g +  *M * *G * p];}
        else{neg_part -= thetastar[*m + *M * *g +  *M * *G * p];}
    }
    if(neg_part < 0){
        slope_update[*g]  = 1;
        for(p = 1; p < *P; p++){
            theta[*m + *M * *g +  *M * *G * p] = 0;
        }
    }
}



void gamma_sample(int *N, double *shape, double *scale, double *sample){
    int n;
    for (n = 0; n < *N; n++){
    sample[n] =   rgamma(*shape,*scale);
    }
}

void make_betahat(int *G, double *GG, int *l, double *theta, double *betahat){
            for(int q = 0; q < *l; q++){
                betahat[q] = 0;
                for(int g = 0; g < *G; g++){
                    betahat[q] += GG[g + *G * *l] * theta[g];
                }
            }
            betahat[0] = betahat[0] / *G;
}

/*for AR-1 cov function that returns betahat and the Cholesky decomposition of a matrix V */
void make_betahat_AR(int *G, double *GG, int *D, int *M, int *m, int *p, double *theta, double *rho, double *V, double *chol_V,
                     double *chol_V_inv, double *V_inv, double *betahat){
    int g, i, j, k, info;

    double rho2 = *rho * *rho;
    double a = 1/(1-rho2);
    double b =     1+2*rho2/(1-rho2);
    double c = -*rho/(1-rho2);
    double rhoquad2 = 0;
    double temp_sum = 0;
    double x1[*G];
    double x2[*G];
    /* part 1: build V = t(GG)%*%AR%*%GG */
    for(i = 0; i < *D; i++){ /*row index*/
        for(g = 0; g < *G; g++){x1[g] = GG[g + *G * i];}
            for(j = 0; j <= i; j++){ /*col index*/
                temp_sum = 0;
                rhoquad2 = 0;
                for(g = 0; g < *G; g++){x2[g] = GG[g + *G * j];}
                for(g = 0; g < *G-1; g++){rhoquad2 += x1[g] * x2[g+1] + x2[g] * x1[g+1];}
                rhoquad2 = rhoquad2 * c;
                for(g = 1; g < (*G - 1); g++){temp_sum += x1[g] * x2[g];}
                rhoquad2 += b * temp_sum;
                rhoquad2 = rhoquad2 + a * (x1[0] * x2[0] + x1[*G-1] * x2[*G-1]);
                V[i + *D * j] = rhoquad2;
                V[j + *D * i] = rhoquad2;
                chol_V[i + *D * j] = rhoquad2;
                chol_V[j + *D * i] = rhoquad2;
         }
    }

	// part 2: compute chol(V)

    char uplo = 'L';
	F77_CALL(dpotrf)(&uplo, D, chol_V, D, &info);
	if (info) {
		Rprintf("rmvnorm: Error with chol(sigma1): info = %d\n", info);
    }
    for(i = 0; i < *D; i++){
        for(j = 0; j < *D; j++){
            chol_V_inv[j + *D * i] = 0;
            chol_V_inv[i + *D * j] = chol_V[i + *D * j];
        }
    }
/*invert chol_V and create the cholesky decomposition of V_inv*/
    char diag = 'N';
	F77_CALL(dtrtri)(&uplo, &diag, D, chol_V_inv, D, &info);//make sure you keep chol_V_inv

/*create V_inv*/
	for(i = 0; i < *D; i++){ /*row*/
        for(j = 0; j <= i; j++){ /*col*/
            V_inv[i + *D *j] = 0;
            for(k = i; k < *D;  k++){V_inv[i + *D * j] += chol_V_inv[k + *D * i] * chol_V_inv[k + *D * j];}
            V_inv[j + *D * i] = V_inv[i + *D * j];
        }
	}
    for(i = 0; i < *D; i++){ /*element of betahat index*/
        for(g = 0; g < *G; g++){ /*create the G-dimensional x1 vector for the ith element*/
            x1[g] = 0;
            for(k = 0; k < *D; k++){x1[g] += V_inv[i + *D * k] * GG[g + *G * k];}
        }
                temp_sum = 0;
                rhoquad2 = 0;
                for(g = 0; g < *G-1; g++){rhoquad2 += x1[g] * theta[*m + *M * (g+1) + *M * *G * *p] + theta[*m + *M * g + *M * *G * *p] * x1[g+1];}
                rhoquad2 = rhoquad2 * c;
                for(g = 1; g < (*G - 1); g++){temp_sum += x1[g] * theta[*m + *M * g + *M * *G * *p];}
                rhoquad2 += b * temp_sum;
                rhoquad2 += a * (x1[0] * theta[*m + *M * 0 + *M * *G * *p] + x1[*G-1] * theta[*m + *M * (*G-1)+ *M * *G * *p]);
                betahat[i] = rhoquad2;
    }
}

/*computes (theta-mu)%*%AR(PREC)(theta-mu) where mu is a vector*/
void rho_quad_vector(double *rho, int *M, int *G, int *P, int *m, int *p, double *theta, double *mu, double *rhoquad){
    int g;
    double rho2 = *rho * *rho;
    double a = 1/(1-rho2);
    double b =     1+2*rho2/(1-rho2);
    double c = -*rho/(1-rho2);
    double x[*G];
    *rhoquad = 0;
    for(g = 0; g < *G; g++){
        x[g] = theta[*m + *M * g + *M * *G * *p] - mu[*m + *M * g + *M * *G * *p];
        }
    for(g = 0; g < *G-1; g++){
        *rhoquad += x[g] * x[g+1];
    }
    *rhoquad = *rhoquad * (2 * c);
    for(g = 1; g < (*G - 1); g++){
        *rhoquad += b * x[g] * x[g];
    }
    *rhoquad += (a) * (x[0] * x[0] + x[*G-1] * x[*G-1]);
    }

    void beta_ll_AR(int *G, int *D, double *beta, double *sigma2, double *GG, double *theta, double *beta_ll_sum){
    double x[*D];
    for(int q = 0; q < *D; q++){
        for(int g = 0; g < *G; g++){
            x[q] += GG[g + *G * q] * theta[g];
            }
            x[q]= beta[q] - x[q];
    }
    *beta_ll_sum = x[0]*x[0]* *G;
  if(*D > 1){
    for(int q = 1; q < *D; q++){
        *beta_ll_sum += x[q]*x[q];
    }
  }
    *beta_ll_sum *= -0.5 * *sigma2;
}

void beta_ll(int *G, int *l, double *beta, double *sigma2, double *GG, double *theta, double *beta_ll_sum){
    double x[*l];
    for(int q = 0; q < *l; q++){
        for(int g = 0; g < *G; g++){
            x[q] += GG[g + *G * q] * theta[g];
            }
            x[q]= beta[q] - x[q];
    }
    *beta_ll_sum = x[0]*x[0]* *G;
  if(*l > 1){
    for(int q = 1; q < *l; q++){
        *beta_ll_sum += x[q]*x[q];
    }
  }
    *beta_ll_sum *= -0.5 * *sigma2;
}
/*function that makes the lth row of theta proper*/
void threshold(int *M, int *P, int *m, double *thetastar, double *theta, int *slope_update){
    int p;
    *slope_update = 0;
    for(p = 0; p < *P; p++){theta[*m + *M * p] = thetastar[*m + *M * p];}
    if(theta[*m] < 0){theta[*m] = 0.001;}
    double neg_part = theta[*m];
    for(p = 1; p < *P; p++){
        if(thetastar[*m + *M * p] < 0){neg_part +=  thetastar[*m + *M * p];}
        else{neg_part -= thetastar[*m + *M * p];}
    }
    if(neg_part < 0){
        *slope_update = 1;
        for(p = 1; p < *P; p++){
            theta[*m + *M * p] = 0;
        }
    }
}
