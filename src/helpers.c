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
#include "sys/time.h" /*for timing stuff*/
#include "helpers.h"
#include "helpers_np.h"

void sum_alpha(double *rho, int *L, int *P, int *p, double *alpha, double *sumalpha, double *sumprec){
    int l;
    double rho2 = *rho * *rho;
    double a = 1 / (1-rho2);
    double b =     1 + 2 * rho2 / (1 - rho2);
    double c = -*rho / (1 - rho2);
    *sumprec = (2/(1-rho2) + (*L - 2) * (1 + 2 * rho2 / (1 - rho2)) + 2*(*L - 1)*(-*rho/(1 - rho2)));
    *sumalpha = 0;
    for(l = 1; l < *L - 1; l++){
        *sumalpha += alpha[l + *L * *p];
    }
    *sumalpha *= (b + 2 * c);
    *sumalpha += (a + c) * (alpha[*L * *p] + alpha[(*L - 1) + *L * *p]);
    }

void rho_quad_cens(double *rho, int *L, int *P, int *p, double *alpha, double *mu, double *rhoquad){
    int l;
    double rho2 = *rho * *rho;
    double a = 1/(1 - rho2);
    double b =     1 + 2 * rho2 / (1 - rho2);
    double c = -*rho / (1 - rho2);
    double x[*L];
    *rhoquad = 0;
    for(l = 0; l < *L; l++){ /*ignore the first basis function*/
        x[l] = alpha[l + *L * *p] - mu[*p];
        }
    for(l = 0; l < (*L - 1); l++){
        *rhoquad += x[l] * x[l+1];
    }
    *rhoquad *= (2 * c);
    for(l = 1; l < (*L - 1); l++){
        *rhoquad += b * x[l] * x[l];
    }
    *rhoquad += a * (x[0] * x[0] + x[*L - 1] * x[*L - 1]);
    }


/*************** Quantile Functions    ***********/

void q_norm(double *tau, double *mn, double *scale, double *shape, double *q_tau){
    *q_tau = *mn + *scale * qnorm(*tau, 0, 1, 1, 0);
}

void q_t(double *tau, double *mn, double *scale, double *shape, double *q_tau){
    *q_tau = *mn + *scale * qt(*tau, *shape, 1, 0);
}

void q_logistic(double *tau, double *mn, double *scale, double *shape, double *q_tau){
    *q_tau = *mn + *scale * (log(*tau) - log(1-*tau));
}

void q_alap (double *tau, double *mn ,double *scale, double *shape, double *q_tau){
    double kappa = pow(*shape/(1 - *shape),0.5);
    double temp5 = (kappa * kappa)/(1 + kappa * kappa);
    int index1 = 0;
    if(*tau <= temp5){index1 = 1;}
    if(index1 == 1){
        *q_tau = *mn + (*scale * kappa) * log(*tau / temp5) / pow(2,.5);
    }
    else{
        *q_tau = *mn - (*scale/kappa) * (log1p(kappa * kappa) + log1p(-*tau))/(pow(2,.5));
        }
}

void q_weibull(double *tau, double *mn ,double *scale, double *shape, double *q_tau){
    *q_tau = *mn + qweibull(*tau, *shape, *scale, 1, 0);
}

 void q_gamma(double *tau, double *mn ,double *scale, double *shape, double *q_tau){
    *q_tau = *mn + qgamma(*tau, *shape, *scale, 1, 0);
}

/*************** log CDF Functions  ***********/
void log_p_norm(double *y, double *mn, double *scale, double *shape, double *p_y){
    *p_y = pnorm(*y, *mn, *scale, 1, 1);
}

void log_p_t(double *y, double *mn, double *scale, double *shape, double *p_y){
    double z = (*y - *mn)/ *scale;
    *p_y = pt(z, *shape, 1, 1);
}

void log_p_logistic(double *y, double *mn, double *scale, double *shape, double *p_y){
    *p_y = -log(1+exp((*mn - *y)/ *scale  ));

}

void log_p_alap(double *y, double *mn, double *scale, double *shape, double *p_y){
    double kappa = pow(*shape/(1 - *shape), 0.5);
    double exponent = -(pow(2,.5)/ *scale) * (*y - *mn) * kappa;
    if(*y < *mn){exponent *= -1/(kappa * kappa);}
    double temp5 = exp(exponent)/(1 + kappa * kappa);
    if(*y < *mn){*p_y = 2 * log(kappa) + log(temp5);}
    else{*p_y = log(1 - temp5);}
}

void log_p_weibull(double *y, double *mn ,double *scale, double *shape, double *p_y){
    double z = (*y - *mn);
    *p_y = pweibull(z, *shape, *scale, 1, 1);
}

void log_p_gamma(double *y, double *mn ,double *scale, double *shape, double *p_y){
    double z = (*y - *mn);
    *p_y = pgamma(z, *shape, *scale, 1, 1);
}

/*************** Log Density Functions    ***********/
void log_d_norm(double *y, double *mn, double *scale, double *shape, double *d_Y){
    *d_Y = dnorm(*y, *mn, *scale, 1);
}

void log_d_t(double *y, double *mn, double *scale, double *shape, double *d_Y){
    *d_Y =  dt((*y - *mn) / *scale, *shape, 1) -log(*scale);
}

void log_d_alap (double *y, double *mn ,double *scale, double *shape, double *d_Y){
    double kappa = pow(*shape/(1 - *shape),0.5);
    double kappa2 = kappa * kappa;
    *d_Y = 0.5 * log(2) - log(*scale) + log(kappa) - log1p(kappa2);
    double abs_diff = *y - *mn;
    if (*y < *mn){
        kappa = 1/kappa;
        abs_diff *= -1;
        }
    *d_Y -= pow(2,.5) / *scale * abs_diff * kappa;
}

void log_d_logistic (double *y, double *mn ,double *scale, double *shape, double *d_Y){
        double z = (*y - *mn)/ *scale;
        *d_Y = (-z - 2 * log(1 + exp(-z)) - log(*scale));

}

void log_d_weibull(double *y, double *mn ,double *scale, double *shape, double *d_Y){
    *d_Y = dweibull(*y - *mn, *shape, *scale, 1);
}

void log_d_gamma(double *y, double *mn ,double *scale, double *shape, double *d_Y){
    *d_Y = dgamma(*y - *mn, *shape, *scale, 1);
}

/*xt is now matrix of dim N x n
 tau is vector of length N*/
 /***********************************************
this function is based on the source code for the
"findInterval" function in R
************************************************/

void find_int_2(int *n, int *N, double *xt, double *tau, int *bin){
    int i;
    int istep, middle, ihi;
    for(i = 0; i < *N; i++){
        middle = 0;
        ihi = bin[i] + 1;
        if (tau[i] <  xt[i + *N * ihi]) {
        if (tau[i] >=  xt[i + *N * bin[i]]) {} /* `lucky': same interval as last time */
        else  {
 /* **** now *x< * xt[* bin] . decrease * bin to capture *x*/
            for(istep = 1; ; istep *= 2) {
            ihi = bin[i];
            bin[i] = ihi - istep;
            if (bin[i] <= 1)
            break;
            if (tau[i] >=  xt[i + *N * bin[i]]) goto L50;
            }
            bin[i] = 1;
        }
        }
        else {
        /* **** now *tau>= * xt[ihi] . increase ihi to capture *x*/
            for(istep = 1; ; istep *= 2) {
                bin[i] = ihi;
            ihi = bin[i] + istep;
            if (ihi >= *n)
                break;
            if (tau[i]< xt[i + *N * ihi]) goto L50;
            }
            ihi = *n;
        }
        L50:
        /* **** now * xt[* bin] <= *x< * xt[ihi] . narrow the interval. */
        while(middle != bin[i]){
        /* note. it is assumed that middle = * bin in case ihi = * bin+1 . */
            if (tau[i] >= xt[i + *N * middle])
            bin[i] = middle;
            else
            ihi = middle;
            middle = (bin[i] + ihi) / 2;
        }
    }
}

/*indicator = 0 => normal*/
/*indicator = 1 => t*/
/*indicator = 2 => logistic*/
/*indicator = 3 => asymmetric laplace*/
/*indicator = 4 => weibull*/
/*indicator = 5 => gamma*/

/*L is the number of basis functions*/
/*returns  (L+1) x L matrix B

this function updates the lower triangular portion of the basis matrix B

The [1,1] element of B is -10000, the [(L+1),L] element of B is 10,000 (I eschewed using infinity)

this function should not be used for general tau (e.g. plotting)

*/


void make_B_check (int *L, double *kappa,
             int *indicator,
             double *shape, double *B){
             int i, j;
             double scale = 1; /*for identifiability mean is 0 and scale is 1*/
             double mn = 0;
             double previous, current;

            /*first column*/

    void (*q_ptr)(double *, double *, double *, double *, double*); /*pointer to the quantile function*/

    if(*indicator == 0){
        q_ptr       = &q_norm;

    }
    else if(*indicator == 1){
        q_ptr       = &q_t;

    }
    else if(*indicator == 2){
        q_ptr       = &q_logistic;

    }
    else if(*indicator == 3){
        q_ptr       = &q_alap;
    }
    else if(*indicator == 4){
        q_ptr       = &q_weibull;
    }
    else{
        q_ptr        = &q_gamma;
    }
            for(i = 1; i < (*L+1); i++){
                q_ptr(&kappa[1], &mn, &scale, shape, &B[i]);
            }
            /*second column to (L-1)th column*/
            for(j = 1; j < (*L - 1); j++){
                q_ptr(&kappa[j], &mn, &scale, shape, &previous);
                q_ptr(&kappa[j+1], &mn, &scale, shape, &current);
                for(i = (j+1); i < (*L + 1); i++){
                    B[i + (*L + 1) * j] = current - previous;
                }
            }
}


void make_B (int *L, double *kappa,
             void q_ptr(double *, double *, double *, double *, double*),
             double *shape, double *B){
             int i, j;
             double scale = 1; /*for identifiability mean is 0 and scale is 1*/
             double mn = 0;
             double previous, current;

            /*first column*/
            for(i = 1; i < (*L+1); i++){
                q_ptr(&kappa[1], &mn, &scale, shape, &B[i]);
            }
            /*second column to (L-1)th column*/
            for(j = 1; j < (*L - 1); j++){
                q_ptr(&kappa[j], &mn, &scale, shape, &previous);
                q_ptr(&kappa[j+1], &mn, &scale, shape, &current);
                for(i = (j+1); i < (*L + 1); i++){
                    B[i + (*L + 1) * j] = current - previous;
                }
            }
}


/*
n1 is the number of observations that were censored below
n2 - n1 is the number of uncensored observations
n - n2 is the number of observations that were censored above

P is the number of predictors
L is the number of basis functions

X is the design matrix (n) x P
y is the observed data

beta are the location parameters (P x 1 vector)
alpha are the scale parameters (L x P matrix)
kappa are the breakpoints
shape is the shape parameter for the basis functions

q_ptr is the quantile function
log_d_ptr is the log density function

B is the matrix of basis functions

ll_sum is the summed log likelihood
bin is the vector of length N containing which bin each observation is in

BM is the break matrix
*/

/*I want to construct the break matrix for each likelihood evaluation,
but do not need to calculate B for each evaluation*/


void clike (int *N, int *n1, int *n2, int *n3, int *n4, double *kappa, int *P, int *L,
            double *X, double *y, double *y_low, double *y_high,
            double *beta, double *alpha,
            double *shape,
            void q_ptr(double *, double *, double *, double *, double*),
            void log_p_ptr(double *, double *, double *, double *, double*),
            void log_d_ptr(double *, double *, double *, double *, double*),
            double *B,
            int *bin, int *bin_low, int *bin_high, double *ll_sum, double *log_like
            ){

            int i, j, k, l, p;
            int L_1 = *L + 1;
            double mn, s;
            double dummy_mean = 0;
            double dummy_scale = 1;

            double *BM = (double *)R_alloc(((*L + 1) * *P), sizeof(double*)); /*break matrix*/
            double breaks[L_1]; /*breaks for each iteration*/

            double q_kappa[*L];
            for(l = 1; l < *L; l++){q_ptr(&kappa[l], &dummy_mean, &dummy_scale, shape, &q_kappa[l]);}


            /**********construct the break matrix************/
            /*compute the rest of the rows*/
            for(i = 1; i < (*L); i++){
                for(j = 0; j < *P; j++){
                    BM[i + (*L + 1) * j] = 0;
                        for(k = 0; k < i; k++){
                            BM[i + (*L + 1) * j] += B[i + (*L + 1) * k] * alpha[k + *L * j];
                        }
                }
            }
            *ll_sum = 0;

            breaks[0] = -1000000.0;
            breaks[*L] = 1000000.0;
            if(*n1 > 0){
                for(i = 0; i < *n1; i++){ /*uncensored observations*/
                    for(l = 1; l < (*L); l++){ /*element of breaks*/
                        breaks[l] = 0;
                        for(p = 0; p < *P; p++){ /*sum across all predictors*/
                            breaks[l] +=   (BM[l + (*L + 1) * p] + beta[p])*  X[i + *N * p];
                        }
                    }
                    find_int(&L_1, breaks, &y[i], &bin[i]);
                    s = 0;
                    if(bin[i]==0){
                        mn = 0;
                        for(p = 0; p < *P; p++){
                            mn += beta[p] * X[i + *N * p];
                            s += alpha[*L * p] * X[i + *N * p];
                        }
                    }
                    else{
                        for(p = 0; p < *P; p++){
                            s += alpha[bin[i] + *L * p] * X[i + *N * p];
                        }
                        mn = breaks[bin[i]] - s * q_kappa[bin[i]];
                    }
                    log_d_ptr(&y[i], &mn, &s, shape, &log_like[i]);
                    *ll_sum += log_like[i];
                }
            }
            if(*n2 > 0){/*obsevations censored below*/
                for(i = *n1; i < (*n1 + *n2); i++){
                    for(l = 1; l < (*L); l++){ /*element of breaks*/
                        breaks[l] = 0;
                        for(p = 0; p < *P; p++){ /*sum across all predictors*/
                            breaks[l] +=   (BM[l + (*L + 1) * p] + beta[p]) *  X[i + *N * p];
                        }
                    }
                    find_int(&L_1, breaks, &y[i], &bin[i]);
                    s = 0;
                    if(bin[i]==0){
                        mn = 0;
                        for(p = 0; p < *P; p++){
                            mn+= beta[p] * X[i + *N * p];
                            s+= alpha[*L * p] * X[i + *N * p];
                        }
                    }
                    else{
                        for(p = 0; p < *P; p++){
                            s+= alpha[bin[i] + *L * p] * X[i + *N * p];
                        }
                        mn = breaks[bin[i]] - s * q_kappa[bin[i]];
                    }
                    log_p_ptr(&y[i], &mn, &s, shape, &log_like[i]);
                    *ll_sum += log_like[i];
                }
            }
            if(*n3 > 0){ /*observations censored above*/
                for(i = *n1 + *n2; i < (*n1 + *n2 + *n3); i++){
                for(l = 1; l < (*L); l++){ /*element of breaks*/
                breaks[l] = 0;
                    for(p = 0; p < *P; p++){ /*sum across all predictors*/
                        breaks[l] +=   (BM[l + (*L + 1) * p] + beta[p])*  X[i + *N * p];
                        }
                }
                find_int(&L_1, breaks, &y[i], &bin[i]);
                s = 0;
                if(bin[i]==0){
                    mn = 0;
                    for(p = 0; p < *P; p++){
                        mn+= beta[p] * X[i + *N * p];
                        s+= alpha[*L * p] * X[i + *N * p];
                        }
                }
                else{
                    for(p = 0; p < *P; p++){
                        s += alpha[bin[i] + *L * p] * X[i + *N * p];
                    }
                    mn = breaks[bin[i]] - s * q_kappa[bin[i]];
                    }
                log_p_ptr(&y[i], &mn, &s, shape, &log_like[i]);
                log_like[i] = log(1 - exp(log_like[i]));
                *ll_sum += log_like[i];
                }
            }
            if(*n4 > 0){ /*interval censored*/
                double s_low, s_high, mn_low, mn_high, tau_low, tau_high;
                for(i = *n1 + *n2 + *n3; i < *N; i++){ /*uncensored observations*/
                for(l = 1; l < (*L); l++){ /*element of breaks*/
                    breaks[l] = 0;
                    for(p = 0; p < *P; p++){ /*sum across all predictors*/
                        breaks[l] +=   (BM[l + (*L + 1) * p] + beta[p])*  X[i + *N * p];
                        }
                }
                find_int(&L_1, breaks, &y_low[i], &bin_low[i]);
                find_int(&L_1, breaks, &y_high[i], &bin_high[i]);
                s_low = 0;
                s_high = 0;
                if(bin_low[i]==0){
                    mn_low = 0;
                    for(p = 0; p < *P; p++){
                        mn_low += beta[p] * X[i + *N * p];
                        s_low += alpha[*L * p] * X[i + *N * p];
                        }
                }
                else{
                    for(p = 0; p < *P; p++){
                        s_low += alpha[bin_low[i] + *L * p] * X[i + *N * p];
                    }
                    mn_low = breaks[bin_low[i]] - s_low * q_kappa[bin_low[i]];
                    }
                if(bin_high[i]==0){
                    mn_high = 0;
                    for(p = 0; p < *P; p++){
                        mn_high += beta[p] * X[i + *N * p];
                        s_high += alpha[*L * p] * X[i + *N * p];
                        }
                }
                else{
                    for(p = 0; p < *P; p++){
                        s_high += alpha[bin_high[i] + *L * p] * X[i + *N * p];
                    }
                    mn_high = breaks[bin_high[i]] - s_high * q_kappa[bin_high[i]];
                    }
                log_p_ptr(&y_low[i], &mn_low, &s_low, shape, &tau_low);
                log_p_ptr(&y_high[i], &mn_high, &s_high, shape, &tau_high);
                if(tau_low == tau_high){log_like[i] = -99;}
                else{log_like[i] = log(exp(tau_high) - exp(tau_low));}
                *ll_sum += log_like[i];
            }
        }
}

/*thought this would be faster...*/
void clike_2 (int *N, int *n1, int *n2, double *kappa, int *P, int *L,
            double *X, double *y,
            double *beta, double *alpha,
            double *shape,
            void q_ptr(double *, double *, double *, double *, double*),
            void log_p_ptr(double *, double *, double *, double *, double*),
            void log_d_ptr(double *, double *, double *, double *, double*),
            double *B,
            int *bin, double *ll_sum, double *log_like
            ){
            int i, j, k, l, p;
            int L_1 = *L + 1;
            double s;
            double dummy_mean = 0;
            double dummy_scale = 1;

            double *BM = (double *)R_alloc((*L + 1) * *P, sizeof(double*));
            double *BREAKS = (double *)R_alloc(*N * (*L + 1), sizeof(double*));
            double *MN = (double *)R_alloc(*N, sizeof(double*));

            /**********construct the break matrix************/
            /*start with the first row*/
            for(j = 0; j < *P; j++){
                BM[(*L+1) * j] = B[0] * alpha[*L * j];
                }
            /*compute the rest of the rows*/
            for(i = 1; i < (*L + 1); i++){
                for(j = 0; j < *P; j++){
                    BM[i + (*L+1) * j] = 0;
                        for(k = 0; k < i; k++){
                            BM[i + (*L + 1) * j] += B[i + (*L + 1) * k] * alpha[k + *L * j];
                        }
                }
            }
            int T_A = 0;
            int T_B = 0;
            int dummy_one = 1;
            double dummy_alpha = 1;
            double dummy_beta = 0;
            MM (&T_A, &T_B, N, &dummy_one, P, &dummy_alpha, X, beta, &dummy_beta, MN);

            k = 0;
            for(j = 0; j < L_1; j++){
                for(i = 0; i < *N; i++){
                    BREAKS[k++] = MN[i];
                }
            }

            dummy_beta = 1;
            T_B = 1;
            MM (&T_A, &T_B, N, &L_1, P, &dummy_alpha, X, BM, &dummy_beta, BREAKS);

            find_int_2(&L_1, N, BREAKS, y, bin);

            double q_kappa[L_1];
            for(l = 0; l < L_1; l++){q_ptr(&kappa[l], &dummy_mean, &dummy_scale, shape, &q_kappa[l]);}


            *ll_sum = 0;

            if(*n1 > 0){/*obsevations censored below*/
                    for(i = 0; i < *n1; i++){
                        s = 0;
                        for(p = 0; p < *P; p++){
                            s+= alpha[bin[i] + *L * p] * X[i + *N * p];
                        }
                if(bin[i] > 0){
                    MN[i] = BREAKS[i + *N * bin[i]] - s * q_kappa[bin[i]];
                }
                    log_p_ptr(&y[i], &MN[i], &s, shape, &log_like[i]);
                    *ll_sum += log_like[i];
                }
            }
            /*uncensored observations*/
            for(i = *n1; i < *n2; i++){
                s = 0;
                    for(p = 0; p < *P; p++){
                        s += alpha[bin[i] + *L * p] * X[i + *N * p];
                    }
                if(bin[i] > 0){
                    MN[i] = BREAKS[i + *N * bin[i]] - s * q_kappa[bin[i]];
                }
                log_d_ptr(&y[i], &MN[i], &s, shape, &log_like[i]);
                *ll_sum += log_like[i];
            }
            if(*n2 < *N){ /*observations censored above*/
                for(i = *n2; i < *N; i++){
                s = 0;
                    for(p = 0; p < *P; p++){
                        s+= alpha[bin[i] + *L * p] * X[i + *N * p];
                    }
                if(bin[i] > 0){
                    MN[i] = BREAKS[i + *N * bin[i]] - s * q_kappa[bin[i]];
                }
                log_p_ptr(&y[i], &MN[i], &s, shape, &log_like[i]);
                log_like[i] = log(1 - exp(log_like[i]));
                *ll_sum += log_like[i];
                }
            }
}


/*thought this would be faster...*/
void clike_3 (int *N, int *n1, int *n2, double *kappa, int *P, int *L,
            double *X, double *y,
            double *beta, double *alpha,
            double *shape,
            void q_ptr(double *, double *, double *, double *, double*),
            void log_p_ptr(double *, double *, double *, double *, double*),
            void log_d_ptr(double *, double *, double *, double *, double*),
            double *B,
            int *bin, double *ll_sum, double *log_like
            ){
            int i, j, k, l;
            int L_1 = *L + 1;
            double dummy_mean = 0;
            double dummy_scale = 1;

            double *BREAKS = (double *)R_alloc(*N * (*L + 1), sizeof(double*));
            double *MN = (double *)R_alloc(*N, sizeof(double*));
            double *S = (double *)R_alloc(*N * *L, sizeof(double*));

            int T_A = 0;
            int T_B = 0;
            int dummy_one = 1;
            double dummy_alpha = 1;
            double dummy_beta = 0;
            MM (&T_A, &T_B, N, &dummy_one, P, &dummy_alpha, X, beta, &dummy_beta, MN);

            T_B = 1;
            MM (&T_A, &T_B, N, L, P, &dummy_alpha, X, alpha, &dummy_beta, S);

            T_B = 1;
            MM (&T_A, &T_B, N, &L_1, L, &dummy_alpha, S, B, &dummy_beta, BREAKS);

            k = 0;
                for(j = 0; j < L_1; j++){
                    for(i = 0; i < *N; i++){
                        BREAKS[k++] += MN[i];
                    }
                }

            find_int_2(&L_1, N, BREAKS, y, bin);

            *ll_sum = 0;

            double q_kappa[L_1];
            for(l = 0; l < L_1; l++){q_ptr(&kappa[l], &dummy_mean, &dummy_scale, shape, &q_kappa[l]);}

            if(*n1 > 0){/*obsevations censored below*/
                    for(i = 0; i < *n1; i++){
                    if(bin[i] > 0){
                        MN[i] = BREAKS[i + *N * bin[i]] - S[i + *N * bin[i]] * q_kappa[bin[i]];
                    }
                    log_p_ptr(&y[i], &MN[i], &S[i + *N * bin[i]], shape, &log_like[i]);
                    *ll_sum += log_like[i];
                }
            }
            /*uncensored observations*/
            for(i = *n1; i < *n2; i++){
                if(bin[i] > 0){
                    MN[i] = BREAKS[i + *N * bin[i]] - S[i + *N * bin[i]] * q_kappa[bin[i]];
                }
                log_d_ptr(&y[i], &MN[i], &S[i + *N * bin[i]], shape, &log_like[i]);
                *ll_sum += log_like[i];
            }
            if(*n2 < *N){ /*observations censored above*/
                for(i = *n2; i < *N; i++){
                if(bin[i] > 0){
                    MN[i] = BREAKS[i + *N * bin[i]] - S[i + *N * bin[i]] * q_kappa[bin[i]];
                }
                log_p_ptr(&y[i], &MN[i], &S[i + *N * bin[i]], shape, &log_like[i]);
                log_like[i] = log(1 - exp(log_like[i]));
                *ll_sum += log_like[i];
                }
            }
}

void clike_3_wrapper (int *N, int *n1, int *n2, double *kappa, int *P, int *L,
            double *X, double *y,
            double *beta, double *alpha,
            double *shape,
            double *B,
            int *bin, double *ll_sum, double *log_like,
            int *indicator){

    B[0] = -100000000;
    B[*L + (*L + 1) * (*L - 1)] = 100000000;


    void (*q_ptr)(double *, double *, double *, double *, double*); /*pointer to the quantile function*/
    void (*log_p_ptr)(double *, double *, double *, double *, double*); /*pointer to the log CDF*/
    void (*log_d_ptr)(double *, double *, double *, double *, double*); /*pointer to the log density*/

    if(*indicator == 0){
        q_ptr               = &q_norm;
        log_p_ptr           = &log_p_norm;
        log_d_ptr           = &log_d_norm;
    }
    else if(*indicator == 1){
        q_ptr               = &q_t;
        log_p_ptr           = &log_p_t;
        log_d_ptr           = &log_d_t;
    }
    else if(*indicator == 2){
        q_ptr               = &q_logistic;
        log_p_ptr           = &log_p_logistic;
        log_d_ptr           = &log_d_logistic;
    }
    else {
        q_ptr               = &q_alap;
        log_p_ptr           = &log_p_alap;
        log_d_ptr           = &log_d_alap;
    }

    clike_3(N, n1, n2, kappa, P, L, X, y,
           beta, alpha,
           shape,
           q_ptr, log_p_ptr, log_d_ptr,
           B,
           bin, ll_sum, log_like);
}


/*function that makes the lth row of alpha proper*/
void threshold_cens(int *L, int *P, int *l, double *alphastar, double *alpha, int *slope_update){
    int p;
    *slope_update = 0;
    for(p = 0; p < *P; p++){alpha[*l + *L * p] = alphastar[*l + *L * p];}
    if(alpha[*l] < 0){alpha[*l] = 0.001;}
    double neg_part = alpha[*l];
    for(p = 1; p < *P; p++){
        if(alphastar[*l + *L * p] < 0){neg_part +=  alphastar[*l + *L * p];}
        else{neg_part -= alphastar[*l + *L * p];}
    }
    if(neg_part < 0){
        *slope_update = 1;
        for(p = 1; p < *P; p++){
            alpha[*l + *L * p] = 0;
        }
    }
}

void update_shape_logit( double *tuning_shape, double *Z, double *E,
                        double *shapeprec, int *ACC_SHAPE,
                        int *N, int *n1, int *n2, int *n3, int *n4, double *kappa, int *P, int *L,
                        double *X, double *y, double *y_low, double *y_high,
                        double *beta, double *alpha,
                        double *shape,
                        void q_ptr(double *, double *, double *, double *, double*),
                        void log_p_ptr(double *, double *, double *, double *, double*),
                        void log_d_ptr(double *, double *, double *, double *, double*),
                        double *B,
                        int *bin, int *bin_low, int *bin_high, double *ll_sum, double *log_like
                        ){
            double *can_B  = (double *)R_alloc((*L+1) * *L, sizeof(double*)); /*candidate basis matrix*/
            double eta = log(*shape) - log(1 - *shape);
            double caneta =  eta + *tuning_shape * *Z;
            double canshape = 1 / (1 + exp(-caneta));
            double can_ll_sum = 0;
            /*construct the candidate basis matrix*/
            make_B(L, kappa, q_ptr, &canshape, can_B);
            /*candidate likelihood*/
            clike (N, n1, n2, n3, n4, kappa, P, L, X, y, y_low, y_high,
                beta, alpha,
                &canshape,
                q_ptr, log_p_ptr, log_d_ptr,
                can_B, bin, bin_low, bin_high, &can_ll_sum, log_like
                );
            /*I scaled the precision above*/
            if (*E > (*ll_sum - can_ll_sum  + *shapeprec * ( caneta * caneta - eta * eta ))){
                *shape = canshape;
                make_B(L, kappa, q_ptr, shape, B);
                *ll_sum = can_ll_sum;
                (*ACC_SHAPE) ++;
            }
}

void update_shape_log( double *tuning_shape, double *Z, double *E,
                        double *shapeprec, int *ACC_SHAPE,
                        int *N, int *n1, int *n2, int *n3, int *n4, double *kappa, int *P, int *L,
                        double *X, double *y, double *y_low, double *y_high,
                        double *beta, double *alpha,
                        double *shape,
                        void q_ptr(double *, double *, double *, double *, double*),
                        void log_p_ptr(double *, double *, double *, double *, double*),
                        void log_d_ptr(double *, double *, double *, double *, double*),
                        double *B,
                        int *bin, int *bin_low, int *bin_high, double *ll_sum, double *log_like
                        ){
            double *can_B  = (double *)R_alloc((*L+1) * *L, sizeof(double*)); /*candidate basis matrix*/
            double canshape =  exp(*tuning_shape * rnorm(0,1) +  log(*shape));
            double can_ll_sum;
            /*construct the candidate basis matrix*/
            make_B(L, kappa, q_ptr, &canshape, can_B);
            /*candidate likelihood*/
            clike(N, n1, n2, n3, n4, kappa, P, L, X, y, y_low, y_high,
                beta, alpha,
                &canshape,
                q_ptr, log_p_ptr, log_d_ptr,
                can_B, bin, bin_low, bin_high, &can_ll_sum, log_like
                );
            /*I scaled the precision above*/
            if (*E > (*ll_sum - can_ll_sum  + *shapeprec * ( log(canshape) * log(canshape) - log(*shape) * log(*shape)))){
                *shape = canshape;
                make_B(L, kappa, q_ptr, shape, B);
                *ll_sum = can_ll_sum;
                (*ACC_SHAPE) ++;
            }
}

void update_shape_null( double *tuning_shape, double *Z, double *E,
                        double *shapeprec, int *ACC_SHAPE,
                        int *N, int *n1, int *n2, int *n3, int *n4, double *kappa, int *P, int *L,
                        double *X, double *y, double *y_low, double *y_high,
                        double *beta, double *alpha,
                        double *shape,
                        void q_ptr(double *, double *, double *, double *, double*),
                        void log_p_ptr(double *, double *, double *, double *, double*),
                        void log_d_ptr(double *, double *, double *, double *, double*),
                        double *B,
                        int *bin, int *bin_low, int *bin_high, double *ll_sum, double *log_like
                        )
                        {}
