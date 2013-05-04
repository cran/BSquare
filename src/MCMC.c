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
#include "helpers.h"

/*
n1 is the number of observations that were censored below
n2 - n1 is the number of uncensored observations
n - n2 is the number of observations that were censored above

P is the number of predictors
P1 is the subset of predictors that affect the shape
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

void MCMC(int *burn, int *sweeps,
          double *tuning_alpha, double *tuning_beta, double *tuning_shape,
          double *beta_eps, double *alpha_eps,
          int *basis_ind, int *L,
          int *N, int *n1, int *n2, int *n3, int *n4, int *P, int *P1,
          double *X, double *y, double *y_low, double *y_high,
          double *beta, double *alphastar, double *shape,
          double *mu, double *sigma2, double *rho,
          double *beta_var, double *shape_var,
          double *mu_var, double *sig_a,double *sig_b,
          double BETA[*sweeps][*P],
          double ALPHA[*sweeps][*L * *P],
          double MU[*sweeps][*P],
          double SIGMA2[*sweeps][*P],
          double RHO[*sweeps][*P],
          double SHAPE[*sweeps],
          double *LPML,
          int ACC_BETA[*P],
          int ACC_ALPHA[*L * *P],
          int ACC_RHO[*P1],
          int *ACC_SHAPE,
          int ATT_BETA[*P],
          int ATT_ALPHA[*L * *P],
          int *verbose
           ){


///***************** Part I: allocate memory, initialize stuff *********************/
    double *alpha = (double *)Calloc(*L * *P, double);
    double *canalphastar = (double *)Calloc(*L * *P, double);
    double *canalpha = (double *)Calloc(*L * *P, double);
    double *canbeta = (double *)Calloc(*P, double);
    double *log_like = (double *)Calloc(*N, double);


    double ll_sum, can_ll_sum, canrho, canrhoquad, logdet, canlogdet;

    double *B  = (double *)Calloc((*L+1) * *L, double); /*basis matrix*/
    double *can_B  = (double *)Calloc((*L+1) * *L, double); /*candidate basis matrix*/
    double kappa[*L + 1]; /*knots*/

    double *CPO  = (double *)Calloc(*N, double); /*candidate basis matrix*/

    int nsamps_int, slope_update;

    int i, j, k, l, p;
    double E, Z;

    int *bin = (int *)Calloc(*N, int);
    int *bin_low = (int *)Calloc(*N, int);
    int *bin_high = (int *)Calloc(*N, int);



    for(i = 0; i < *N; i++){
        bin[i] = 1;
        bin_low[i] = 1;
        bin_high[i] = 1;
    }


    /*construct pointers to the quantile function, log cdf and log density*/

    void (*q_ptr)(double *, double *, double *, double *, double*); /*pointer to the quantile function*/
    void (*log_p_ptr)(double *, double *, double *, double *, double*); /*pointer to the log CDF*/
    void (*log_d_ptr)(double *, double *, double *, double *, double*); /*pointer to the log density*/

  /*pointer to the update shape function*/
    void (*update_shape_ptr)( double *, double *, double *,
                        double *, int *,
                        int *, int *, int *, int *, int *, double *, int *, int *,
                        double *, double *, double *, double *,
                        double *, double *,
                        double *,
                        void (double *, double *, double *, double *, double*),
                        void (double *, double *, double *, double *, double*),
                        void (double *, double *, double *, double *, double*),
                        double *,
                        int *, int *, int *, double *, double *
                        );


    if(*basis_ind == 0){
        q_ptr               = &q_norm;
        log_p_ptr           = &log_p_norm;
        log_d_ptr           = &log_d_norm;
        update_shape_ptr    = &update_shape_null;

    }
    else if(*basis_ind == 1){
        q_ptr               = &q_t;
        log_p_ptr           = &log_p_t;
        log_d_ptr           = &log_d_t;
        update_shape_ptr    = &update_shape_log;
    }
    else if(*basis_ind == 2){
        q_ptr               = &q_logistic;
        log_p_ptr           = &log_p_logistic;
        log_d_ptr           = &log_d_logistic;
        update_shape_ptr    = &update_shape_null;

    }
    else if(*basis_ind == 3){
        q_ptr               = &q_alap;
        log_p_ptr           = &log_p_alap;
        log_d_ptr           = &log_d_alap;
        update_shape_ptr    = &update_shape_logit;
    }

    else if(*basis_ind == 4){
        q_ptr               = &q_weibull;
        log_p_ptr           = &log_p_weibull;
        log_d_ptr           = &log_d_weibull;
        update_shape_ptr    = &update_shape_log;
    }
    else{
        q_ptr               = &q_gamma;
        log_p_ptr           = &log_p_gamma;
        log_d_ptr           = &log_d_gamma;
        update_shape_ptr    = &update_shape_log;
    }

    /*construct knots*/
    for(k = 0; k < (*L + 1); k++){
        kappa[k] = k /  (double) *L ;
    }


    /*populate alpha, canalpha canalphastar*/
    for(p = 0; p < *P; p++){
        for(l = 0; l < *L; l++){
            alpha[l + *L * p]        = alphastar[l + *L * p];
            canalpha[l + *L * p]     = alphastar[l + *L * p];
            canalphastar[l + *L * p] = alphastar[l + *L * p];
        }
    }

    /*turn alpha and canalpha into valid process*/
    for(l = 0; l < *L; l++){
        threshold_cens(L, P, &l, alphastar, alpha, &slope_update);
        threshold_cens(L, P, &l, canalphastar, canalpha, &slope_update);
    }

    /*define the elements of the basis matrix that are constant through the MCMC*/
    for(i = 0; i < (*L + 1); i++){
        for(j = 0; j < *L; j++){
        B[i + (*L + 1) * j] = 0;
        }
    }
    /*initialize the basis matrix*/
    make_B(L, kappa, q_ptr, shape, B);
    /*initialize the candidate basis matrix*/
    make_B(L, kappa, q_ptr, shape, can_B);
    /* initialize the likelihood*/

    clike(N, n1, n2, n3, n4, kappa, P, L,
           X, y, y_low, y_high,
           beta, alpha,
           shape,
           q_ptr, log_p_ptr, log_d_ptr,
           B,
           bin, bin_low, bin_high, &ll_sum, log_like
           );

  	double rhoquad [*P];   /*quadratic terms in likelihood for rho*/
  	double sumprec[*P];    /*sums of the precision matrices*/
    double sumalpha[*P]; /*t(one)%*%PREC%*%alpha*/
    double betaprec = 1 /  (2 * *beta_var);  /*scaled prior precision for constant basis functions*/
    double shapeprec = 1 / (2 * *shape_var); /*scaled prior precision for shape parameter*/
    double muprec = 1 / *mu_var;

    /*initialize rhoquad and sigma2 in case L = 1*/
    for(p = 0; p < *P; p++){
        rhoquad[p] = 0;
        sigma2[p] = 2 * muprec;
    }
    canrhoquad = 1;
///***************** Part II: burn *********************/

    if(*verbose == 1){Rprintf("burn, burn, burn... \n");}

    GetRNGstate();
    for(i = 0; i< *burn; i++){
        if(*basis_ind < 4){
            for(p = 0; p < *P; p++){
            /*update location parameters*/
            nsamps_int = rgeom(*beta_eps);
            while(nsamps_int > 0){
                ATT_BETA[p]++;
                /*draw a candidate*/
                canbeta[p] = tuning_beta[p] * rnorm(0,1) + beta[p];
                /*evaluate the candidate likelihood*/
                clike(N, n1, n2, n3, n4, kappa, P, L,
                    X, y, y_low, y_high,
                    canbeta, alpha,
                    shape,
                    q_ptr, log_p_ptr, log_d_ptr,
                    B,
                    bin, bin_low, bin_high, &can_ll_sum, log_like);
                /*I scaled the precision above*/
                if (rexp(1) > (ll_sum - can_ll_sum  + betaprec * (canbeta[p] * canbeta[p] - beta[p] * beta[p]))){
                    beta[p] = canbeta[p];
                    ll_sum = can_ll_sum;
                    ACC_BETA[p] ++ ;
                }
                /*restore canbeta to beta*/
                else{canbeta[p] = beta[p];}
                    nsamps_int -- ;
                }
            }
        }
        for(p = 0; p < *P1; p++){
            /*update mu, sigma2, rho*/
            if(*L > 1){
                sum_alpha(&rho[p], L, P, &p, alphastar, &sumalpha[p], &sumprec[p]);
                /*Gibbs sample mu, the mean of the basis functions by predictor across gestational age*/
                mu[p] = pow((sigma2[p] * sumprec[p] + muprec),-0.5) * rnorm(0,1) + (sigma2[p] * sumalpha[p])/(sigma2[p] * sumprec[p] + muprec);
                /*update the quadratic term for rho*/
                rho_quad_cens(&rho[p], L, P, &p, alphastar, mu, &rhoquad[p]);
                /*Gibbs sample precision sigma2 */
                sigma2[p] = rgamma(*L / 2 + *sig_a, (2 * *sig_b)/(*sig_b * rhoquad[p] + 2));
                /*Independent Metropolis sample rho*/
                log_det_AR(&rho[p], L, &logdet); /*dimension of precision matrix is L - 1*/
                canrho = runif(0.05,.95);
                log_det_AR(&canrho, L, &canlogdet);
                rho_quad_cens(&canrho, L, P, &p, alpha, mu, &canrhoquad);
                if (rexp(1) > 0.5*(logdet -  canlogdet    - sigma2[p] * (rhoquad[p] - canrhoquad))){
                    rho[p] = canrho;
                    rhoquad[p] = canrhoquad;
                }
            }
             //update scale parameters
            for(l = 0; l < *L; l++){
                nsamps_int = rgeom(*alpha_eps);
                while(nsamps_int > 0){
                    ATT_ALPHA[l + *L * p] ++;
                    canalphastar[l + *L * p] = tuning_alpha[l + *L * p] * rnorm(0,1) + alphastar[l + *L * p]; /*update latent process*/
                    threshold_cens(L, P, &l, canalphastar, canalpha, &slope_update); /*may need P instead of P1*/
                    clike(N, n1, n2, n3, n4, kappa, P, L,
                    X, y, y_low, y_high,
                    beta, canalpha,
                    shape,
                    q_ptr, log_p_ptr, log_d_ptr,
                    B,
                    bin, bin_low, bin_high, &can_ll_sum, log_like);

                    if(*L > 1){rho_quad_cens(&rho[p], L, P, &p, canalphastar, mu, &canrhoquad);}
                    if (rexp(1) > ( 0.5 * sigma2[p] * (canrhoquad - rhoquad[p]) + ll_sum  - can_ll_sum)){
                        /*accept candidate*/
                        if(slope_update == 1){
                            for(j = 0; j < *P; j++){
                                alpha[l + *L * j] = canalpha[l + *L * j];
                            }
                        }
                        alphastar[l + *L * p] = canalphastar[l + *L * p];
                        alpha[l + *L * p] = canalpha[l + *L * p];
                        ll_sum = can_ll_sum;
                        rhoquad[p] = canrhoquad;
                        ACC_ALPHA[l + *L * p] ++ ;
                        }
                        /*reject candidate*/
                        else{
                            canalphastar[l + *L * p] = alphastar[l + *L * p];
                            canalpha[l + *L * p] = alpha[l + *L * p];
                            if(slope_update == 1){
                                for(j = 0; j < *P; j++){
                                    canalpha[l + *L * j] = alpha[l + *L * j];
                                }
                            }
                        }
                        nsamps_int -- ;
                }
            }
        }
        /*update the shape*/
        Z = rnorm(0,1);
        E = rexp(1);
        update_shape_ptr(tuning_shape, &Z, &E,
                        &shapeprec, ACC_SHAPE,
                        N, n1, n2, n3, n4, kappa, P, L,
                        X, y, y_low, y_high,
                        beta, alpha,
                        shape,
                        q_ptr,
                        log_p_ptr,
                        log_d_ptr,
                        B,
                        bin, bin_low, bin_high, &ll_sum, log_like
                        );

/*        adjust candidate standard deviations */
        if ((i+1) % 100 == 0 ){
            for(p = 0; p < *P; p++){
                if (ACC_BETA[p]  > (0.5 * ATT_BETA[p])){tuning_beta[p] *= 1.2;}
                else if (ACC_BETA[p]  < (0.3 * ATT_BETA[p])){tuning_beta[p] *= 0.8;}
                ACC_BETA[p] = 0;
                ATT_BETA[p] = 0;
                }
        for(p = 0; p < *P1; p++){
            for(l = 0; l < *L; l++){
                if (ACC_ALPHA[l + *L * p]  > (0.5 *  ATT_ALPHA[l + *L * p])){tuning_alpha[l + *L * p] *= 1.2;}
                else if (ACC_ALPHA[l + *L * p]  < (0.3 * ATT_ALPHA[l + *L * p])){tuning_alpha[l + *L * p] *= 0.8;}
                    ACC_ALPHA[l + *L * p] = 0;
                    ATT_ALPHA[l + *L * p] = 0;
                }
            }
        if(*ACC_SHAPE > 50){*tuning_shape *= 1.2;}
        else if(*ACC_SHAPE < 30){*tuning_shape *= 0.8;}
        *ACC_SHAPE = 0;
        }
    }


/***************** Part III: keep *********************/
//reset acceptance probabilities
    for(p = 0; p < *P; p++){
        ACC_BETA[p] = 0;
        ATT_BETA[p] = 0;
        for(l = 0; l < *L; l++){
            ACC_ALPHA[l + *L * p] = 0;
            ATT_ALPHA[l + *L * p] = 0;
        }
    }
    *ACC_SHAPE = 0;

    if(*verbose == 1){Rprintf("Burn-in Finished. \n");}

    for(i = 0; i < *sweeps; i++){
        if(*verbose == 1){
            if((i) % 1000 == 0 ){
                Rprintf("Keepers %d %% done.", 100 * i /  *sweeps );
                Rprintf("\n");
            }
        }

        if(*basis_ind < 4){
            for(p = 0; p < *P; p++){
            /*update location parameters*/
            nsamps_int = rgeom(*beta_eps);
//             = (int) nsamps;
            while(nsamps_int > 0){
                ATT_BETA[p]++;
                /*draw a candidate*/
                canbeta[p] = tuning_beta[p] * rnorm(0,1) + beta[p];
                /*evaluate the candidate likelihood*/
                clike(N, n1, n2, n3, n4, kappa, P, L,
                    X, y, y_low, y_high,
                    canbeta, alpha,
                    shape,
                    q_ptr, log_p_ptr, log_d_ptr,
                    B,
                    bin, bin_low, bin_high, &can_ll_sum, log_like);
                /*I scaled the precision above*/
                if (rexp(1) > (ll_sum - can_ll_sum  + betaprec * (canbeta[p] * canbeta[p] - beta[p] * beta[p]))){
                    beta[p] = canbeta[p];
                    ll_sum = can_ll_sum;
                    ACC_BETA[p] ++ ;
                }
                /*restore canbeta to beta*/
                else{canbeta[p] = beta[p];}
                    nsamps_int -- ;
            }
            BETA[i][p] = beta[p];
            }
        }
        /*update mu, sigma2, rho*/
        for(p = 0; p < *P1; p++){
            if(*L > 1){
                sum_alpha(&rho[p], L, P, &p, alphastar, &sumalpha[p], &sumprec[p]);
                /*Gibbs sample mu, the mean of the basis functions by predictor across gestational age*/
                mu[p] = pow((sigma2[p] * sumprec[p] + muprec),-0.5) * rnorm(0,1) + (sigma2[p] * sumalpha[p])/(sigma2[p] * sumprec[p] + muprec);
                /*update the quadratic term for rho*/
                rho_quad_cens(&rho[p], L, P, &p, alphastar, mu, &rhoquad[p]);
                /*Gibbs sample precision sigma2 */
                sigma2[p] = rgamma(*L / 2 + *sig_a, (2 * *sig_b)/(*sig_b * rhoquad[p] + 2));
                /*Independent Metropolis sample rho*/
                log_det_AR(&rho[p], L, &logdet); /*dimension of precision matrix is L - 1*/
                canrho = runif(0.05,.95);
                log_det_AR(&canrho, L, &canlogdet);
                rho_quad_cens(&canrho, L, P, &p, alpha, mu, &canrhoquad);
                if (rexp(1) > 0.5*(logdet -  canlogdet    - sigma2[p] * (rhoquad[p] - canrhoquad))){
                    rho[p] = canrho;
                    rhoquad[p] = canrhoquad;
                    ACC_RHO[p] ++;
                }
                /*store parameters*/
                MU[i][p] = mu[p];
                SIGMA2[i][p] = sigma2[p];
                RHO[i][p] = rho[p];
            }
             //update scale parameters
            for(l = 0; l < *L; l++){

                nsamps_int = rgeom(*alpha_eps);
                while(nsamps_int > 0){
                    ATT_ALPHA[l + *L * p] ++;
                    canalphastar[l + *L * p] = tuning_alpha[l + *L * p] * rnorm(0,1) + alphastar[l + *L * p]; /*update latent process*/
                    threshold_cens(L, P, &l, canalphastar, canalpha, &slope_update);
                    clike(N, n1, n2, n3, n4, kappa, P, L,
                    X, y, y_low, y_high,
                    beta, canalpha,
                    shape,
                    q_ptr, log_p_ptr, log_d_ptr,
                    B,
                    bin, bin_low, bin_high, &can_ll_sum, log_like);
                    if(*L > 1){rho_quad_cens(&rho[p], L, P, &p, canalphastar, mu, &canrhoquad);}
                    if (rexp(1) > ( 0.5 * sigma2[p] * (canrhoquad - rhoquad[p]) + ll_sum  - can_ll_sum)){
                        if(slope_update == 1){
                            for(j = 0; j < *P; j++){
                                alpha[l + *L * j] = canalpha[l + *L * j];
                            }
                        }
                        alphastar[l + *L * p] = canalphastar[l + *L * p];
                        alpha[l + *L * p] = canalpha[l + *L * p];
                        ll_sum = can_ll_sum;
                        rhoquad[p] = canrhoquad;
                        ACC_ALPHA[l + *L * p] ++ ;
                    }
                    /*restore canalpha to alpha*/
                    else{
                        canalphastar[l + *L * p] = alphastar[l + *L * p];
                        canalpha[l + *L * p] = alpha[l + *L * p];
                        if(slope_update == 1){
                            for(j = 0; j < *P; j++){
                                canalpha[l + *L * j] = alpha[l + *L * j];
                            }
                        }
                    }
                        nsamps_int -- ;
                }
                ALPHA[i][l + *L * p] = alpha[l + *L * p];
            }
        }
        /*update the shape*/
        Z = rnorm(0,1);
        E = rexp(1);
        update_shape_ptr(tuning_shape, &Z, &E,
                        &shapeprec, ACC_SHAPE,
                        N, n1, n2, n3, n4, kappa, P, L,
                        X, y, y_low, y_high,
                        beta, alpha,
                        shape,
                        q_ptr,
                        log_p_ptr,
                        log_d_ptr,
                        B,
                        bin, bin_low, bin_high, &ll_sum, log_like
                        );
        SHAPE[i] = *shape;

        clike(N, n1, n2, n3, n4, kappa, P, L,
            X, y, y_low, y_high,
            beta, alpha,
            shape,
            q_ptr, log_p_ptr, log_d_ptr,
            B,
            bin, bin_low, bin_high, &ll_sum, log_like);

        for(k = 0; k < *N; k++){
            CPO[k] += exp(-log_like[k]);
        }
    }
    for(k = 0; k < *N; k++){
        CPO[k] = *sweeps / CPO[k];
        *LPML += log(CPO[k]);
        }
    PutRNGstate();

    Free(alpha);
    Free(canalphastar);
    Free(canalpha);
    Free(canbeta);
    Free(log_like);

    Free(B);
    Free(can_B);
    Free(canbeta);
    Free(log_like);

    Free(bin);
    Free(bin_low);
    Free(bin_high);
    Free(CPO);

}

