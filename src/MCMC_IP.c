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



void MCMC_IP(int *burn, int *iters,
        double *tuning_parms, double *tuning_tail,
        double *cbf_eps, double *theta_eps,
        int *M, int *P, int *P1, int *N, int *n1, int *n2, int *n3, int *n4, int *n5,
        double *y, double *y_low, double *y_high, double *X,
        int *M_knots_length, int *I_knots_length, int *spline_df,
        double *M_knots,double *I_knots,
        double *M_low, double *M_high, double *I_low, double *I_high,
        double *thetastar, double *mu, double *sigma2, double *rho, double *xi_low, double *xi_high,
        double *q_low, double *q_high,
        int *xi_zero,
        double *mu_var, double *cbf_var, double *tail_mean, double *tail_var, double *sig_a, double *sig_b,
        double THETA[*iters][(*M) * (*P)],
        double MU[*iters][*P],
        double SIGMA2[*iters][*P],
        double RHO[*iters][*P],

        double XI_LOW[*iters],
        double XI_HIGH[*iters],
        double *LPML,

        int ACC_THETA[*M * *P],
        int ATT_THETA[*M * *P],

        int ACC_RHO[*P],
        int ACC_TAIL[2],

        double *IKM, double *MKM, int *M_1,
        int *verbose

){
    /*************************I: Declare variables, initialize stuff **********************/
    double *tau = (double *)Calloc(*N, double);
    double *tau_low = (double *)Calloc(*N, double);
    double *tau_high = (double *)Calloc(*N, double);
    double *theta = (double *)Calloc(*M * *P, double);
    double *cantheta = (double *)Calloc(*M * *P, double);
    double *canthetastar = (double *)Calloc(*M * *P, double);
    int *bin = (int *)Calloc(*N, int);
    int *bin_low = (int *)Calloc(*N, int);
    int *bin_high = (int *)Calloc(*N, int);
    double *log_like = (double *)Calloc(*N, double);
    int i, j, k, m, n, p, slope_update, nsamps_int;
    double ll_sum, can_ll_sum;
    double E_vec[2], Z_vec[2];

    double *reset_value = (double *)Calloc(29, double);

    for(n = 0; n < 10; n++){
    reset_value[n] = *q_low + 0.001 * (n + 1);
    }
    for(n = 10; n < 19; n++){
    reset_value[n] = 0.1 * (n - 9);
    }
    for(n = 19; n < 29; n++){
    reset_value[n] = *q_high + 0.001 * (19 - n);
    }

    double *CPO  = (double *)Calloc(*N, double); /*candidate basis matrix*/

/*    initialize tau and bin*/
    for (n = 0; n < *N; n++){
        tau[n] = 0.5;
        tau_low[n] = 0.5;
        tau_high[n] = 0.5;
        bin[n] = 1;
        bin_low[n] = 1;
        bin_high[n] = 1;
        CPO[n] = 0;
    }

    /*pointer to the whether or not the tail is updated*/
      void (*tail_ptr)(
        int *, int *,
        int *, int *, int *, int *, int *, int *,
        double *, double *y, double *y_low, double *,
        int *, int *, int *, double *, double *,
        double *, double *, double *, double *,
        double *, double *, int *,
        double *,double *, double *,
        double *, double *, double *, double *, double *,
        int *, int *, int *,
        void (double *, double *, double *, double *, double*), /*pointer to the log density for low obs*/
        void (double *, double *, double *, double *, double*), /*pointer to the log density for high obs*/
        void (double *, double *, double *, double *, double*), /*pointer to the cdf*/
        void (double *, double *, double *, double *, double*),
        double *, double *, int *, double *,
        double *,
        double *,
        double *,
        double *,
        double *,
        int *);

    /*pointer to tail densities*/
        void (*ld_low_ptr)(double *, double *, double *, double *, double*); /*pointer to the log density for low obs*/
        void (*ld_high_ptr)(double *, double *, double *, double *, double*); /*pointer to the log density for high obs*/
        void (*tau_low_ptr)(double *, double *, double *, double *, double*); /*pointer to the cdf*/
        void (*tau_high_ptr)(double *, double *, double *, double *, double*); /*pointer to the cdf*/

    if(*xi_zero == 1){
        tail_ptr = &no_update_tail;
        ld_low_ptr   = &exponential_low;
        ld_high_ptr  = &exponential_high;
        tau_low_ptr  = &exponential_tau_low;
        tau_high_ptr = &exponential_tau_high;
    }
    else{
        tail_ptr = &update_tail;
        ld_low_ptr   = &pareto_low;
        ld_high_ptr  = &pareto_high;
        tau_low_ptr  = &pareto_tau_low;
        tau_high_ptr = &pareto_tau_high;
    }

    /*initialize parameters*/
    for (m = 0; m < *M; m++){
        for(p = 0; p < *P; p++){
            theta[m + *M * p] = thetastar[m + *M * p];
            cantheta[m + *M * p] = thetastar[m + *M * p];
            canthetastar[m + *M * p] = thetastar[m + *M * p];
        }
    }
    for(m = 1; m < *M; m++){
            threshold(M, P, &m, thetastar, theta, &slope_update);
            threshold(M, P, &m, canthetastar, cantheta, &slope_update);
    }

      ll(M, P, N, n1, n2, n3, n4, n5,
        X, y, y_low, y_high,
        M_knots_length, I_knots_length, spline_df, M_knots, I_knots,
        M_low, M_high, I_low, I_high,
        q_low, q_high, xi_zero,
        theta, xi_low, xi_high,
        tau, tau_low, tau_high, log_like, &ll_sum,
        bin, bin_low, bin_high,
        ld_low_ptr,
        ld_high_ptr,
        tau_low_ptr,
        tau_high_ptr,
        IKM, MKM, M_1, reset_value
        );
    /******** prior variables ************/

    double can_lp, logdet, rho_ll,  canlogdet, canrho_ll, canrho, canrhoquad;
  	double lp [*P];   /*log prior for each m x p combination*/
  	double rhoquad [*P];   /*quadratic terms in likelihood for rho*/
  	double sumprec[*P];    /*sums of the precision matrices*/
    double sumtheta[*P]; /*t(one)%*%PREC%*%theta*/

    double lp_cbf[*P]; /*log prior for the constant basis functions*/
    double muprec = 1/ *mu_var;
    double cbfprec= 1/ (2 * *cbf_var); /*note the 1/2 is incorporated here*/
    double tail_prec= 1/ (2 * *tail_var); /*note the 1/2 is incorporated here*/

    for(p = 0; p < *P; p++){
            lp_cbf[p] = -1 * cbfprec * thetastar[*M * p] * thetastar[*M * p];
        }
    /**********tail variables **************/
    if(*verbose == 1){Rprintf("burn, burn, burn... \n");}
//    start burn
    GetRNGstate();
    for(i = 0; i< *burn; i++){
        for(p = 0; p < *P; p++){
/*update constant basis functions*/
            nsamps_int = rgeom(*cbf_eps);
            while(nsamps_int > 0){
                (ATT_THETA[*M * p]) ++ ;
                canthetastar[*M * p] = tuning_parms[*M * p] * rnorm(0,1) + thetastar[*M * p];
                cantheta[*M * p] = canthetastar[*M * p];
                ll(M, P, N, n1, n2, n3, n4, n5,
                    X, y, y_low, y_high,
                    M_knots_length, I_knots_length, spline_df, M_knots, I_knots,
                    M_low, M_high, I_low, I_high,
                    q_low, q_high, xi_zero,
                    cantheta, xi_low, xi_high,
                    tau, tau_low, tau_high, log_like, &can_ll_sum,
                    bin, bin_low, bin_high,
                    ld_low_ptr,
                    ld_high_ptr,
                    tau_low_ptr,
                    tau_high_ptr,
                    IKM, MKM, M_1, reset_value);
                can_lp = -1 * cbfprec * canthetastar[*M * p] * canthetastar[*M * p];
                if (rexp(1) > (lp_cbf[p] + ll_sum - can_lp - can_ll_sum)){
                /*accept candidate*/
                    theta[*M * p] = cantheta[*M * p];
                    thetastar[*M * p] = canthetastar[*M * p];
                    ll_sum = can_ll_sum;
                    lp_cbf[p] = can_lp;
                    (ACC_THETA[*M * p]) ++;
                }
                /*reject candidate*/
                else{
                    cantheta[*M * p] = theta[*M * p];
                    canthetastar[*M * p] = thetastar[*M * p];
                }
                nsamps_int -- ;
            }
        }
        /*update predictors that have effects that change with quantile level*/
        for(p = 0; p < *P1; p++){
            /******update hyperparameters*****/
            /*Gibbs sample mu*/
            sum_theta_IP(&rho[p], M, P, &p, thetastar, &sumtheta[p], &sumprec[p]);
            mu[p] = pow((sigma2[p] * sumprec[p] + muprec),-0.5) * rnorm(0,1) + (sigma2[p] * sumtheta[p])/(sigma2[p] * sumprec[p] + muprec);
            /*update the quadratic term for rho*/
            rho_quad_IP(&rho[p], M, P, &p, thetastar, mu, &rhoquad[p]);
            /*Gibbs sample precision sigma2 */
            sigma2[p] = rgamma(*M_1 / 2 + *sig_a,(2 * *sig_b)/(*sig_b * rhoquad[p] + 2));
            /*Independent Metropolis sample rho*/
            log_det_AR(&rho[p], M_1, &logdet);
            rho_ll = 0.5 * (logdet - sigma2[p] * rhoquad[p]);
            canrho = runif(0,1);
            log_det_AR(&canrho, M_1, &canlogdet);
            rho_quad_IP(&canrho, M, P, &p, thetastar, mu, &canrhoquad);
            canrho_ll  = 0.5 * (canlogdet - sigma2[p] * canrhoquad);
            if (rexp(1) > (rho_ll - canrho_ll)){
                rho[p] = canrho;
                rhoquad[p] = canrhoquad;
            }
            /*	update prior for theta*/
            lp [p] = -0.5 * sigma2[p] * rhoquad [p];
            /*update theta*/
            for(m = 1; m < *M; m++){
                nsamps_int = rgeom(*theta_eps);
                while(nsamps_int > 0){
                    (ATT_THETA[m + *M * p]) ++;
                    /*draw a candidate*/
                    canthetastar[m + *M * p] = tuning_parms[m + *M * p] * rnorm(0,1) + thetastar[m + *M * p];
                    threshold(M, P, &m, canthetastar, cantheta, &slope_update);
                    /*candidate loglikelihood*/
                    ll(M, P, N, n1, n2, n3, n4, n5,
                    X, y, y_low, y_high,
                    M_knots_length, I_knots_length, spline_df, M_knots, I_knots,
                    M_low, M_high, I_low, I_high,
                    q_low, q_high, xi_zero,
                    cantheta, xi_low, xi_high,
                    tau, tau_low, tau_high, log_like, &can_ll_sum,
                    bin, bin_low, bin_high,
                    ld_low_ptr,
                    ld_high_ptr,
                    tau_low_ptr,
                    tau_high_ptr,
                    IKM, MKM, M_1, reset_value);

                    rho_quad_IP(&rho[p], M, P, &p, canthetastar, mu, &canrhoquad);
                    can_lp = -0.5 * sigma2[p] * canrhoquad;  /*update the log prior for theta*/
                    if (rexp(1) > (lp[p] + ll_sum - can_lp - can_ll_sum)){
                        thetastar[m + *M * p] = canthetastar[m + *M * p];
                        theta[m + *M * p] = cantheta[m + *M * p];
                        if(slope_update == 1){
                            for(j = 0; j < *P; j++){
                                theta[m + *M * j] = cantheta[m + *M * j] ;
                            }
                        }
                        ll_sum = can_ll_sum;
                        lp[p] = can_lp;
                        rhoquad[p] = canrhoquad;
                        (ACC_THETA[m + *M * p]) ++;
                    }
                        /*restore cantheta to theta*/
                        else{
                            canthetastar[m + *M * p] = thetastar[m + *M * p];
                                for(j = 0; j < *P; j++){
                                    cantheta[m + *M * j] = theta[m + *M * j];
                                }
                        }
                        nsamps_int -- ;
                    }
                }
            }

            /******update the tails******/
            Z_vec[0] = rnorm(0,1);
            Z_vec[1] = rnorm(0,1);

            E_vec[0] = rexp(1);
            E_vec[1] = rexp(1);


            tail_ptr(
            M, P,
            N, n1, n2, n3, n4, n5,
            X, y, y_low, y_high,
            M_knots_length, I_knots_length, spline_df, M_knots, I_knots,
            M_low, M_high, I_low, I_high,
            q_low, q_high, xi_zero,
            theta, xi_low, xi_high,
            tau, tau_low, tau_high, log_like, &ll_sum,
            bin, bin_low, bin_high,
            ld_low_ptr,
            ld_high_ptr,
            tau_low_ptr,
            tau_high_ptr,
            IKM, MKM, M_1, reset_value,
            tuning_tail,
            tail_mean,
            &tail_prec,
            E_vec,
            Z_vec,
            ACC_TAIL);

            ll(M, P, N, n1, n2, n3, n4, n5,
                X, y, y_low, y_high,
                M_knots_length, I_knots_length, spline_df, M_knots, I_knots,
                M_low, M_high, I_low, I_high,
                q_low, q_high, xi_zero,
                theta, xi_low, xi_high,
                tau, tau_low, tau_high, log_like, &ll_sum,
                bin, bin_low, bin_high,
                ld_low_ptr,
                ld_high_ptr,
                tau_low_ptr,
                tau_high_ptr,
                IKM, MKM, M_1, reset_value);

/*        adjust candidate standard deviations */
        if ((i+1) % 100 == 0 ){
            for(m = 0; m < *M; m++){
                for(p = 0; p < *P; p++){
                    if (ACC_THETA[m + *M * p] > 0.5 * ATT_THETA[m + *M * p]){tuning_parms[m + *M * p] *= 1.2;}
                        if (ACC_THETA[m + *M * p] < 0.3 * ATT_THETA[m + *M * p]){tuning_parms[m + *M * p] *= 0.8;}
                        ACC_THETA[m + *M * p] = 0;
                        ATT_THETA[m + *M * p] = 0;
                        }
                    }

            if (ACC_TAIL[0] > 50){tuning_tail[0] *=  1.2;}
            if (ACC_TAIL[0] < 20){tuning_tail[0] *=  0.8;}

            if (ACC_TAIL[1] > 50){tuning_tail[1] *=  1.2;}
            if (ACC_TAIL[1] < 20){tuning_tail[1] *=  0.8;}

            ACC_TAIL[0] = 0;
            ACC_TAIL[1] = 0;
        }

    }

///**** Part II: burn in is over ******/
    /*reset acceptance values*/
    for(m = 0; m < *M; m++){
            for(p = 0; p < *P; p++){
                ACC_THETA[m + *M * p] = 0;
                ATT_THETA[m + *M * p] = 0;
            }
        }
    ACC_TAIL[0] = 0;
    ACC_TAIL[1] = 0;
//
/////**************start final burn******************/
    if(*verbose == 1){Rprintf("Burn-in Finished. \n");}

    for(i = 0; i< *iters; i++){
        if(*verbose == 1){
            if((i) % 1000 == 0 ){
                Rprintf("Keepers %d %% done.", 100 * i /  *iters );
                Rprintf("\n");
            }
        }
        for(p = 0; p < *P; p++){
/*update constant basis functions*/
            nsamps_int = rgeom(*cbf_eps);
            while(nsamps_int > 0){
                (ATT_THETA[*M * p]) ++ ;
                canthetastar[*M * p] = tuning_parms[*M * p] * rnorm(0,1) + thetastar[*M * p];
                cantheta[*M * p] = canthetastar[*M * p];
                /*candidate loglikelihood*/
                ll(M, P, N, n1, n2, n3, n4, n5,
                    X, y, y_low, y_high,
                    M_knots_length, I_knots_length, spline_df, M_knots, I_knots,
                    M_low, M_high, I_low, I_high,
                    q_low, q_high, xi_zero,
                    cantheta, xi_low, xi_high,
                    tau, tau_low, tau_high, log_like, &can_ll_sum,
                    bin, bin_low, bin_high,
                    ld_low_ptr,
                    ld_high_ptr,
                    tau_low_ptr,
                    tau_high_ptr,
                    IKM, MKM, M_1, reset_value);
                /*candidate prior*/
                can_lp = -1 * cbfprec * canthetastar[*M * p] * canthetastar[*M * p];
                if (rexp(1) > (lp_cbf[p] + ll_sum - can_lp - can_ll_sum)){
                    theta[*M * p] = cantheta[*M * p];
                    thetastar[*M * p] = canthetastar[*M * p];
                    ll_sum = can_ll_sum;
                    lp_cbf[p] = can_lp;
                    (ACC_THETA[*M * p]) ++;
                }
                /*restore cantheta to theta*/
                else{
                    cantheta[*M * p] = theta[*M * p];
                    canthetastar[*M * p] = thetastar[*M * p];
                }
                nsamps_int -- ;
            }
            THETA[i][*M * p] = theta[*M * p];
        }
        /*update predictors that have effects that change with quantile level*/
        for(p = 0; p < *P1; p++){
            /******update hyperparameters*****/
            /*Gibbs sample mu*/
            sum_theta_IP(&rho[p], M, P, &p, thetastar, &sumtheta[p], &sumprec[p]);
            mu[p] = pow((sigma2[p] * sumprec[p] + muprec),-0.5) * rnorm(0,1) + (sigma2[p] * sumtheta[p])/(sigma2[p] * sumprec[p] + muprec);
            /*update the quadratic term for rho*/
            rho_quad_IP(&rho[p], M, P, &p, thetastar, mu, &rhoquad[p]);
            /*Gibbs sample precision sigma2 */
            sigma2[p] = rgamma(*M_1 / 2 + *sig_a,(2 * *sig_b)/(*sig_b * rhoquad[p] + 2));
            /*Independent Metropolis sample rho*/
            log_det_AR(&rho[p], M_1, &logdet);
            rho_ll = 0.5 * (logdet - sigma2[p] * rhoquad[p]);
            canrho = runif(0,1);
            log_det_AR(&canrho, M_1, &canlogdet);
            rho_quad_IP(&canrho, M, P, &p, thetastar, mu, &canrhoquad);
            canrho_ll  = 0.5 * (canlogdet - sigma2[p] * canrhoquad);
            if (rexp(1) > (rho_ll - canrho_ll)){
                rho[p] = canrho;
                rhoquad[p] = canrhoquad;
            }
            /*	update prior for theta*/
            lp [p] = -0.5 * sigma2[p] * rhoquad [p];
            MU[i][p] = mu[p];
            SIGMA2[i][p] = sigma2[p];
            RHO[i][p] = rho[p];

            /*update theta*/
            for(m = 1; m < *M; m++){
                nsamps_int = rgeom(*theta_eps);
                while(nsamps_int > 0){
                    (ATT_THETA[m + *M * p]) ++;
                    /*draw a candidate*/
                    canthetastar[m + *M * p] = tuning_parms[m + *M * p] * rnorm(0,1) + thetastar[m + *M * p];
                    threshold(M, P, &m, canthetastar, cantheta, &slope_update);
                    /*candidate loglikelihood*/
                    ll(M, P, N, n1, n2, n3, n4, n5,
                    X, y, y_low, y_high,
                    M_knots_length, I_knots_length, spline_df, M_knots, I_knots,
                    M_low, M_high, I_low, I_high,
                    q_low, q_high, xi_zero,
                    cantheta, xi_low, xi_high,
                    tau, tau_low, tau_high, log_like, &can_ll_sum,
                    bin, bin_low, bin_high,
                    ld_low_ptr,
                    ld_high_ptr,
                    tau_low_ptr,
                    tau_high_ptr,
                    IKM, MKM, M_1, reset_value);

                    rho_quad_IP(&rho[p], M, P, &p, canthetastar, mu, &canrhoquad);
                    can_lp = -0.5 * sigma2[p] * canrhoquad;  /*update the log prior for theta*/
                    if (rexp(1) > (lp[p] + ll_sum - can_lp - can_ll_sum)){
                        thetastar[m + *M * p] = canthetastar[m + *M * p];
                        theta[m + *M * p] = cantheta[m + *M * p];
                        if(slope_update == 1){
                            for(j = 0; j < *P; j++){
                                theta[m + *M * j] = cantheta[m + *M * j] ;
                            }
                        }
                        ll_sum = can_ll_sum;
                        lp[p] = can_lp;
                        rhoquad[p] = canrhoquad;
                        (ACC_THETA[m + *M * p]) ++;
                    }
                    /*restore cantheta to theta*/
                    else{
                        canthetastar[m + *M * p] = thetastar[m + *M * p];
                        for(j = 0; j < *P; j++){
                            cantheta[m + *M * j] = theta[m + *M * j];
                        }
                    }
                    nsamps_int -- ;
                }
                THETA[i][m + *M * p] = theta[m + *M * p];
            }
        }
            /******update the tails******/
        Z_vec[0] = rnorm(0,1);
        Z_vec[1] = rnorm(0,1);

        E_vec[0] = rexp(1);
        E_vec[1] = rexp(1);

        tail_ptr(
        M, P,
        N, n1, n2, n3, n4, n5,
        X, y, y_low, y_high,
        M_knots_length, I_knots_length, spline_df, M_knots, I_knots,
        M_low, M_high, I_low, I_high,
        q_low, q_high, xi_zero,
        theta, xi_low, xi_high,
        tau, tau_low, tau_high, log_like, &ll_sum,
        bin, bin_low, bin_high,
        ld_low_ptr,
        ld_high_ptr,
        tau_low_ptr,
        tau_high_ptr,
        IKM, MKM, M_1, reset_value,
        tuning_tail,
        tail_mean,
        &tail_prec,
        E_vec,
        Z_vec,
        ACC_TAIL);

        ll(M, P, N, n1, n2, n3, n4, n5,
            X, y, y_low, y_high,
            M_knots_length, I_knots_length, spline_df, M_knots, I_knots,
            M_low, M_high, I_low, I_high,
            q_low, q_high, xi_zero,
            theta, xi_low, xi_high,
            tau, tau_low, tau_high, log_like, &ll_sum,
            bin, bin_low, bin_high,
            ld_low_ptr,
            ld_high_ptr,
            tau_low_ptr,
            tau_high_ptr,
            IKM, MKM, M_1, reset_value);

        XI_LOW[i] = *xi_low;
        XI_HIGH[i] = *xi_high;

        for(k = 0; k < *N; k++){
            CPO[k] += exp(-log_like[k]);
        }
    }
    for(k = 0; k < *N; k++){
        CPO[k] = *iters / CPO[k];
        *LPML += log(CPO[k]);
    }
    PutRNGstate();

    Free(tau);
    Free(tau_low);
    Free(tau_high);
    Free(theta);
    Free(cantheta);
    Free(canthetastar);
    Free(bin);
    Free(bin_low);
    Free(bin_high);
    Free(log_like);
    Free(reset_value);
    Free(CPO);
}

