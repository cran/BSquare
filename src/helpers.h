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

void sum_alpha(double *rho, int *L, int *P, int *p, double *alpha, double *sumalpha, double *sumprec);

void rho_quad_cens(double *rho, int *L, int *P, int *p, double *alpha, double *mu, double *rhoquad);

void q_norm(double *tau, double *mn, double *scale, double *shape, double *q_tau);

void q_t(double *tau, double *mn, double *scale, double *shape, double *q_tau);

void q_logistic(double *tau, double *mn, double *scale, double *shape, double *q_tau);

void q_alap (double *tau, double *mn ,double *scale, double *shape, double *q_tau);

void q_weibull(double *tau, double *mn ,double *scale, double *shape, double *q_tau);

 void q_gamma(double *tau, double *mn ,double *scale, double *shape, double *q_tau);

void log_p_norm(double *y, double *mn, double *scale, double *shape, double *p_y);

void log_p_t(double *y, double *mn, double *scale, double *shape, double *p_y);

void log_p_logistic(double *y, double *mn, double *scale, double *shape, double *p_y);

void log_p_alap(double *y, double *mn, double *scale, double *shape, double *p_y);

void log_p_weibull(double *y, double *mn ,double *scale, double *shape, double *p_y);

void log_p_gamma(double *y, double *mn ,double *scale, double *shape, double *p_y);

void log_d_norm(double *y, double *mn, double *scale, double *shape, double *d_Y);

void log_d_t(double *y, double *mn, double *scale, double *shape, double *d_Y);

void log_d_alap (double *y, double *mn ,double *scale, double *shape, double *d_Y);

void log_d_logistic (double *y, double *mn ,double *scale, double *shape, double *d_Y);

void log_d_weibull(double *y, double *mn ,double *scale, double *shape, double *d_Y);

void log_d_gamma(double *y, double *mn ,double *scale, double *shape, double *d_Y);

void find_int_2(int *n, int *N, double *xt, double *tau, int *bin);


void make_B_check (int *L, double *kappa,
             int *indicator,
             double *shape, double *B);


void make_B (int *L, double *kappa,
             void q_ptr(double *, double *, double *, double *, double*),
             double *shape, double *B);

void clike (int *N, int *n1, int *n2, int *n3, int *n4, double *kappa, int *P, int *L,
            double *X, double *y, double *y_low, double *y_high,
            double *beta, double *alpha,
            double *shape,
            void q_ptr(double *, double *, double *, double *, double*),
            void log_p_ptr(double *, double *, double *, double *, double*),
            void log_d_ptr(double *, double *, double *, double *, double*),
            double *B,
            int *bin, int *bin_low, int *bin_high, double *ll_sum, double *log_like
            );

void clike_2 (int *N, int *n1, int *n2, double *kappa, int *P, int *L,
            double *X, double *y,
            double *beta, double *alpha,
            double *shape,
            void q_ptr(double *, double *, double *, double *, double*),
            void log_p_ptr(double *, double *, double *, double *, double*),
            void log_d_ptr(double *, double *, double *, double *, double*),
            double *B,
            int *bin, double *ll_sum, double *log_like
            );

void clike_3 (int *N, int *n1, int *n2, double *kappa, int *P, int *L,
            double *X, double *y,
            double *beta, double *alpha,
            double *shape,
            void q_ptr(double *, double *, double *, double *, double*),
            void log_p_ptr(double *, double *, double *, double *, double*),
            void log_d_ptr(double *, double *, double *, double *, double*),
            double *B,
            int *bin, double *ll_sum, double *log_like
            );

void clike_3_wrapper (int *N, int *n1, int *n2, double *kappa, int *P, int *L,
            double *X, double *y,
            double *beta, double *alpha,
            double *shape,
            double *B,
            int *bin, double *ll_sum, double *log_like,
            int *indicator);

void threshold_cens(int *L, int *P, int *l, double *alphastar, double *alpha, int *slope_update);

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
                        );

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
                        );

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
                        );
