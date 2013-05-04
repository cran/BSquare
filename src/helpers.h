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


void MM (int *T_A, int *T_B, int *M, int *N, int *K, double *alpha, double *A, double *B, double *beta, double *C);

void dot_product(int *D_vector, double *v1, double *v2, double *dp);

void make_prec(double *rho, int *G, double *OMEGA);

void chol2inv(int *G, double *OMEGA, double *OMEGA1);

void symm_log_det(int*G, double *OMEGA, double *logdet);

void sum_theta_IP(double *rho, int *M, int *P, int *p, double *theta, double *sumtheta, double *sumprec);

void rho_quad_IP(double *rho, int *M, int *P, int *p, double *theta, double *mu, double *rhoquad);

void rho_quad(double *rho, int *M, int *G, int *P, int *m, int *p, double *theta, double *mu, double *rhoquad);

void sum_theta(double *rho, int *M, int *G, int *P, int *m, int *p, double *theta, double *sumtheta, double *sumprec);

void sum_theta_constant(double *rho, int *M, int *G, int *P, int *g, int *p, double *theta, double *sumtheta, double *sumprec);

void rho_quad_constant(double *rho, int *M, int *G, int *P, int *g, int *p, double *theta, double *mu, double *rhoquad);

void log_det_AR(double *rho, int *G, double *logdet);

void find_int(int *n, double *xt, double *tau, int *bin);

double ispline3(double tau, int spline_df, int  m, double  I_knots[], int bin, double IKM[], int M_1);

double mspline3(double tau, int spline_df, int  m, double  I_knots[], int bin, double MKM[], int M_1);

double Q_3 (int M, double tau, double w[], int spline_df, double I_knots[], int bin, double IKM[], int M_1);

double q_3 (int M, double tau, double w[], int  spline_df, double  I_knots[], int bin, double MKM[], int M_1);

void rootfind_GPU (int *M_knots_length, int *I_knots_length, int *M, int *spline_df, double *tau_scalar, double *w,
               double *y_scalar,  double  *M_knots, double  *I_knots, int *bin, double *q_low, double *q_high,
               double *IKM, double *MKM, int *M_1, double *reset_value);

void exponential_low(double *sigma, double *xi, double *v, double *q_level, double *y_scalar);

void exponential_high(double *sigma, double *xi, double *v, double *q_level, double *y_scalar);

void pareto_low(double *sigma, double *xi, double *v, double *q_level, double *y_scalar);

void pareto_high(double *sigma, double *xi, double *v, double *q_level, double *y_scalar);

void exponential_tau_low(double *sigma, double *xi, double *v, double *q_level, double *tau_scalar);

void exponential_tau_high(double *sigma, double *xi, double *v, double *q_level, double *tau_scalar);

void pareto_tau_low(double *sigma, double *xi, double *v, double *q_level, double *tau_scalar);

void pareto_tau_high(double *sigma, double *xi, double *v, double *q_level, double *tau_scalar);

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
        );

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
        );

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
        int *ACC_TAIL);

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
        int *ACC_TAIL);

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
        int *ACC_TAIL);

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
        int *ACC_TAIL);

void threshold_CP(int *M, int *G, int *P, int *m, int *g, double *thetastar, double *theta, int *slope_update);

void gamma_sample(int *N, double *shape, double *scale, double *sample);

void make_betahat(int *G, double *GG, int *l, double *theta, double *betahat);

void make_betahat_AR(int *G, double *GG, int *D, int *M, int *m, int *p, double *theta, double *rho, double *V, double *chol_V,
                     double *chol_V_inv, double *V_inv, double *betahat);

void rho_quad_vector(double *rho, int *M, int *G, int *P, int *m, int *p, double *theta, double *mu, double *rhoquad);

void beta_ll(int *G, int *l, double *beta, double *sigma2, double *GG, double *theta, double *beta_ll_sum);

void threshold(int *M, int *P, int *m, double *thetastar, double *theta, int *slope_update);

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
