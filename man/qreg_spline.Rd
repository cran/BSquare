\name{qreg_spline}
\alias{qreg_spline}
\title{
Quantile regression with splines.
}
\description{
Quantile regression using spline basis functions for the quantile process.
}
\usage{
qreg_spline(X, Y = NULL, Y_low = NULL, Y_high = NULL, 
status = NULL, knots_inter = c(0.1, 0.5, 0.9), 
Pareto = TRUE, 
varying_effect = NULL,  
tau = seq(0.05, 0.95, 0.05), 
burn = 10000, iters = 50000,  
q_low = 0.01, q_high = 0.99, 
sig_a = .1, sig_b = .1, 
mu_var = 10^2, cbf_var = 10^3, 
tail_mean = -1.2, tail_var = .4,
cbf_eps = 0.5, theta_eps = 0.5, 
tuning_theta = 1, tuning_tail = rep(1, 2), 
cred_low = 0.025, cred_high = 0.975, 
seed = 1, verbose = TRUE)
}
\arguments{
  \item{X}{Matrix of predictors with the first column consisting of all ones and all other values between -1 and 1.}
  \item{Y}{A vector of responses.}
  \item{Y_low,Y_high}{Vectors of endpoints for interval-censored values.}
  \item{status}{Censoring status taking values \cr 
                0 if uncensored\cr
                1 if left-censored on the interval (-Inf,Y)\cr
                2 if right-censored on the interval (Y,Inf)\cr
                3 if censored on the interval (Y_low,Y_high).}
  \item{knots_inter}{
  An ordered sequence of numbers in (0,1) specifying knot locations.
}
\item{Pareto}{
  An indicator for whether Pareto or exponential tails are fit, corresponding to heavy and light tailed densities respectively.  
  Distributions that decay slowly in the tail (e.g. t-distribution) are better   
}
\item{varying_effect}{If varying_effect = j, then only the covariates in the first j columns of X have different effects on different quantile levels.} 
\item{tau}{Vector of quantile levels for output.}
\item{burn}{Number of MCMC samples to discard as burn-in.}
\item{iters}{Number of MCMC samples to generate after the burn-in.}
\item{q_low}{The quantile level below which the lower parametric tail is fit.}
\item{q_high}{The quantile level above which the upper parametric tail is fit.}
\item{sig_a}{The shape hyperparameter for the prior precision of the basis functions.}
\item{sig_b}{The scale hyperparameter for the prior precision of the basis functions.}
\item{mu_var}{The prior variance of the prior mean of the basis functions.}
\item{cbf_var}{The prior variance of the constant basis functions.}
\item{tail_mean}{The prior mean for the log of the shape parameters for Pareto tails. 
  Only used if Pareto = TRUE.
}
\item{tail_var}{The prior variance for the log of the shape parameters for Pareto tails. 
  Only used if Pareto = TRUE.
}
\item{cbf_eps}{
A parameter in [.1,.5] indicating the degree of stochastic thinning for the location basis functions.
Lower is stronger.
}
\item{theta_eps}{A parameter in [.1,.5] indicating the degree of stochastic thinning for the other basis functions.
Lower is stronger.}
\item{tuning_theta}{The initial candidate variance for the parameters updated by random walk Metropolis Hastings. }
\item{tuning_tail}{The initial candidate variance for the tail parameters.}
\item{cred_low}{The lower limit of the posterior credible intervals.}
\item{cred_high}{The upper limit of the posterior credible intervals.}
\item{seed}{MCMC seed.}
\item{verbose}{An indicator for outputting real-time MCMC updates. }
}
\value{
\item{q}{A (iters x N_tau x P) array of posterior quantile effects}
\item{q_lower}{An (N_tau x P) array of lower limits of credible sets for posterior quantile effects}
\item{q_upper}{An (N_tau x P) array of upper limit of credible sets for posterior quantile effects}
\item{q_mean}{An (N_tau x P) array of posterior means for posterior quantile effects}
\item{theta}{An (iters x M x P) array of posterior parameters}
\item{tuning_parms}{An (M x P) array of candidate precisions}
\item{acc_theta}{An (M x P) array of acceptance ratios of theta for the keepers}
\item{post_mu}{An (N_tau x M x P) array of posterior mean hyperparameters}
\item{post_sigma2}{An (N_tau x M x P) array of posterior precision hyperparameters}
\item{post_rho}{An (N_tau x M x P) array of posterior correlation hyperparameters}
\item{post_xi_low}{An (iters x 1) array of posterior lower tail shape parameters}
\item{post_xi_high}{An (iters x 1) array of posterior upper tail shape parameters}
\item{tau}{A vector specifying the quantile levels of interest (includes the thresholds)}
\item{MCMC_time}{The MCMC run time.}
\item{LPML}{The log pseudo marginal likelihood.}
\item{iters}{The number of MCMC iterations kept.}
\item{burn}{The number of MCMC iterations burned.}

}
\references{
  Smith LB, Fuentes M, Herring AH, Reich BJ (2013) Bayesian dependent quantile regression processes for birth outcomes. Submitted.\cr\cr
  Reich BJ, Fuentes M, Dunson DB (2011) Bayesian spatial quantile regression. JASA, 106, 6-20.
}
\details{
  See http://www4.stat.ncsu.edu/~reich/QR/ for more detailed descriptions and examples. 
}

\note{
The example is used to illustrate the method.  
In practice MCMC chains should be longer.
Code for modeling multiple quantile functions correlated across space or time or for running on graphics processing units is available from the authors on request.
}
\seealso{
  \code{\link{qr_plot}}
}
\examples{
data(airquality)
ozone=airquality[,1]
solar=airquality[,2]

#Remove missing observations
missing=is.na(ozone) | is.na(solar)
ozone=ozone[!missing]
solar=solar[!missing]

#Create design matrix.  First column must be all ones, others must be between -1 and 1
solar_std = 1.8 * (solar - min(solar))/(max(solar)-min(solar)) - 0.9
X = cbind(1,solar_std)

tau=seq(0.05,0.95,0.05)
#use longer chains in practice
fit<-qreg_spline(X,Y = ozone,iters = 1000, burn = 1000, knots_inter = c(.5))

qr_plot(fit,index=2, main = "Solar Effect")
}
