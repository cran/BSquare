\name{qreg}
\alias{qreg}
\title{
Quantile regression with parametric basis functions.)
}
\description{
MCMC code for the quantile regression model of Reich and Smith, 2013.
}
\usage{
qreg(X,Y=NULL,Y_low=NULL,Y_high=NULL,status=NULL,
          L=4,base="Gaussian",varying_effect=NULL,
          tau=seq(0.05,0.95,0.05),
          burn=10000,iters=50000)
          
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
  \item{L}{The number of basis functions in quantile function}
  \item{base}{The centering distribution which can take values "Gaussian", "t", "logistic", "gamma", "weibull", or "ALAP."}
  \item{varying_effect}{If varying_effect = j, then only the covariates in the first j columns of X have different effects on different quantile levels.}
  \item{tau}{Vector of quantile levels for output.}
  \item{burn}{Number of MCMC samples to discard as burn-in.}
  \item{iters}{Number of MCMC samples to generate after the burn-in.}
}
\details{
  See http://www4.stat.ncsu.edu/~reich/QR/ for more detailed descriptions and examples.
}
\note{
The example is used to illustrate the method.  
In practice MCMC chains should be longer.
}
\value{
  \item{q}{Posterior samples of the quantile function.}
  \item{LPML}{Log pseudo-maximum likelihood statistic for model comparisons.}
}
\references{
  Reich BJ, Smith LB (2013).  Bayesian quantile regression for censored data. In press, Biometrics.\cr\cr
  Smith LB, Fuentes M, Herring AH, Reich BJ (2013) Bayesian dependent quantile regression processes for birth outcomes. Submitted.\cr\cr
  Reich BJ (2012) Spatiotemporal quantile regression for detecting distributional changes in environmental processes. JRSS-C, 64, 535-553.\cr\cr
  Reich BJ, Fuentes M, Dunson DB (2011) Bayesian spatial quantile regression. JASA, 106, 620.
}

\seealso{
  \code{\link{dqreg}}
  \code{\link{qr_plot}}
}
\examples{
  #Continuous data example
  #Load the air quality data
  data(airquality)
  ozone<-airquality[,1]
  solar<-airquality[,2]
  
  #Remove missing observations
  missing<-is.na(ozone+solar)
  ozone<-ozone[!missing]
  solar<-solar[!missing]
  solar_std<-1.8*(solar - min(solar))/(max(solar)-min(solar)) - 0.9
  
  #Fit the model and plot results
  X<-cbind(1,solar_std)
  #use longer chains in practice
  fit<-qreg(X,ozone,L=4,base="gamma", iters = 1000, burn = 1000)
  qr_plot(fit,2, main = "Solar Effect")

  
  #Right-censored data example
  
  library(survival)
  data(veteran)
  
  trt<-ifelse(veteran[,1]==2,-1,1)
  logtime<-log(veteran[,3])
  event<-veteran[,4]
  status<-ifelse(event==1,0,2)
  X<-cbind(1,trt)
  #use longer chains in practice
  fit<-qreg(X,Y=logtime,status=status,iters =1000, burn = 1000)
  qr_plot(fit,index=2,main="Treatment effect")
}
