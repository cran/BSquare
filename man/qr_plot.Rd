\name{qr_plot}
\alias{qr_plot}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Plot
}
\description{
Plot of posterior distribution of effects
}
\usage{
qr_plot(object, index = 1, 
xlab = "Quantile level", ylab = "Covariate effect", 
main = "", col = gray(0.75), lwd = 1, add = FALSE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{object}{An object of class qreg or qreg_spline}
\item{index}{Integer describing which predictor to plot}
\item{xlab}{X axis label}
\item{ylab}{Y axis label}
\item{main}{plot title}
\item{col}{plot color}
\item{lwd}{line width}
\item{add}{Superimposed plot}
}
