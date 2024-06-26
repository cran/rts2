% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/print_mcmlrts.R
\name{formula.rtsFit}
\alias{formula.rtsFit}
\title{Extracts the formula from a \code{rtsFit} object.}
\usage{
\method{formula}{rtsFit}(x, ...)
}
\arguments{
\item{x}{A \code{rtsFit} object.}

\item{...}{Further arguments passed from other methods}
}
\value{
A \link[stats]{formula} object.
}
\description{
Extracts the \link[stats]{formula} from a \code{rtsFit} object. Only returns the top level formula. For region models
this is the formula at the region level, otherwise the grid-level formula is returned. No random effects
specifications are included in the returned formula.
}
