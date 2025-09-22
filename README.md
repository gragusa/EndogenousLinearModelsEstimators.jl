# EndogenousLinearModelsEstimators.jl

[![CI](https://github.com/gragusa/EndogenousLinearModelsEstimators.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/gragusa/EndogenousLinearModelsEstimators.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/gragusa/EndogenousLinearModelsEstimators.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/gragusa/EndogenousLinearModelsEstimators.jl)

A Julia package for estimating endogenous linear models using instrumental variables. Provides robust implementations of LIML (Limited Information Maximum Likelihood), Fuller bias-corrected, and 2SLS (Two-Stage Least Squares) estimators with unified API and cross-platform validation.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/gragusa/EndogenousLinearModelsEstimators.jl")
```

## Quick Start

This example replicates the classic Card (1995) study on returns to education using college proximity as an instrument.

```julia
using EndogenousLinearModelsEstimators
using CSV
using DataFrames
using Downloads
cardcsv = Downloads.download("https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/refs/heads/master/csv/wooldridge/card.csv")


# Load the Card dataset
card_data = CSV.read(cardcsv, DataFrame)

# Prepare the data following R ivmodel example
Y = card_data.lwage                     # Log wages (outcome)
D = card_data.educ[:,:]                 # Education (endogenous regressor)
Z = [card_data.nearc4 card_data.nearc2] # Near 4-year college (instrument)

# Exogenous control variables (same as in R ivmodel example)
Xnames = ["exper", "expersq", "black", "south",
          "smsa", "reg661", "reg662", "reg663",
          "reg664", "reg665", "reg666", "reg667",
          "reg668", "smsa66"]
X = Matrix(card_data[:, Xnames])
```

### LIML Results:

```julia
result_liml = liml(Y, D, Z, X; vcov=:HC0)
result_liml
```

Fuller Results (α=1):

```julia
result_fuller = fuller(Y, D, Z, X; a=1.0, vcov=:HC0)
result_fuller
```

### 2SLS Estimation

```julia
result_2sls = tsls(Y, D, Z, X; vcov=:HC0)
result_2sls
```

## API Reference

### Estimators

All estimators share the same unified interface:

```julia
liml(y, x, Z, W; vcov=:HC0, weights=nothing, add_intercept=true)
fuller(y, x, Z, W; a=1.0, vcov=:HC0, weights=nothing, add_intercept=true)
tsls(y, x, Z, W; vcov=:HC0, weights=nothing, add_intercept=true)
```

**Arguments:**

- `y`: Dependent variable (n×1)
- `x`: Endogenous regressor(s) (n×k)
- `Z`: Instruments (n×L), must have L ≥ k
- `W`: Exogenous control variables (n×p), optional
- `vcov`: Variance estimator (`:homoskedastic`, `:HC0`, `:HC1`)
- `weights`: Observation weights (not implemented yet)
- `add_intercept`: Whether to include intercept (default: true)
- `a`: Fuller bias correction parameter (Fuller only, default: 1.0)

### Results Structure

All estimators return `EndogenousLinearModelsEstimationResults`:

```julia
result.beta          # Coefficient estimates
result.vcov          # Variance-covariance matrix
result.stderr        # Standard errors
result.residuals     # Model residuals
result.df            # Degrees of freedom
result.estimator     # "LIML", "Fuller", or "2SLS"
result.kappa         # Kappa value (LIML/Fuller only)
result.vcov_type     # Variance estimator used
result.n             # Sample size
```

**Accessor methods:**

```julia
coef(result)         # coefficients
stderror(result)       # standard errors
vcov(result)         # variance-covariance matrix
residuals(result)    # residuals
dof(result)          # degrees of freedom
```

### Variance Estimators

- **`:homoskedastic`**: Traditional homoskedastic standard errors
- **`:HC0`**: White heteroskedastic-consistent (default, matches R's `heteroSE=FALSE`)
- **`:HC1`**: HC0 with degrees of freedom adjustment

## References

- Card, D. (1995). "Using Geographic Variation in College Proximity to Estimate the Return to Schooling." _Aspects of Labour Market Behaviour: Essays in Honour of John Vanderkamp_. University of Toronto Press.
- Fuller, W. A. (1977). "Some Properties of a Modification of the Limited Information Estimator." _Econometrica_, 45(4), 939-953.
- Wang, J. and Zivot, E. (1998). "Inference on Structural Parameters in Instrumental Variables Regression with Weak Instruments." _Econometrica_, 66(6), 1389-1404.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
