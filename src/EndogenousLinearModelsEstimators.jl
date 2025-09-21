"""
# EndogenousLinearModelsEstimators.jl

A Julia package for estimating endogenous linear models using instrumental variables.

Provides robust implementations of:
- LIML (Limited Information Maximum Likelihood) estimator
- Fuller bias-corrected estimator
- 2SLS (Two-Stage Least Squares) estimator

All estimators support:
- Exogenous control variables
- Multiple variance estimator types (homoskedastic, HC0, HC1)
- Unified API and result format
- Cross-platform validation against R and Python implementations

## Example Usage

```julia
using EndogenousLinearModelsEstimators

# Basic usage
result_liml = iv_liml(y, x, Z)
result_fuller = iv_fuller(y, x, Z; a=1.0)
result_2sls = iv_2sls(y, x, Z)

# With exogenous variables
result = iv_liml(y, x, Z; X=exog_vars, vcov=:HC0)

# Results provide unified interface
result.beta          # Coefficient estimates
result.stderr        # Standard errors
result.vcov          # Variance-covariance matrix
result.residuals     # Model residuals
result.df            # Degrees of freedom
```
"""
module EndogenousLinearModelsEstimators

using LinearAlgebra
using Statistics

# Include submodules
include("results.jl")
include("utils.jl")
include("estimators.jl")

# Export main types
export EndogenousLinearModelsEstimationResults

# Export main functions
export iv_liml, iv_fuller, iv_2sls

# Export utility functions for advanced users
export simulate_iv

end # module