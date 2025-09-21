"""
# Results

Defines the unified result structure for all endogenous linear model estimators.
"""

using Printf

"""
    EndogenousLinearModelsEstimationResults

Unified result structure for all endogenous linear model estimators (LIML, Fuller, 2SLS).

## Fields

- `beta::Vector{Float64}` - Coefficient estimates
- `vcov::Matrix{Float64}` - Variance-covariance matrix
- `stderr::Vector{Float64}` - Standard errors
- `residuals::Vector{Float64}` - Model residuals
- `df::Int` - Degrees of freedom
- `estimator::String` - Estimator type ("LIML", "Fuller", "2SLS")
- `kappa::Union{Float64,Nothing}` - Kappa value (for LIML/Fuller only)
- `vcov_type::Symbol` - Variance estimator type (:homoskedastic, :HC0, :HC1)
- `n::Int` - Sample size
- `nparams::Int` - Number of estimated parameters
- `ninstruments::Int` - Number of instruments
- `nexogenous::Int` - Number of exogenous variables (including intercept)

## Methods

- `show(io, result)` - Pretty printing
- `coef(result)` - Extract coefficients (alias for `beta`)
- `stderr(result)` - Extract standard errors
- `vcov(result)` - Extract variance-covariance matrix
- `residuals(result)` - Extract residuals
- `dof(result)` - Extract degrees of freedom
"""
struct EndogenousLinearModelsEstimationResults
    beta::Vector{Float64}
    vcov::Matrix{Float64}
    stderr::Vector{Float64}
    residuals::Vector{Float64}
    df::Int
    estimator::String
    kappa::Union{Float64,Nothing}
    vcov_type::Symbol
    n::Int
    nparams::Int
    ninstruments::Int
    nexogenous::Int
end

# Convenience constructors
function EndogenousLinearModelsEstimationResults(
    beta::Vector,
    vcov::AbstractMatrix,
    stderr::Vector,
    residuals::Vector,
    df::Int,
    estimator::String;
    kappa::Union{Float64,Nothing} = nothing,
    vcov_type::Symbol = :HC0,
    n::Int,
    nparams::Int,
    ninstruments::Int,
    nexogenous::Int
)
    return EndogenousLinearModelsEstimationResults(
        Float64.(beta),
        Matrix(Float64.(vcov)),
        Float64.(stderr),
        Float64.(residuals),
        df,
        estimator,
        kappa,
        vcov_type,
        n,
        nparams,
        ninstruments,
        nexogenous
    )
end

# Accessor methods following Julia conventions
"""
    coef(result::EndogenousLinearModelsEstimationResults)

Extract coefficient estimates.
"""
coef(result::EndogenousLinearModelsEstimationResults) = result.beta

"""
    stderr(result::EndogenousLinearModelsEstimationResults)

Extract standard errors.
"""
Base.stderr(result::EndogenousLinearModelsEstimationResults) = result.stderr

"""
    vcov(result::EndogenousLinearModelsEstimationResults)

Extract variance-covariance matrix.
"""
vcov(result::EndogenousLinearModelsEstimationResults) = result.vcov

"""
    residuals(result::EndogenousLinearModelsEstimationResults)

Extract model residuals.
"""
residuals(result::EndogenousLinearModelsEstimationResults) = result.residuals

"""
    dof(result::EndogenousLinearModelsEstimationResults)

Extract degrees of freedom.
"""
dof(result::EndogenousLinearModelsEstimationResults) = result.df

# Pretty printing
function Base.show(io::IO, result::EndogenousLinearModelsEstimationResults)
    println(io, "$(result.estimator) Estimation Results")
    println(io, "=" * "="^(length(result.estimator) + 19))
    println(io, "Sample size:      $(result.n)")
    println(io, "Parameters:       $(result.nparams)")
    println(io, "Instruments:      $(result.ninstruments)")
    println(io, "Exogenous vars:   $(result.nexogenous)")
    println(io, "Degrees of freedom: $(result.df)")
    println(io, "Variance type:    $(result.vcov_type)")

    if result.kappa !== nothing
        println(io, "Kappa:            $(round(result.kappa, digits=6))")
    end

    println(io, "")
    println(io, "Coefficients:")
    println(io, "-" * "-"^20)

    for i in eachindex(result.beta)
        param_name = i == 1 ? "x₁" : "x$i"
        if i > 1 && result.nexogenous > 1
            if i == 2 && result.nexogenous > 1
                param_name = "intercept"
            else
                param_name = "exog$(i-2)"
            end
        end

        @printf(io, "%-12s %10.6f  %10.6f  %8.3f\n",
                param_name, result.beta[i], result.stderr[i],
                result.beta[i] / result.stderr[i])
    end

    println(io, "")
    println(io, "Note: Last column shows t-statistics")
end

function Base.show(io::IO, ::MIME"text/plain", result::EndogenousLinearModelsEstimationResults)
    show(io, result)
end