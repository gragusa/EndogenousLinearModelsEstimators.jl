"""
# Estimators

Unified implementations of LIML, Fuller, and 2SLS estimators with consistent API.
"""

"""

    liml(y, X, Z, W; vcov=:HC0, weights=nothing, add_intercept=true)
    liml(y, X, Z; vcov=:HC0, weights=nothing, add_intercept=true)

Limited Information Maximum Likelihood (LIML) estimator for endogenous linear models.

## Arguments
- `y::AbstractVector`: Dependent variable (n×1)
- `X::AbstractVecOrMat`: Endogenous regressor(s) to be instrumented (n×k)
- `Z::AbstractMatrix`: Instruments (n×L), must have L ≥ k
- `W::Union{AbstractMatrix,Nothing}`: Exogenous control variables (n×p), optional

## Keyword Arguments
- `vcov::Symbol=:HC0`: Variance estimator (:homoskedastic, :HC0, :HC1)
- `weights::Union{Nothing,AbstractVector}=nothing`: Observation weights (not implemented)
- `add_intercept::Bool=true`: Whether to include intercept in the model

## Returns
`EndogenousLinearModelsEstimationResults{T}` with LIML estimates and diagnostics, where T is inferred from input types.

## Notes
Uses corrected degrees of freedom: df = n - L - p where L = number of instruments,
p = number of exogenous variables (including intercept if add_intercept=true).

## New API (recommended)
```julia
# With exogenous variables

liml(y, X, Z, W; vcov=:HC0)

# Without exogenous variables
liml(y, X, Z; vcov=:HC0)
```
"""
function liml(
    y::AbstractVector,
    X::AbstractVecOrMat,
    Z::AbstractMatrix,
    W::Union{AbstractMatrix,Nothing} = nothing;
    vcov::Symbol = :HC0,
    weights::Union{Nothing,AbstractVector} = nothing,
    add_intercept::Bool = true,
)
    # Input validation
    _check_weights(weights)
    n, x_vec, L, p_exog = _validate_inputs(y, X, Z; X = W)

    # Prepare exogenous matrix
    Wmat = _prep_X(n; X = W, add_intercept = add_intercept)
    p = size(Wmat, 2)  # Total exogenous parameters (including intercept)

    # Corrected degrees of freedom: n - L - p
    df = n - L - p

    # LIML estimation
    κ = _liml_kappa(y, x_vec, Z; X = Wmat)
    θ, u, A, invA, Adj = _kclass_fit(y, x_vec, Z; X = Wmat, k = κ)
    V = _kclass_vcov(invA, u, Adj; vcov_mode = vcov, df = df, n = n)

    # Extract coefficients and compute standard errors
    beta = θ
    stderr_vec = sqrt.(diag(V))

    # Return unified result
    return EndogenousLinearModelsEstimationResults(
        beta,
        V,
        stderr_vec,
        u,
        df,
        "LIML";
        kappa = κ,
        vcov_type = vcov,
        n = n,
        nparams = length(beta),
        ninstruments = L,
        nexogenous = p,
    )
end

# Legacy methods are handled by adding explicit keyword-only methods
# This avoids method ambiguity while maintaining backward compatibility

"""

    fuller(y, X, Z, W; a=1.0, vcov=:HC0, weights=nothing, add_intercept=true)
    fuller(y, X, Z; a=1.0, vcov=:HC0, weights=nothing, add_intercept=true)

Fuller bias-corrected estimator for endogenous linear models.

## Arguments
- `y::AbstractVector`: Dependent variable (n×1)
- `X::AbstractVecOrMat`: Endogenous regressor(s) to be instrumented (n×k)
- `Z::AbstractMatrix`: Instruments (n×L), must have L ≥ k
- `W::Union{AbstractMatrix,Nothing}`: Exogenous control variables (n×p), optional

## Keyword Arguments
- `a::Real=1.0`: Fuller bias correction parameter (α)
- `vcov::Symbol=:HC0`: Variance estimator (:homoskedastic, :HC0, :HC1)
- `weights::Union{Nothing,AbstractVector}=nothing`: Observation weights (not implemented)
- `add_intercept::Bool=true`: Whether to include intercept in the model

## Returns
`EndogenousLinearModelsEstimationResults{T}` with Fuller estimates and diagnostics, where T is inferred from input types.

## Notes
Uses Fuller formula: κ_Fuller = κ_LIML - a/(n - L - p).
When a=0, reduces to LIML estimator.

## New API (recommended)
```julia
# With exogenous variables

fuller(y, X, Z, W; a=1.0, vcov=:HC0)

# Without exogenous variables
fuller(y, X, Z; a=1.0, vcov=:HC0)
```
"""
function fuller(
    y::AbstractVector,
    X::AbstractVecOrMat,
    Z::AbstractMatrix,
    W::Union{AbstractMatrix,Nothing} = nothing;
    a::Real = 1.0,
    vcov::Symbol = :HC0,
    weights::Union{Nothing,AbstractVector} = nothing,
    add_intercept::Bool = true,
)
    # Input validation
    _check_weights(weights)
    n, x_vec, L, p_exog = _validate_inputs(y, X, Z; X = W)

    # Prepare exogenous matrix
    Wmat = _prep_X(n; X = W, add_intercept = add_intercept)
    p = size(Wmat, 2)  # Total exogenous parameters (including intercept)

    # Corrected degrees of freedom: n - L - p
    df = n - L - p

    # First compute LIML kappa
    κ_liml = _liml_kappa(y, x_vec, Z; X = Wmat)

    # Fuller correction: κ_Fuller = κ_LIML - a/df
    κ_fuller = κ_liml - a / df

    # Fuller estimation
    θ, u, A, invA, Adj = _kclass_fit(y, x_vec, Z; X = Wmat, k = κ_fuller)
    V = _kclass_vcov(invA, u, Adj; vcov_mode = vcov, df = df, n = n)

    # Extract coefficients and compute standard errors
    beta = θ
    stderr_vec = sqrt.(diag(V))

    # Return unified result
    return EndogenousLinearModelsEstimationResults(
        beta,
        V,
        stderr_vec,
        u,
        df,
        "Fuller";
        kappa = κ_fuller,
        vcov_type = vcov,
        n = n,
        nparams = length(beta),
        ninstruments = L,
        nexogenous = p,
    )
end

# Legacy Fuller methods handled similarly

"""

    tsls(y, X, Z, W; vcov=:HC0, weights=nothing, add_intercept=true)
    tsls(y, X, Z; vcov=:HC0, weights=nothing, add_intercept=true)

Two-Stage Least Squares (2SLS) estimator for endogenous linear models.

## Arguments
- `y::AbstractVector`: Dependent variable (n×1)
- `X::AbstractVecOrMat`: Endogenous regressor(s) to be instrumented (n×k)
- `Z::AbstractMatrix`: Instruments (n×L), must have L ≥ k
- `W::Union{AbstractMatrix,Nothing}`: Exogenous control variables (n×p), optional

## Keyword Arguments
- `vcov::Symbol=:HC0`: Variance estimator (:homoskedastic, :HC0, :HC1)
- `weights::Union{Nothing,AbstractVector}=nothing`: Observation weights (not implemented)
- `add_intercept::Bool=true`: Whether to include intercept in the model

## Returns
`EndogenousLinearModelsEstimationResults{T}` with 2SLS estimates and diagnostics, where T is inferred from input types.

## Notes
Implements 2SLS as: β̂ = (X'P_Z X)^(-1) X'P_Z y where P_Z = Z(Z'Z)^(-1)Z'.
For multiple endogenous regressors, X includes all regressors (endogenous + exogenous).

## New API (recommended)
```julia
# With exogenous variables

tsls(y, X, Z, W; vcov=:HC0)

# Without exogenous variables
tsls(y, X, Z; vcov=:HC0)
```
"""
function tsls(
    y::AbstractVector,
    X::AbstractVecOrMat,
    Z::AbstractMatrix,
    W::Union{AbstractMatrix,Nothing} = nothing;
    vcov::Symbol = :HC0,
    weights::Union{Nothing,AbstractVector} = nothing,
    add_intercept::Bool = true,
)
    # Input validation
    _check_weights(weights)
    n, x_vec, L, p_exog = _validate_inputs(y, X, Z; X = W)

    # Prepare exogenous matrix
    Wmat = _prep_X(n; X = W, add_intercept = add_intercept)

    # For 2SLS, combine endogenous and exogenous regressors
    if size(Wmat, 2) == 0
        # No exogenous variables, just the endogenous regressor
        RegMat = reshape(x_vec, :, 1)  # Ensure matrix form
        k = 1
    else
        # Include both endogenous and exogenous regressors
        RegMat = hcat(reshape(x_vec, :, 1), Wmat)
        k = size(RegMat, 2)
    end

    # Instruments must include exogenous variables for identification
    if size(Wmat, 2) > 0
        Z_full = hcat(Z, Wmat)  # Instruments include exogenous variables
    else
        Z_full = Z
    end

    L_full = size(Z_full, 2)

    # Check identification: need at least as many instruments as regressors
    if L_full < k
        throw(
            ArgumentError(
                "Model is underidentified: need at least $k instruments, got $L_full",
            ),
        )
    end

    # 2SLS estimation using efficient implementation (avoid forming projection matrix)
    ZtZ = Symmetric(Z_full'Z_full)
    ZtRegMat = Z_full'RegMat
    Zty = Z_full'y

    # First stage: solve (Z'Z)^(-1) * (Z'RegMat)
    RegMatZ = ZtZ \ ZtRegMat
    # Second stage coefficients: (RegMat'P_Z RegMat)^(-1) * (RegMat'P_Z y)
    RegMatPzRegMat = ZtRegMat' * RegMatZ
    RegMatPzy = ZtRegMat' * (ZtZ \ Zty)

    β = RegMatPzRegMat \ RegMatPzy

    # Second-stage residuals
    u = y - RegMat * β

    # Degrees of freedom for 2SLS
    df = n - k

    # Variance-covariance matrix
    if vcov == :homoskedastic
        σ2 = dot(u, u) / df
        V = σ2 * Symmetric(inv(RegMatPzRegMat))
    elseif vcov == :HC0 || vcov == :HC1
        # Robust variance: (RegMat'P_Z RegMat)^(-1) * (RegMat' P_Z Ω P_Z RegMat) * (RegMat'P_Z RegMat)^(-1)
        # where Ω = diag(u_i^2) and P_Z = Z(Z'Z)^(-1)Z'
        # Efficient computation without forming full P_Z matrix
        H = Z_full' * (Z_full .* (u .^ 2))  # Z' * diag(u^2) * Z
        RegMat_H = ZtZ \ H
        M = ZtRegMat' * (RegMat_H * RegMatZ)
        V_temp = inv(RegMatPzRegMat) * M * inv(RegMatPzRegMat)
        if vcov == :HC1
            V_temp .*= n / df
        end
        V = Symmetric(V_temp)
    else
        error("vcov must be :homoskedastic, :HC0, or :HC1")
    end

    stderr_vec = sqrt.(diag(V))

    # For consistency with LIML/Fuller, extract only the endogenous coefficient
    # and adjust for unified return format
    if size(Wmat, 2) > 0
        # We have exogenous variables, so β[1] is endogenous, β[2:end] are exogenous
        beta_unified = β  # Keep all coefficients
        vcov_unified = V
        stderr_unified = stderr_vec
    else
        # Only endogenous variable
        beta_unified = β
        vcov_unified = V
        stderr_unified = stderr_vec
    end

    return EndogenousLinearModelsEstimationResults(
        beta_unified,
        vcov_unified,
        stderr_unified,
        u,
        df,
        "2SLS";
        kappa = nothing,  # 2SLS doesn't have kappa
        vcov_type = vcov,
        n = n,
        nparams = length(beta_unified),
        ninstruments = L,  # Original instrument count
        nexogenous = size(Wmat, 2),
    )
end