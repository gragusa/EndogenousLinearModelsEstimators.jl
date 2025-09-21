"""
# Estimators

Unified implementations of LIML, Fuller, and 2SLS estimators with consistent API.
"""

"""
    iv_liml(y, x, Z; X=nothing, vcov=:HC0, weights=nothing, add_intercept=true)

Limited Information Maximum Likelihood (LIML) estimator for endogenous linear models.

## Arguments
- `y::AbstractVector`: Dependent variable (n×1)
- `x::AbstractVecOrMat`: Endogenous regressor(s) to be instrumented (n×k)
- `Z::AbstractMatrix`: Instruments (n×L), must have L ≥ k

## Keyword Arguments
- `X::Union{Nothing,AbstractMatrix}=nothing`: Exogenous control variables (n×p)
- `vcov::Symbol=:HC0`: Variance estimator (:homoskedastic, :HC0, :HC1)
- `weights::Union{Nothing,AbstractVector}=nothing`: Observation weights (not implemented)
- `add_intercept::Bool=true`: Whether to include intercept in the model

## Returns
`EndogenousLinearModelsEstimationResults` with LIML estimates and diagnostics.

## Notes
Uses corrected degrees of freedom: df = n - L - p where L = number of instruments,
p = number of exogenous variables (including intercept if add_intercept=true).
"""
function iv_liml(
    y::AbstractVector,
    x::AbstractVecOrMat,
    Z::AbstractMatrix;
    X::Union{Nothing,AbstractMatrix} = nothing,
    vcov::Symbol = :HC0,
    weights::Union{Nothing,AbstractVector} = nothing,
    add_intercept::Bool = true,
)
    # Input validation
    _check_weights(weights)
    n, x_vec, L, p_exog = _validate_inputs(y, x, Z; X=X)

    # Prepare exogenous matrix
    Xmat = _prep_X(n; X=X, add_intercept=add_intercept)
    p = size(Xmat, 2)  # Total exogenous parameters (including intercept)

    # Corrected degrees of freedom: n - L - p
    df = n - L - p

    # LIML estimation
    κ = _liml_kappa(y, x_vec, Z; X=Xmat)
    θ, u, A, invA, Adj = _kclass_fit(y, x_vec, Z; X=Xmat, k=κ)
    V = _kclass_vcov(invA, u, Adj; vcov_mode=vcov, df=df, n=n)

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
        nexogenous = p
    )
end

"""
    iv_fuller(y, x, Z; X=nothing, a=1.0, vcov=:HC0, weights=nothing, add_intercept=true)

Fuller bias-corrected estimator for endogenous linear models.

## Arguments
- `y::AbstractVector`: Dependent variable (n×1)
- `x::AbstractVecOrMat`: Endogenous regressor(s) to be instrumented (n×k)
- `Z::AbstractMatrix`: Instruments (n×L), must have L ≥ k

## Keyword Arguments
- `X::Union{Nothing,AbstractMatrix}=nothing`: Exogenous control variables (n×p)
- `a::Real=1.0`: Fuller bias correction parameter (α)
- `vcov::Symbol=:HC0`: Variance estimator (:homoskedastic, :HC0, :HC1)
- `weights::Union{Nothing,AbstractVector}=nothing`: Observation weights (not implemented)
- `add_intercept::Bool=true`: Whether to include intercept in the model

## Returns
`EndogenousLinearModelsEstimationResults` with Fuller estimates and diagnostics.

## Notes
Uses Fuller formula: κ_Fuller = κ_LIML - a/(n - L - p).
When a=0, reduces to LIML estimator.
"""
function iv_fuller(
    y::AbstractVector,
    x::AbstractVecOrMat,
    Z::AbstractMatrix;
    X::Union{Nothing,AbstractMatrix} = nothing,
    a::Real = 1.0,
    vcov::Symbol = :HC0,
    weights::Union{Nothing,AbstractVector} = nothing,
    add_intercept::Bool = true,
)
    # Input validation
    _check_weights(weights)
    n, x_vec, L, p_exog = _validate_inputs(y, x, Z; X=X)

    # Prepare exogenous matrix
    Xmat = _prep_X(n; X=X, add_intercept=add_intercept)
    p = size(Xmat, 2)  # Total exogenous parameters (including intercept)

    # Corrected degrees of freedom: n - L - p
    df = n - L - p

    # First compute LIML kappa
    κ_liml = _liml_kappa(y, x_vec, Z; X=Xmat)

    # Fuller correction: κ_Fuller = κ_LIML - a/df
    κ_fuller = κ_liml - a / df

    # Fuller estimation
    θ, u, A, invA, Adj = _kclass_fit(y, x_vec, Z; X=Xmat, k=κ_fuller)
    V = _kclass_vcov(invA, u, Adj; vcov_mode=vcov, df=df, n=n)

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
        nexogenous = p
    )
end

"""
    iv_2sls(y, x, Z; X=nothing, vcov=:HC0, weights=nothing, add_intercept=true)

Two-Stage Least Squares (2SLS) estimator for endogenous linear models.

## Arguments
- `y::AbstractVector`: Dependent variable (n×1)
- `x::AbstractVecOrMat`: Endogenous regressor(s) to be instrumented (n×k)
- `Z::AbstractMatrix`: Instruments (n×L), must have L ≥ k

## Keyword Arguments
- `X::Union{Nothing,AbstractMatrix}=nothing`: Exogenous control variables (n×p)
- `vcov::Symbol=:HC0`: Variance estimator (:homoskedastic, :HC0, :HC1)
- `weights::Union{Nothing,AbstractVector}=nothing`: Observation weights (not implemented)
- `add_intercept::Bool=true`: Whether to include intercept in the model

## Returns
`EndogenousLinearModelsEstimationResults` with 2SLS estimates and diagnostics.

## Notes
Implements 2SLS as: β̂ = (X'P_Z X)^(-1) X'P_Z y where P_Z = Z(Z'Z)^(-1)Z'.
For multiple endogenous regressors, X includes all regressors (endogenous + exogenous).
"""
function iv_2sls(
    y::AbstractVector,
    x::AbstractVecOrMat,
    Z::AbstractMatrix;
    X::Union{Nothing,AbstractMatrix} = nothing,
    vcov::Symbol = :HC0,
    weights::Union{Nothing,AbstractVector} = nothing,
    add_intercept::Bool = true,
)
    # Input validation
    _check_weights(weights)
    n, x_vec, L, p_exog = _validate_inputs(y, x, Z; X=X)

    # Prepare exogenous matrix
    Xmat = _prep_X(n; X=X, add_intercept=add_intercept)

    # For 2SLS, combine endogenous and exogenous regressors
    if size(Xmat, 2) == 0
        # No exogenous variables, just the endogenous regressor
        W = reshape(x_vec, :, 1)  # Ensure matrix form
        k = 1
    else
        # Include both endogenous and exogenous regressors
        W = hcat(reshape(x_vec, :, 1), Xmat)
        k = size(W, 2)
    end

    # Instruments must include exogenous variables for identification
    if size(Xmat, 2) > 0
        Z_full = hcat(Z, Xmat)  # Instruments include exogenous variables
    else
        Z_full = Z
    end

    L_full = size(Z_full, 2)

    # Check identification: need at least as many instruments as regressors
    if L_full < k
        throw(ArgumentError("Model is underidentified: need at least $k instruments, got $L_full"))
    end

    # 2SLS estimation using efficient implementation (avoid forming projection matrix)
    ZtZ = Symmetric(Z_full'Z_full)
    ZtW = Z_full'W
    Zty = Z_full'y

    # First stage: solve (Z'Z)^(-1) * (Z'W)
    WZ = ZtZ \ ZtW
    # Second stage coefficients: (W'P_Z W)^(-1) * (W'P_Z y)
    WPzW = ZtW' * WZ
    WPzy = ZtW' * (ZtZ \ Zty)

    β = WPzW \ WPzy

    # Second-stage residuals
    u = y - W * β

    # Degrees of freedom for 2SLS
    df = n - k

    # Variance-covariance matrix
    if vcov == :homoskedastic
        σ2 = dot(u, u) / df
        V = σ2 * Symmetric(inv(WPzW))
    elseif vcov == :HC0 || vcov == :HC1
        # Robust variance: (W'P_Z W)^(-1) * (W' P_Z Ω P_Z W) * (W'P_Z W)^(-1)
        # where Ω = diag(u_i^2) and P_Z = Z(Z'Z)^(-1)Z'
        # Efficient computation without forming full P_Z matrix
        H = Z_full' * (Z_full .* (u .^ 2))  # Z' * diag(u^2) * Z
        W_H = ZtZ \ H
        M = ZtW' * (W_H * WZ)
        V = Symmetric(inv(WPzW) * M * inv(WPzW))
        if vcov == :HC1
            V .*= n / df
        end
    else
        error("vcov must be :homoskedastic, :HC0, or :HC1")
    end

    stderr_vec = sqrt.(diag(V))

    # For consistency with LIML/Fuller, extract only the endogenous coefficient
    # and adjust for unified return format
    if size(Xmat, 2) > 0
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
        nexogenous = size(Xmat, 2)
    )
end