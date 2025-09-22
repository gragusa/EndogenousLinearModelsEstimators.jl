"""
# Utility Functions

Shared utility functions for endogenous linear model estimation.
Extracted and unified from the existing kclass.jl implementation.
"""

using Random
using LinearAlgebra
using Statistics

# Vector conversion utility
"""
    _asvec(u)

Convert matrix input to vector if it's n×1, otherwise return as-is.
"""
_asvec(u::AbstractVector) = u
function _asvec(u::AbstractMatrix)
    size(u, 2) == 1 ? vec(u) :
    throw(ArgumentError("Expected a vector or n×1 matrix, got $(size(u))"))
end

"""
    _qr_resid(A, B)

Compute residuals B | A using pivoted QR decomposition (matches R's qr.resid).
Returns resid(B|A) = B - A * (qr(A, pivoted) \\\\ B).
"""
function _qr_resid(A::AbstractMatrix, B::AbstractVecOrMat)
    if size(A, 2) == 0
        return B
    end
    F = qr(A, ColumnNorm())        # rank-revealing QR (pivoted)
    return B .- A * (F \ B)
end

"""
    _prep_X(n; X=nothing, add_intercept=true)

Prepare exogenous matrix X. If X is nothing or empty, returns intercept only (if add_intercept=true).
If add_intercept=true, prepends a column of ones to X.
"""
function _prep_X(
        n::Int;
        X::Union{Nothing, AbstractMatrix} = nothing,
        add_intercept::Bool = true
)
    if X === nothing || size(X, 2) == 0
        return add_intercept ? ones(n, 1) : zeros(n, 0)
    else
        return add_intercept ? hcat(ones(n), X) : X
    end
end

"""
    _rank_Z_given_X(Z, X)

Compute rank of Z residualized on X using rank-revealing QR.
"""
_rank_Z_given_X(Z::AbstractMatrix, X::AbstractMatrix) = rank(_qr_resid(X, Z))

"""
    _liml_kappa(y, d, Z; X)

Compute LIML kappa using generalized symmetric eigenproblem.
Stable implementation avoiding explicit matrix inverse.
"""
function _liml_kappa(
        y::AbstractVector,
        d::AbstractVector,
        Z::AbstractMatrix;
        X::AbstractMatrix
)
    # adj variables: residualize on X (mirrors R ivmodel$Yadj, $Dadj, $Zadj)
    Yadj = _qr_resid(X, y)
    Dadj = _qr_resid(X, d)
    Zadj = _qr_resid(X, Z)

    # M1 = R'R
    yy = dot(Yadj, Yadj)
    yd = dot(Yadj, Dadj)
    dd = dot(Dadj, Dadj)
    M1 = Symmetric([yy yd; yd dd])

    # M2 via residuals on Zadj: R'(I - P_Zadj)R
    Yres = _qr_resid(Zadj, Yadj)
    Dres = _qr_resid(Zadj, Dadj)
    yyr = dot(Yres, Yres)
    ydr = dot(Yres, Dres)
    ddr = dot(Dres, Dres)
    M2 = Symmetric([yyr ydr; ydr ddr])

    vals = try
        eigen(M1, M2).values
    catch
        τ = max(eps(eltype(M2))*tr(M2), 1e-12)
        eigen(M1, Symmetric(Matrix(M2) + τ*I)).values
    end
    return minimum(real(vals))     # LIML uses the smaller generalized eigenvalue
end

"""
    _kclass_fit(y, d, Z; X, k)

Core K-class fitting routine. Returns (θ, u, A, invA, Adj) where:
- θ: K-class coefficient vector (first element is for d)
- u: residuals y - W*θ
- A: W'W - k W'W_res
- invA: inv(A)
- Adj: W - k W_res (adjustment matrix for variance calculation)
"""
function _kclass_fit(
        y::AbstractVector,
        d::AbstractVector,
        Z::AbstractMatrix;
        X::AbstractMatrix,
        k::Real
)
    ZX = hcat(Z, X)                  # n×(L+p)
    W = hcat(d, X)                   # n×(1+p)
    Wres = _qr_resid(ZX, W)          # qr.resid(ZXQR, W)
    Yrzx = _qr_resid(ZX, y)          # qr.resid(ZXQR, Y)

    A = W' * W .- k .* (W' * Wres)
    b = W' * y .- k .* (W' * Yrzx)
    θ = A \ b
    u = y .- W * θ
    invA = inv(A)                    # small (1+p)×(1+p); fine to invert explicitly
    Adj = W .- k .* Wres             # rows are the R "adjustVec"
    return θ, u, A, invA, Adj
end

"""
    _kclass_vcov(invA, u, Adj; vcov_mode, df, n)

Compute variance-covariance matrix for K-class estimate.
vcov_mode ∈ (:homoskedastic, :HC0, :HC1)
"""
function _kclass_vcov(
        invA::AbstractMatrix,
        u::AbstractVector,
        Adj::AbstractMatrix;
        vcov_mode::Symbol,
        df::Int,
        n::Int
)
    if vcov_mode == :homoskedastic
        σ2 = sum(abs2, u) / df
        return σ2 .* invA
    else
        # HC0 / HC1 sandwich
        meat = (Adj .* u)' * (Adj .* u)
        if vcov_mode == :HC1
            meat .*= n / df
        end
        return invA * meat * invA
    end
end

"""
    _check_weights(weights)

Check if weights are specified and error if they are (not implemented yet).
"""
function _check_weights(weights)
    if weights !== nothing
        error("Weights are not implemented yet. Please set weights=nothing.")
    end
end

"""
    _validate_inputs(y, x, Z; X=nothing)

Validate input dimensions and types.
"""
function _validate_inputs(y::AbstractVector, x::AbstractVecOrMat, Z::AbstractMatrix;
        X::Union{Nothing, AbstractMatrix} = nothing)
    n = length(y)

    # Check basic dimensions
    if size(Z, 1) != n
        throw(ArgumentError("y and Z must have the same number of rows"))
    end

    x_vec = _asvec(x)
    if length(x_vec) != n
        throw(ArgumentError("y and x must have the same length"))
    end

    # Check exogenous variables if provided
    if X !== nothing && size(X, 1) != n
        throw(ArgumentError("y and X must have the same number of rows"))
    end

    # Check for sufficient instruments
    L = size(Z, 2)
    p = X === nothing ? 0 : size(X, 2)
    total_params = 1 + p + 1  # endogenous + exogenous + intercept

    if L < 1
        throw(ArgumentError("At least one instrument is required"))
    end

    return n, x_vec, L, p
end

"""
    simulate_iv(rng=Random.default_rng(); n, K=1, R2=0.1, ρ=0.1, β0=0.0)

Simulate one sample from the linear IV model:
y = x*β0 + ε ;  x = Z*γ + u
- Z ~ N(0, I_K)
- (ε, u) ~ N(0, Σ) with Σ = [1 ρ; ρ 1]

Returns (y::Vector, x::Vector, Z::Matrix).
"""
function simulate_iv(
        rng = Random.default_rng();
        n::Int,
        K::Int = 1,
        R2::Float64 = 0.1,
        ρ::Float64 = 0.1,
        β0::Float64 = 0.0
)
    @assert -0.999 ≤ ρ ≤ 0.999 "ρ must be in [-0.999, 0.999] for a valid covariance."
    γ = _gamma_vector(K, R2)

    # Draw instruments
    Z = randn(rng, n, K)

    # Draw (ε, u) with correlation ρ
    Σ = [1.0 ρ; ρ 1.0]
    U = cholesky(Symmetric(Σ)).U
    E = randn(rng, n, 2) * U
    ε = view(E, :, 1)
    u = view(E, :, 2)

    x = Z * γ .+ u
    y = x .* β0 .+ ε

    return (y = y, x = x[:, :], z = Z)
end

"""
    _gamma_vector(K, R2)

Return γ ∈ ℝ^K such that, with Z ~ N(0, I_K) and Var(u)=1, the first-stage R² is R2.
γ = sqrt(R2 / (K*(1 - R2))) * ones(K)
"""
function _gamma_vector(K::Int, R2::Float64)
    @assert 0.0 ≤ R2 < 1.0 "R2 must be in [0,1)."
    scale = sqrt(R2 / (K * (1 - R2)))
    return fill(scale, K)
end
