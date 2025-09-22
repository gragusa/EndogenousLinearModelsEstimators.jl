"""
Basic functionality tests for all estimators (LIML, Fuller, 2SLS)

Tests both with and without exogenous variables, different variance estimators,
and validates internal consistency.
"""

using Test
using Random
using LinearAlgebra
using Printf

@testset "Basic Estimator Tests" begin
    println("\n📊 Testing basic estimator functionality...")

    # Set up test data
    rng = Random.Xoshiro(787878)
    n, K = 50, 8
    R2, ρ, β0 = 0.15, 0.3, 1.5

    @testset "Data Generation" begin
        println("  ✓ Testing data generation...")

        # Test basic simulation
        data = simulate_iv(rng; n = n, K = K, R2 = R2, ρ = ρ, β0 = β0)
        @test length(data.y) == n
        @test length(data.x) == n
        @test size(data.z) == (n, K)

        # Test with different parameters
        data2 = simulate_iv(rng; n = 20, K = 3, R2 = 0.1, ρ = 0.1, β0 = 0.0)
        @test length(data2.y) == 20
        @test size(data2.z) == (20, 3)
    end

    @testset "LIML Estimator" begin
        println("  ✓ Testing LIML estimator...")

        data = simulate_iv(rng; n = n, K = K, R2 = R2, ρ = ρ, β0 = β0)
        outcome, endogenous, instruments = data.y, data.x, data.z

        # Basic LIML estimation
        result = liml(outcome, endogenous, instruments)
        @test isa(result, EndogenousLinearModelsEstimationResults)
        @test result.estimator == "LIML"
        @test length(result.beta) >= 1
        @test result.kappa !== nothing
        @test result.kappa > 1.0  # LIML kappa should be > 1
        @test length(result.stderr) == length(result.beta)
        @test size(result.vcov) == (length(result.beta), length(result.beta))
        @test length(result.residuals) == n
        @test result.n == n
        @test result.df > 0

        # Test different variance estimators
        result_hc0 = liml(outcome, endogenous, instruments; vcov = :HC0)
        result_hc1 = liml(outcome, endogenous, instruments; vcov = :HC1)
        result_homo = liml(outcome, endogenous, instruments; vcov = :homoskedastic)

        # Point estimates should be the same
        @test result_hc0.beta ≈ result_hc1.beta
        @test result_hc0.beta ≈ result_homo.beta
        @test result_hc0.kappa ≈ result_hc1.kappa

        # Standard errors should differ
        @test result_hc0.vcov_type == :HC0
        @test result_hc1.vcov_type == :HC1
        @test result_homo.vcov_type == :homoskedastic

        # Test with exogenous variables
        exogenous = randn(rng, n, 2)
        result_exog = liml(outcome, endogenous, instruments, exogenous)
        @test length(result_exog.beta) == 4  # endogenous + 2 exogenous + intercept
        @test result_exog.nexogenous == 3  # 2 exogenous + intercept

        # Test without intercept
        result_no_int = liml(outcome, endogenous, instruments; add_intercept = false)
        @test length(result_no_int.beta) == 1
        @test result_no_int.nexogenous == 0
    end

    @testset "Fuller Estimator" begin
        println("  ✓ Testing Fuller estimator...")

        data = simulate_iv(rng; n = n, K = K, R2 = R2, ρ = ρ, β0 = β0)
        outcome, endogenous, instruments = data.y, data.x, data.z

        # Basic Fuller estimation
        result = fuller(outcome, endogenous, instruments; a = 1.0)
        @test isa(result, EndogenousLinearModelsEstimationResults)
        @test result.estimator == "Fuller"
        @test length(result.beta) >= 1
        @test result.kappa !== nothing
        @test length(result.stderr) == length(result.beta)
        @test size(result.vcov) == (length(result.beta), length(result.beta))

        # Test a=0 should give LIML
        result_a0 = fuller(outcome, endogenous, instruments; a = 0.0)
        result_liml = liml(outcome, endogenous, instruments)
        @test result_a0.beta ≈ result_liml.beta rtol=1e-10
        @test result_a0.kappa ≈ result_liml.kappa rtol=1e-10

        # Test different alpha values
        result_a05 = fuller(outcome, endogenous, instruments; a = 0.5)
        result_a2 = fuller(outcome, endogenous, instruments; a = 2.0)

        # Fuller kappa should differ with different alpha
        @test result_a05.kappa != result.kappa
        @test result_a2.kappa != result.kappa

        # Test with exogenous variables
        exogenous = randn(rng, n, 2)
        result_exog = fuller(outcome, endogenous, instruments, exogenous; a = 1.0)
        @test length(result_exog.beta) == 4  # endogenous + 2 exogenous + intercept
        @test result_exog.nexogenous == 3
    end

    @testset "2SLS Estimator" begin
        println("  ✓ Testing 2SLS estimator...")

        data = simulate_iv(rng; n = n, K = K, R2 = R2, ρ = ρ, β0 = β0)
        outcome, endogenous, instruments = data.y, data.x, data.z

        # Basic 2SLS estimation
        result = tsls(outcome, endogenous, instruments)
        @test isa(result, EndogenousLinearModelsEstimationResults)
        @test result.estimator == "2SLS"
        @test length(result.beta) >= 1
        @test result.kappa === nothing  # 2SLS doesn't have kappa
        @test length(result.stderr) == length(result.beta)
        @test size(result.vcov) == (length(result.beta), length(result.beta))

        # Test different variance estimators
        result_hc0 = tsls(outcome, endogenous, instruments; vcov = :HC0)
        result_hc1 = tsls(outcome, endogenous, instruments; vcov = :HC1)
        result_homo = tsls(outcome, endogenous, instruments; vcov = :homoskedastic)

        # Point estimates should be the same
        @test result_hc0.beta ≈ result_hc1.beta
        @test result_hc0.beta ≈ result_homo.beta

        # Test with exogenous variables
        exogenous = randn(rng, n, 2)
        result_exog = tsls(outcome, endogenous, instruments, exogenous)
        @test length(result_exog.beta) == 4  # endogenous + 2 exogenous + intercept
        @test result_exog.nexogenous == 3
    end

    @testset "Input Validation" begin
        println("  ✓ Testing input validation...")

        data = simulate_iv(rng; n = 20, K = 3, R2 = 0.1, ρ = 0.1, β0 = 0.0)
        outcome, endogenous, instruments = data.y, data.x, data.z

        # Test dimension mismatches
        @test_throws ArgumentError liml(outcome[1:10], endogenous, instruments)  # outcome-endogenous mismatch
        @test_throws ArgumentError liml(outcome, endogenous, instruments[1:10, :])  # outcome-instruments mismatch

        # Test exogenous variable dimension mismatch
        exogenous_wrong = randn(10, 2)
        @test_throws ArgumentError liml(outcome, endogenous, instruments, exogenous_wrong)

        # Test weights not implemented
        @test_throws ErrorException liml(
            outcome,
            endogenous,
            instruments;
            weights = ones(20),
        )
        @test_throws ErrorException fuller(
            outcome,
            endogenous,
            instruments;
            weights = ones(20),
        )
        @test_throws ErrorException tsls(
            outcome,
            endogenous,
            instruments;
            weights = ones(20),
        )

        # Test underidentification for 2SLS
        instruments_under = instruments[:, 1:1]  # Only 1 instrument
        exogenous_over = randn(20, 3)  # 3 exogenous variables + intercept + endogenous = 5 params
        @test_throws ArgumentError tsls(
            outcome,
            endogenous,
            instruments_under,
            exogenous_over,
        )
    end

    @testset "Result Structure Tests" begin
        println("  ✓ Testing result structure and methods...")

        data = simulate_iv(rng; n = 30, K = 5, R2 = 0.1, ρ = 0.2, β0 = 1.0)
        outcome, endogenous, instruments = data.y, data.x, data.z

        result = liml(outcome, endogenous, instruments)

        # Test accessor methods
        @test coef(result) == result.beta
        @test stderr(result) == result.stderr
        @test vcov(result) == result.vcov
        @test residuals(result) == result.residuals
        @test dof(result) == result.df

        # Test that we can display the result without errors
        io = IOBuffer()
        show(io, result)
        output = String(take!(io))
        @test occursin("LIML Estimation Results", output)
        @test occursin("Sample size:", output)
        @test occursin("Kappa:", output)
    end

    @testset "Consistency Tests" begin
        println("  ✓ Testing cross-estimator consistency...")

        # Generate data with strong instruments (high R2)
        data = simulate_iv(rng; n = 100, K = 10, R2 = 0.5, ρ = 0.1, β0 = 1.0)
        outcome, endogenous, instruments = data.y, data.x, data.z

        result_liml = liml(outcome, endogenous, instruments; vcov = :HC0)
        result_fuller = fuller(outcome, endogenous, instruments; a = 1.0, vcov = :HC0)
        result_tsls = tsls(outcome, endogenous, instruments; vcov = :HC0)

        # With strong instruments, estimators should give similar results
        # (though not identical due to different asymptotic properties)
        @test abs(result_liml.beta[1] - result_tsls.beta[1]) < 0.5
        @test abs(result_fuller.beta[1] - result_tsls.beta[1]) < 0.5

        # All should be reasonably close to true value β0 = 1.0
        @test abs(result_liml.beta[1] - 1.0) < 0.3
        @test abs(result_fuller.beta[1] - 1.0) < 0.3
        @test abs(result_tsls.beta[1] - 1.0) < 0.3

        # All estimators should have positive semi-definite variance matrices
        # Note: With simulated data, may not always be positive definite due to numerical issues
        @test all(eigvals(result_liml.vcov) .>= -1e-10)
        @test all(eigvals(result_fuller.vcov) .>= -1e-10)
        @test all(eigvals(result_tsls.vcov) .>= -1e-10)
    end

    println("  ✅ All basic estimator tests passed!")
end
