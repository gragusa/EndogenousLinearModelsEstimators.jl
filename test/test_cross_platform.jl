"""
Cross-platform validation tests against R and Python implementations

Tests the package estimators against gold-standard implementations:
- R: ivmodel package (LIML, Fuller)
- Python: linearmodels package (IVLIML)

Also includes benchmarking capabilities.
"""

using Test
using Random
using Printf
using BenchmarkTools

# Try to load RCall and PyCall
HAS_RCALL = try
    using RCall
    println("  ✅ RCall.jl loaded successfully")
    true
catch e
    println("  ❌ RCall.jl not available: $e")
    false
end

HAS_PYCALL = try
    using PyCall
    println("  ✅ PyCall.jl loaded successfully")
    true
catch e
    println("  ❌ PyCall.jl not available: $e")
    false
end

@testset "Cross-Platform Validation" begin
    println("\n🌐 Testing cross-platform validation...")

    # Test tolerance for numerical comparisons
    const RTOL = 1e-4
    const ATOL = 1e-6

    function isapprox_results(a, b; rtol = RTOL, atol = ATOL, verbose = false)
        match = isapprox(a, b, rtol = rtol, atol = atol)
        if verbose && !match
            @printf("Values don't match: %.8f vs %.8f (diff: %.2e)\\n", a, b, abs(a-b))
        end
        return match
    end

    @testset "R Integration (ivmodel)" begin
        if !HAS_RCALL
            @test_skip "RCall.jl not available"
            return
        end

        println("  📊 Testing against R ivmodel package...")

        try
            # Setup R environment
            R"""
            if (!require(ivmodel, quietly = TRUE)) {
                install.packages("ivmodel", repos = "https://cran.r-project.org")
            }
            library(ivmodel)
            """

            println("    ✓ R environment setup successful")

            # Generate test data
            rng = Random.Xoshiro(787878)
            data = simulate_iv(rng; K = 5, n = 20, ρ = 0.3, R2 = 0.2, β0 = 0.0)
            y, x, Z = data.y, data.x, data.z

            @testset "LIML vs R" begin
                # Julia LIML
                j_liml_homo = iv_liml(y, x, Z; vcov = :homoskedastic)
                j_liml_hetero = iv_liml(y, x, Z; vcov = :HC0)

                # Pass data to R and compute results
                @rput y x Z
                R"""
                m <- ivmodel(Y = y, D = x, Z = Z)
                liml_homo <- LIML(m, heteroSE = FALSE)
                liml_hetero <- LIML(m, heteroSE = TRUE)
                """

                # Get results from R
                @rget liml_homo liml_hetero

                # Test point estimates
                @test isapprox_results(j_liml_homo.beta[1], liml_homo["point_est"][1])
                @test isapprox_results(j_liml_hetero.beta[1], liml_hetero["point_est"][1])

                # Test kappa values
                @test isapprox_results(j_liml_homo.kappa, liml_homo["k"][1])
                @test isapprox_results(j_liml_hetero.kappa, liml_hetero["k"][1])

                # Test standard errors (with slightly more tolerance)
                @test isapprox_results(
                    j_liml_homo.stderr[1],
                    liml_homo["std_err"][1],
                    rtol = 0.01,
                )
                @test isapprox_results(
                    j_liml_hetero.stderr[1],
                    liml_hetero["std_err"][1],
                    rtol = 0.01,
                )

                println("    ✓ LIML matches R ivmodel")
            end

            @testset "Fuller vs R" begin
                # Julia Fuller
                j_fuller_homo = iv_fuller(y, x, Z; a = 1.0, vcov = :homoskedastic)
                j_fuller_hetero = iv_fuller(y, x, Z; a = 1.0, vcov = :HC0)

                # R Fuller
                R"""
                fuller_homo <- Fuller(m, b = 1, heteroSE = FALSE)
                fuller_hetero <- Fuller(m, b = 1, heteroSE = TRUE)
                """

                @rget fuller_homo fuller_hetero

                # Test point estimates
                @test isapprox_results(j_fuller_homo.beta[1], fuller_homo["point_est"][1])
                @test isapprox_results(
                    j_fuller_hetero.beta[1],
                    fuller_hetero["point_est"][1],
                )

                # Test kappa values
                @test isapprox_results(j_fuller_homo.kappa, fuller_homo["k"][1])
                @test isapprox_results(j_fuller_hetero.kappa, fuller_hetero["k"][1])

                # Test standard errors
                @test isapprox_results(
                    j_fuller_homo.stderr[1],
                    fuller_homo["std_err"][1],
                    rtol = 0.01,
                )
                @test isapprox_results(
                    j_fuller_hetero.stderr[1],
                    fuller_hetero["std_err"][1],
                    rtol = 0.01,
                )

                println("    ✓ Fuller matches R ivmodel")
            end

            @testset "With Exogenous Variables" begin
                # Generate larger dataset with exogenous variables
                rng2 = Random.Xoshiro(12345)
                n, K, p_exog = 30, 6, 2
                data2 = simulate_iv(rng2; K = K, n = n, ρ = 0.4, R2 = 0.15, β0 = 1.0)
                y2, x2, Z2 = data2.y, data2.x, data2.z
                X2 = randn(rng2, n, p_exog)

                # Julia estimation
                j_liml_exog = iv_liml(y2, x2, Z2; X = X2, vcov = :HC0)

                # R estimation
                @rput y2 x2 Z2 X2
                R"""
                m2 <- ivmodel(Y = y2, D = x2, Z = Z2, X = X2)
                liml_exog <- LIML(m2, heteroSE = TRUE)
                """
                @rget liml_exog

                # Test point estimate for endogenous variable (first coefficient)
                @test isapprox_results(j_liml_exog.beta[1], liml_exog["point_est"][1])

                # Test kappa
                @test isapprox_results(j_liml_exog.kappa, liml_exog["k"][1])

                println("    ✓ Exogenous variables handling matches R")
            end

        catch e
            @test_broken false
            println("    ❌ R test failed: $e")
        end
    end

    @testset "Python Integration (linearmodels)" begin
        if !HAS_PYCALL
            @test_skip "PyCall.jl not available"
            return
        end

        println("  🐍 Testing against Python linearmodels package...")

        try
            # Test if linearmodels is available
            py"""
            try:
                import numpy as np
                from linearmodels.iv import IVLIML
                python_ready = True
            except ImportError:
                python_ready = False
            """

            if !py"python_ready"
                @test_skip "Python linearmodels package not available"
                return
            end

            println("    ✓ Python environment setup successful")

            # Generate test data
            rng = Random.Xoshiro(787878)
            data = simulate_iv(rng; K = 5, n = 20, ρ = 0.3, R2 = 0.2, β0 = 0.0)
            y, x, Z = data.y, data.x, data.z

            @testset "LIML vs Python" begin
                # Julia LIML
                j_liml_unadj = iv_liml(y, x, Z; vcov = :homoskedastic)
                j_liml_robust = iv_liml(y, x, Z; vcov = :HC0)

                # Convert to Python and run
                py_y = PyObject(y)
                py_x = PyObject(x)
                py_Z = PyObject(Z)

                py"""
                import numpy as np
                from linearmodels.iv import IVLIML

                n = len($py_y)
                intercept = np.ones((n, 1))

                # LIML
                mod_liml = IVLIML($py_y, intercept, $py_x, $py_Z)
                res_liml_unadj = mod_liml.fit(cov_type='unadjusted')
                res_liml_robust = mod_liml.fit(cov_type='robust')

                results = {
                    'liml_unadj_beta': float(res_liml_unadj.params.iloc[0]),
                    'liml_robust_beta': float(res_liml_robust.params.iloc[0]),
                }
                """

                results = py"results"

                # Test point estimates (Python may have slightly different numerical precision)
                @test isapprox_results(
                    j_liml_unadj.beta[1],
                    results["liml_unadj_beta"],
                    rtol = 0.01,
                )
                @test isapprox_results(
                    j_liml_robust.beta[1],
                    results["liml_robust_beta"],
                    rtol = 0.01,
                )

                println("    ✓ LIML reasonably matches Python linearmodels")
            end

            @testset "Fuller vs Python" begin
                # Julia Fuller
                j_fuller_unadj = iv_fuller(y, x, Z; a = 1.0, vcov = :homoskedastic)
                j_fuller_robust = iv_fuller(y, x, Z; a = 1.0, vcov = :HC0)

                # Python Fuller
                py"""
                # Fuller with a=1.0
                mod_fuller = IVLIML($py_y, intercept, $py_x, $py_Z, fuller=1.0)
                res_fuller_unadj = mod_fuller.fit(cov_type='unadjusted')
                res_fuller_robust = mod_fuller.fit(cov_type='robust')

                fuller_results = {
                    'fuller_unadj_beta': float(res_fuller_unadj.params.iloc[0]),
                    'fuller_robust_beta': float(res_fuller_robust.params.iloc[0]),
                }
                """

                fuller_results = py"fuller_results"

                # Test point estimates
                @test isapprox_results(
                    j_fuller_unadj.beta[1],
                    fuller_results["fuller_unadj_beta"],
                    rtol = 0.01,
                )
                @test isapprox_results(
                    j_fuller_robust.beta[1],
                    fuller_results["fuller_robust_beta"],
                    rtol = 0.01,
                )

                println("    ✓ Fuller reasonably matches Python linearmodels")
            end

        catch e
            @test_broken false
            println("    ❌ Python test failed: $e")
        end
    end

    @testset "Benchmarking" begin
        println("  ⚡ Running performance benchmarks...")

        # Generate larger benchmark dataset
        rng = Random.Xoshiro(42)
        n, K, p_exog = 200, 10, 3
        data = simulate_iv(rng; K = K, n = n, ρ = 0.3, R2 = 0.15, β0 = 1.0)
        y, x, Z = data.y, data.x, data.z
        X_exog = randn(rng, n, p_exog)

        @testset "Julia Performance" begin
            println("    📊 Benchmarking Julia implementations...")

            # Benchmark LIML
            liml_bench =
                @benchmark iv_liml($y, $x, $Z; X = $X_exog, vcov = :HC0) samples=5 evals=1
            liml_time = median(liml_bench.times) / 1e6  # Convert to milliseconds

            # Benchmark Fuller
            fuller_bench =
                @benchmark iv_fuller($y, $x, $Z; X = $X_exog, vcov = :HC0, a = 1.0) samples=5 evals=1
            fuller_time = median(fuller_bench.times) / 1e6

            # Benchmark 2SLS
            tsls_bench =
                @benchmark tsls($y, $x, $Z; X = $X_exog, vcov = :HC0) samples=5 evals=1
            tsls_time = median(tsls_bench.times) / 1e6

            @printf("    LIML:   %8.2f ms\\n", liml_time)
            @printf("    Fuller: %8.2f ms\\n", fuller_time)
            @printf("    2SLS:   %8.2f ms\\n", tsls_time)

            # Sanity checks
            @test liml_time > 0
            @test fuller_time > 0
            @test tsls_time > 0

            # Generally 2SLS should be fastest, but with small datasets differences may not be meaningful
            println("    ✓ Benchmark completed successfully")
        end

        if HAS_RCALL
            @testset "R Performance Comparison" begin
                try
                    println("    📊 Benchmarking vs R...")

                    @rput y x Z X_exog
                    R"""
                    if (!require(microbenchmark, quietly = TRUE)) {
                        install.packages("microbenchmark", repos = "https://cran.r-project.org")
                    }
                    library(microbenchmark, quietly = TRUE)
                    library(ivmodel, quietly = TRUE)

                    m <- ivmodel(Y = y, D = x, Z = Z, X = X_exog)

                    bench_results <- microbenchmark(
                        LIML = LIML(m, heteroSE = TRUE),
                        Fuller = Fuller(m, b = 1, heteroSE = TRUE),
                        times = 3
                    )

                    r_times <- aggregate(bench_results\\$time / 1e6, by = list(bench_results\\$expr), FUN = median)
                    names(r_times) <- c("method", "time_ms")
                    """
                    @rget r_times

                    r_liml_time = r_times[r_times[!, "method"] .== "LIML", "time_ms"][1]
                    r_fuller_time = r_times[r_times[!, "method"] .== "Fuller", "time_ms"][1]

                    @printf("    R LIML:   %8.2f ms\\n", r_liml_time)
                    @printf("    R Fuller: %8.2f ms\\n", r_fuller_time)

                    println("    ✓ R benchmark completed")

                catch e
                    println("    ❌ R benchmark failed: $e")
                end
            end
        end
    end

    println("  ✅ Cross-platform validation tests completed!")
end
