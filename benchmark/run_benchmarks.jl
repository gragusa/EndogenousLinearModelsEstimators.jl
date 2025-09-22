#!/usr/bin/env julia
"""
Standalone benchmark runner for EndogenousLinearModelsEstimators.jl

This script allows you to manually run performance benchmarks and compare
against a baseline (e.g., main branch) without interfering with automated tests.

Usage:
    julia benchmark/run_benchmarks.jl [baseline]

Arguments:
    baseline: Git reference to compare against (default: "origin/main")

Examples:
    julia benchmark/run_benchmarks.jl                    # Compare against origin/main
    julia benchmark/run_benchmarks.jl main               # Compare against main branch
    julia benchmark/run_benchmarks.jl HEAD~1             # Compare against previous commit
    julia benchmark/run_benchmarks.jl v0.1.0             # Compare against tag v0.1.0

Note: The working directory must be clean (no uncommitted changes) to compare
against a different git reference. If you have uncommitted changes, the script
will only run benchmarks on the current state without comparison.
"""

using Pkg
Pkg.activate(dirname(@__DIR__))

# Ensure PkgBenchmark is available
try
    using PkgBenchmark
catch
    println("📦 Installing PkgBenchmark dependency...")
    Pkg.add("PkgBenchmark")
    using PkgBenchmark
end

using Printf

const PKGDIR = dirname(@__DIR__)
const BENCH_SCRIPT = joinpath(PKGDIR, "benchmark", "benchmarks.jl")

function main(args::Vector{String} = ARGS)
    baseline = length(args) >= 1 ? args[1] : "origin/main"

    println("🚀 Running EndogenousLinearModelsEstimators.jl Benchmarks")
    println("=" ^ 60)

    # Check if we can do comparison benchmarking
    can_compare = true
    try
        # Test if repository is clean by attempting a dry-run
        run(`git diff --quiet`)
        run(`git diff --cached --quiet`)
        println("✓ Repository is clean - comparison benchmarking enabled")
    catch
        can_compare = false
        println("⚠ Repository has uncommitted changes - running current benchmarks only")
    end

    if can_compare
        println("📊 Running benchmark comparison against: $baseline")
        println()

        try
            comparison = judge(
                PKGDIR,
                baseline;
                script = BENCH_SCRIPT
            )

            println("🔍 Benchmark Results:")
            println(comparison)

            # Extract regressions
            regressions = if hasproperty(comparison, :regressions)
                getproperty(comparison, :regressions)
            elseif isdefined(PkgBenchmark, :regressions)
                PkgBenchmark.regressions(comparison)
            else
                Dict()  # Fallback if we can't extract regressions
            end

            if !isempty(regressions)
                println()
                println("⚠️  Performance Regressions Detected:")
                for (key, regression) in regressions
                    println("  - $key: $(regression)")
                end
            else
                println()
                println("✅ No performance regressions detected!")
            end

        catch e
            if occursin("not found", string(e)) || occursin("does not exist", string(e))
                println("❌ Error: Baseline '$baseline' not found")
                println("   Available branches: ")
                try
                    run(`git branch -a`)
                catch
                    println("   (Could not list branches)")
                end
                return 1
            else
                println("❌ Error running benchmark comparison: $e")
                return 1
            end
        end

    else
        println("📊 Running current benchmarks only")
        println()

        try
            result = benchmarkpkg(
                PKGDIR;
                script = BENCH_SCRIPT
            )

            println("🔍 Benchmark Results:")
            println(result)

        catch e
            println("❌ Error running benchmarks: $e")
            return 1
        end
    end

    println()
    println("✅ Benchmark run completed!")

    return 0
end

# Run if called as script
if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end