using Test
using PkgBenchmark

const PKGDIR = dirname(@__DIR__)
const BENCH_SCRIPT = joinpath(PKGDIR, "benchmark", "benchmarks.jl")

@testset "benchmark regression vs main" begin
    comparison = judge(
        PKGDIR,
        "origin/main";
        script = BENCH_SCRIPT,
        resultdir = mktempdir(),
        progress = false,
    )

    regressions = if hasproperty(comparison, :regressions)
        getproperty(comparison, :regressions)
    elseif isdefined(PkgBenchmark, :regressions)
        PkgBenchmark.regressions(comparison)
    else
        error("Unable to extract regressions from PkgBenchmark comparison")
    end
    if !isempty(regressions)
        @info "Performance regressions detected" regressions
    end
    @test isempty(regressions)
end
