using Test
using CSV
using DataFrames
using EndogenousLinearModelsEstimators

const DATA_DIR = joinpath(@__DIR__, "data")

const reference = Dict(
    # Reference point estimates / standard errors generated via R's ivmodel package.
    "no_exogenous" => Dict(
        :liml => Dict(
            :homoskedastic => (beta = 1.9336876404887946, se = 0.5126952076804047,
                kappa = 1.1603258035496644),
            :HC0 => (beta = 1.9336876404887946, se = 0.6046915026576617,
                kappa = 1.1603258035496644)
        ),
        :fuller => Dict(
            :homoskedastic => (beta = 1.8056208909589633, se = 0.4482976607339658,
                kappa = 1.1076942246022961),
            :HC0 => (beta = 1.8056208909589633, se = 0.5173813149417508,
                kappa = 1.1076942246022961)
        ),
        :tsls => Dict(
            :homoskedastic => (beta = 1.620686455524377, se = 0.3656135976108103),
            :HC0 => (beta = 1.620686455524377, se = 0.3981112423324479)
        )
    ),
    "exogenous" => Dict(
        :liml => Dict(
            :homoskedastic => (beta = 1.2965861503223310, se = 0.2964167733383389,
                kappa = 1.4403639019770107),
            :HC0 => (beta = 1.2965861503223310, se = 0.3116096694709095,
                kappa = 1.4403639019770107)
        ),
        :fuller => Dict(
            :homoskedastic => (beta = 1.2832402012315129, se = 0.2919925272318660,
                kappa = 1.3877323230296423),
            :HC0 => (beta = 1.2832402012315129, se = 0.3000843325841815,
                kappa = 1.3877323230296423)
        ),
        :tsls => Dict(
            :homoskedastic => (beta = 1.2022598360876282, se = 0.26529179840682854),
            :HC0 => (beta = 1.2022598360876282, se = 0.23759100307593714)
        )
    )
)

@testset "ivmodel reference fixtures" begin
    println("  ✓ Testing estimator without exogenous variables ...")

    no_exog_df = DataFrame(CSV.File(joinpath(DATA_DIR, "no_exogenous.csv")))
    no_exog_y = no_exog_df.y
    no_exog_x = no_exog_df.x
    no_exog_Z = Matrix(select(no_exog_df, r"^z"))
    no_exo_reference = reference["no_exogenous"]

    liml_result = liml(no_exog_y, no_exog_x, no_exog_Z; vcov = :homoskedastic)
    liml_ref = no_exo_reference[:liml][:homoskedastic]
    @test isapprox(liml_result.beta[1], liml_ref.beta; atol = 1e-10, rtol = 1e-8)
    @test isapprox(stderror(liml_result)[1], liml_ref.se; atol = 1e-10, rtol = 1e-8)
    @test isapprox(liml_result.kappa, liml_ref.kappa; atol = 1e-10, rtol = 1e-8)

    fuller_result = fuller(no_exog_y, no_exog_x, no_exog_Z; vcov = :homoskedastic)
    fuller_ref = no_exo_reference[:fuller][:homoskedastic]
    @test isapprox(fuller_result.beta[1], fuller_ref.beta; atol = 1e-10, rtol = 1e-8)
    @test isapprox(stderror(fuller_result)[1], fuller_ref.se; atol = 1e-10, rtol = 1e-8)
    @test isapprox(fuller_result.kappa, fuller_ref.kappa; atol = 1e-10, rtol = 1e-8)

    ## The basic tsls fails!!! This is shameful.
    tsls_result = tsls(no_exog_y, no_exog_x, no_exog_Z; vcov = :homoskedastic)
    tsls_ref = no_exo_reference[:tsls][:homoskedastic]
    @test isapprox(tsls_result.beta[1], tsls_ref.beta; atol = 1e-10, rtol = 1e-8)
    @test isapprox(stderror(tsls_result)[1], tsls_ref.se; atol = 1e-10, rtol = 1e-8)

    liml_result = liml(no_exog_y, no_exog_x, no_exog_Z; vcov = :HC0)
    liml_ref = no_exo_reference[:liml][:HC0]
    @test isapprox(liml_result.beta[1], liml_ref.beta; atol = 1e-10, rtol = 1e-8)
    @test isapprox(stderror(liml_result)[1], liml_ref.se; atol = 1e-10, rtol = 1e-8)
    @test isapprox(liml_result.kappa, liml_ref.kappa; atol = 1e-10, rtol = 1e-8)

    fuller_result = fuller(no_exog_y, no_exog_x, no_exog_Z; vcov = :HC0)
    fuller_ref = no_exo_reference[:fuller][:HC0]
    @test isapprox(fuller_result.beta[1], fuller_ref.beta; atol = 1e-10, rtol = 1e-8)
    @test isapprox(stderror(fuller_result)[1], fuller_ref.se; atol = 1e-10, rtol = 1e-8)
    @test isapprox(fuller_result.kappa, fuller_ref.kappa; atol = 1e-10, rtol = 1e-8)

    ## The basic tsls fails!!! This is shameful.
    tsls_result = tsls(no_exog_y, no_exog_x, no_exog_Z; vcov = :HC0)
    tsls_ref = no_exo_reference[:tsls][:HC0]
    @test isapprox(tsls_result.beta[1], tsls_ref.beta; atol = 1e-10, rtol = 1e-8)
    @test isapprox(stderror(tsls_result)[1], tsls_ref.se; atol = 1e-10, rtol = 1e-8)
    println("  ✓ Testing estimator with exogenous variables ...")
    exog_df = DataFrame(CSV.File(joinpath(DATA_DIR, "exogenous.csv")))
    exog_y = exog_df.y
    exog_x = exog_df.x
    exog_Z = Matrix(select(exog_df, r"^z"))
    exog_W = Matrix(select(exog_df, r"^exog"))
    exo_reference = reference["exogenous"]

    liml_result = liml(exog_y, exog_x, exog_Z, exog_W; vcov = :homoskedastic)
    liml_ref = exo_reference[:liml][:homoskedastic]
    @test isapprox(liml_result.beta[1], liml_ref.beta; atol = 1e-10, rtol = 1e-8)
    @test isapprox(stderror(liml_result)[1], liml_ref.se; atol = 1e-10, rtol = 1e-8)
    @test isapprox(liml_result.kappa, liml_ref.kappa; atol = 1e-10, rtol = 1e-8)

    fuller_result = fuller(exog_y, exog_x, exog_Z, exog_W; vcov = :homoskedastic)
    fuller_ref = exo_reference[:fuller][:homoskedastic]
    @test isapprox(fuller_result.beta[1], fuller_ref.beta; atol = 1e-10, rtol = 1e-8)
    @test isapprox(stderror(fuller_result)[1], fuller_ref.se; atol = 1e-10, rtol = 1e-8)
    @test isapprox(fuller_result.kappa, fuller_ref.kappa; atol = 1e-10, rtol = 1e-8)

    ## The basic tsls fails!!! This is shameful.
    tsls_result = tsls(exog_y, exog_x, exog_Z, exog_W; vcov = :homoskedastic)
    tsls_ref = exo_reference[:tsls][:homoskedastic]
    @test isapprox(tsls_result.beta[1], tsls_ref.beta; atol = 1e-10, rtol = 1e-8)
    @test isapprox(stderror(tsls_result)[1], tsls_ref.se; atol = 1e-10, rtol = 1e-8)

    liml_result = liml(exog_y, exog_x, exog_Z, exog_W; vcov = :HC0)
    liml_ref = exo_reference[:liml][:HC0]
    @test isapprox(liml_result.beta[1], liml_ref.beta; atol = 1e-10, rtol = 1e-8)
    @test isapprox(stderror(liml_result)[1], liml_ref.se; atol = 1e-10, rtol = 1e-8)
    @test isapprox(liml_result.kappa, liml_ref.kappa; atol = 1e-10, rtol = 1e-8)

    fuller_result = fuller(exog_y, exog_x, exog_Z, exog_W; vcov = :HC0)
    fuller_ref = exo_reference[:fuller][:HC0]
    @test isapprox(fuller_result.beta[1], fuller_ref.beta; atol = 1e-10, rtol = 1e-8)
    @test isapprox(stderror(fuller_result)[1], fuller_ref.se; atol = 1e-10, rtol = 1e-8)
    @test isapprox(fuller_result.kappa, fuller_ref.kappa; atol = 1e-10, rtol = 1e-8)

    ## The basic tsls fails!!! This is shameful.
    tsls_result = tsls(exog_y, exog_x, exog_Z, exog_W; vcov = :HC0)
    tsls_ref = exo_reference[:tsls][:HC0]
    @test isapprox(tsls_result.beta[1], tsls_ref.beta; atol = 1e-10, rtol = 1e-8)
    @test isapprox(stderror(tsls_result)[1], tsls_ref.se; atol = 1e-10, rtol = 1e-8)
end

println("  ✅ All basic estimator tests passed!")
