using Test
using CSV
using DataFrames
using EndogenousLinearModelsEstimators

const DATA_DIR = joinpath(@__DIR__, "data")

const IVMODEL_REFERENCE = Dict(
    # Reference point estimates / standard errors generated via R's ivmodel package.
    "no_exogenous" => Dict(
        :liml => Dict(
            :homoskedastic => (beta = 1.9336876404887946, se = 0.5126952076804047,
                kappa = 1.1603258035496644),
            :heteroskedastic => (beta = 1.9336876404887946, se = 0.6046915026576617,
                kappa = 1.1603258035496644)
        ),
        :fuller => Dict(
            :homoskedastic => (beta = 1.8056208909589633, se = 0.4482976607339658,
                kappa = 1.1076942246022961),
            :heteroskedastic => (beta = 1.8056208909589633, se = 0.5173813149417508,
                kappa = 1.1076942246022961)
        ),
        :tsls => Dict(
            :homoskedastic => (beta = 1.1347818853655336, se = 0.1908168989521417),
            :heteroskedastic => (beta = 1.1347818853655336, se = 0.1586481393982452)
        )
    ),
    "exogenous" => Dict(
        :liml => Dict(
            :homoskedastic => (beta = 1.2965861503223310, se = 0.2964167733383389,
                kappa = 1.4403639019770107),
            :heteroskedastic => (beta = 1.2965861503223310, se = 0.3116096694709095,
                kappa = 1.4403639019770107)
        ),
        :fuller => Dict(
            :homoskedastic => (beta = 1.2832402012315129, se = 0.2919925272318660,
                kappa = 1.3877323230296423),
            :heteroskedastic => (beta = 1.2832402012315129, se = 0.3000843325841815,
                kappa = 1.3877323230296423)
        ),
        :tsls => Dict(
            :homoskedastic => (beta = 1.0762813464043182, se = 0.2226316552375252),
            :heteroskedastic => (beta = 1.0762813464043182, se = 0.1796025250365075)
        )
    )
)

@testset "ivmodel reference fixtures" begin
    no_exog_df = DataFrame(CSV.File(joinpath(DATA_DIR, "no_exogenous.csv")))
    no_exog_y = no_exog_df.y
    no_exog_x = no_exog_df.x
    no_exog_Z = Matrix(select(no_exog_df, r"^z"))
    case1_reference = IVMODEL_REFERENCE["no_exogenous"]

    for (vcov_mode, ref_key) in
        zip((:homoskedastic, :HC0), (:homoskedastic, :heteroskedastic))
        liml_result = liml(no_exog_y, no_exog_x, no_exog_Z; vcov = vcov_mode)
        liml_ref = case1_reference[:liml][ref_key]
        @test isapprox(liml_result.beta[1], liml_ref.beta; atol = 1e-10, rtol = 1e-8)
        @test isapprox(stderror(liml_result)[1], liml_ref.se; atol = 1e-10, rtol = 1e-8)
        @test isapprox(liml_result.kappa, liml_ref.kappa; atol = 1e-10, rtol = 1e-8)

        fuller_result = fuller(no_exog_y, no_exog_x, no_exog_Z; vcov = vcov_mode)
        fuller_ref = case1_reference[:fuller][ref_key]
        @test isapprox(fuller_result.beta[1], fuller_ref.beta; atol = 1e-10, rtol = 1e-8)
        @test isapprox(stderror(fuller_result)[1], fuller_ref.se; atol = 1e-10, rtol = 1e-8)
        @test isapprox(fuller_result.kappa, fuller_ref.kappa; atol = 1e-10, rtol = 1e-8)

        tsls_result = tsls(no_exog_y, no_exog_x, no_exog_Z; vcov = vcov_mode)
        tsls_ref = case1_reference[:tsls][ref_key]
        @test isapprox(tsls_result.beta[1], tsls_ref.beta; atol = 1e-10, rtol = 1e-8)
        @test isapprox(stderror(tsls_result)[1], tsls_ref.se; atol = 1e-10, rtol = 1e-8)
    end

    exog_df = DataFrame(CSV.File(joinpath(DATA_DIR, "exogenous.csv")))
    exog_y = exog_df.y
    exog_x = exog_df.x
    exog_Z = Matrix(select(exog_df, r"^z"))
    exog_W = Matrix(select(exog_df, r"^exog"))
    case2_reference = IVMODEL_REFERENCE["exogenous"]

    for (vcov_mode, ref_key) in
        zip((:homoskedastic, :HC0), (:homoskedastic, :heteroskedastic))
        liml_result = liml(exog_y, exog_x, exog_Z, exog_W; vcov = vcov_mode)
        liml_ref = case2_reference[:liml][ref_key]
        @test isapprox(liml_result.beta[1], liml_ref.beta; atol = 1e-10, rtol = 1e-8)
        @test isapprox(stderror(liml_result)[1], liml_ref.se; atol = 1e-10, rtol = 1e-8)
        @test isapprox(liml_result.kappa, liml_ref.kappa; atol = 1e-10, rtol = 1e-8)

        fuller_result = fuller(exog_y, exog_x, exog_Z, exog_W; vcov = vcov_mode)
        fuller_ref = case2_reference[:fuller][ref_key]
        @test isapprox(fuller_result.beta[1], fuller_ref.beta; atol = 1e-10, rtol = 1e-8)
        @test isapprox(stderror(fuller_result)[1], fuller_ref.se; atol = 1e-10, rtol = 1e-8)
        @test isapprox(fuller_result.kappa, fuller_ref.kappa; atol = 1e-10, rtol = 1e-8)

        tsls_result = tsls(exog_y, exog_x, exog_Z, exog_W; vcov = vcov_mode)
        tsls_ref = case2_reference[:tsls][ref_key]
        @test isapprox(tsls_result.beta[1], tsls_ref.beta; atol = 1e-10, rtol = 1e-8)
        @test isapprox(stderror(tsls_result)[1], tsls_ref.se; atol = 1e-10, rtol = 1e-8)
    end
end
