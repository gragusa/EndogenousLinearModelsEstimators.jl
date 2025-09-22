using BenchmarkTools
using Random
using EndogenousLinearModelsEstimators

const SUITE = BenchmarkGroup()
const N_OBS = 10_000
const N_INSTRUMENTS = 20
const N_EXOG = 20

const BENCH_DATA = let
    rng = Random.Xoshiro(0x7bcbe4af)
    data = EndogenousLinearModelsEstimators.simulate_iv(
        rng;
        n = N_OBS,
        K = N_INSTRUMENTS,
        R2 = 0.2,
        ρ = 0.3,
        β0 = 1.0,
    )
    y = data.y
    x = data.x
    Z = data.z
    W = randn(rng, N_OBS, N_EXOG)
    (; y, x, Z, W)
end

SUITE["liml/HC0"] = @benchmarkable liml(
    $BENCH_DATA.y,
    $BENCH_DATA.x,
    $BENCH_DATA.Z,
    $BENCH_DATA.W;
    vcov = :HC0,
) samples=3 evals=1

SUITE["fuller/HC0"] = @benchmarkable fuller(
    $BENCH_DATA.y,
    $BENCH_DATA.x,
    $BENCH_DATA.Z,
    $BENCH_DATA.W;
    vcov = :HC0,
    a = 1.0,
) samples=3 evals=1

SUITE["tsls/HC0"] = @benchmarkable tsls(
    $BENCH_DATA.y,
    $BENCH_DATA.x,
    $BENCH_DATA.Z,
    $BENCH_DATA.W;
    vcov = :HC0,
) samples=3 evals=1

SUITE
