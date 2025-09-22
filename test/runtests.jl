"""
Main test runner for EndogenousLinearModelsEstimators.jl

Runs comprehensive test suite including:
- Basic functionality tests
- Exogenous variable support tests
- Cross-platform validation tests (R/Python)
- Benchmarking tests
"""

using Test
using Random
using LinearAlgebra
using Printf

# Load the package
using EndogenousLinearModelsEstimators

# Include individual test files
include("test_estimators.jl")
include("test_cross_platform.jl")

# @testset "EndogenousLinearModelsEstimators.jl" begin
#     println("🧪 Running comprehensive test suite for EndogenousLinearModelsEstimators.jl")
#     println("=" * "="^80)

#     # Run basic estimator tests
#     include("test_estimators.jl")

#     # Run cross-platform validation tests
#     include("test_cross_platform.jl")

#     println("\n🎉 All tests completed successfully!")
# end
