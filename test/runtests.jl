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
include("test_ivmodel_reference.jl")
include("test_benchmarks.jl")
