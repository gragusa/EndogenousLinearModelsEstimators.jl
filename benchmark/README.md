# Performance Benchmarks

This directory contains performance benchmarks for EndogenousLinearModelsEstimators.jl.

## Running Benchmarks

### Basic Usage

Run benchmarks on the current code:
```bash
julia benchmark/run_benchmarks.jl
```

### Comparison Benchmarks

Compare performance against a baseline (requires clean working directory):

```bash
# Compare against origin/main
julia benchmark/run_benchmarks.jl origin/main

# Compare against a specific branch
julia benchmark/run_benchmarks.jl my-feature-branch

# Compare against previous commit
julia benchmark/run_benchmarks.jl HEAD~1

# Compare against a tag
julia benchmark/run_benchmarks.jl v0.1.0
```

### Example Output

```
🚀 Running EndogenousLinearModelsEstimators.jl Benchmarks
============================================================
✓ Repository is clean - comparison benchmarking enabled
📊 Running benchmark comparison against: origin/main

🔍 Benchmark Results:
[Benchmark comparison table would appear here]

✅ No performance regressions detected!
```

## Files

- `benchmarks.jl` - Benchmark definitions using BenchmarkTools.jl
- `run_benchmarks.jl` - Standalone script for running benchmarks
- `tune.json` - Auto-generated tuning parameters (created on first run)

## Notes

- Benchmarks are **not** part of the automatic test suite to avoid hardware-dependent CI failures
- For comparison benchmarking, your working directory must be clean (no uncommitted changes)
- Results may vary between different machines and Julia versions
- The script will automatically fall back to current-only benchmarks if comparison isn't possible