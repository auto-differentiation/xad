
# Benchmarks

The benchmark files are defined at `xad/benchmarks` and the benchmarking script
is defined at `xad/scripts`.

## Benchmark Suite

This section will be updated as benchmark suites are added.

## Benchmark Script

The benchmark script has arguments:

```bash
./scripts/benchmark.sh <run_type: 'benchmark' | 'reference'>
[<local: bool=false>] <test1> [<test2> ... <testN>]"
```

The `run_type` tells the script whether we're running for the reference (from main)
or for our benchmark (from some other branch). The `local` parameter is used to
configure paths. If the script is running from a GitHub Actions container, this should
be set to `true`. The last arguments are the test names, as the name of directories
directly in `xad/benchmarks`.

To correctly run the benchmarks, this directory structure should be followed:

```bash
root
  ↳ main
      ↳ ...
        src
        build
        benchmarks
        scripts
        ...
  ↳ xad
      ↳ ...
        src
        build
        benchmarks
        scripts
        ...
```

A run of the script in `reference` mode from the `main` directory will generate a
`reference.json` report. Then, a run of the script in `benchmark` mode
from the `xad` repository. This will generate a `benchmark.json` report.
In the same run, it will use the report from the other run to make a table
`benchmark_results.md`. For tabulation and further analysis, it will also
create a report, `benchmark_results.json`. These files can be found in
`.../build/benchmarks`.

## Example Report

A report may look like:

**Run Date:** 2025-05-13

| Test Name | Reference (ns) | Benchmark (ns) | Difference (ns) | % Change |
| --------- | --------------:| --------------:| ---------------:| --------:|
| HessianFwdAdj | 13058.95 | 13148.58 | -89.63 | -0.69% |
| HessianFwdFwd | 36307.19 | 36136.78 | 170.41 | 0.47% |
| JacobianAdj | 359026.90 | 364655.53 | -5628.63 | -1.57% |
| JacobianFwd | 5626.40 | 5696.64 | -70.24 | -1.25% |
