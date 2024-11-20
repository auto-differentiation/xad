#!/bin/bash

REFERENCE_LOG="QuantLib/build/test-suite/reference.log"
BENCHMARK_LOG="QuantLib/benchmark-build/test-suite/benchmark.log"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <test_name1> [<test_name2> ... <test_nameN>]"
  exit 1
fi

TEST_NAMES=("$@")

if [[ ! -f "$REFERENCE_LOG" || ! -f "$BENCHMARK_LOG" ]]; then
  echo "Error: One or both log files do not exist."
  exit 1
fi

process_log() {
  local log_file=$1
  local test_name=$2

  grep "Leaving test case \"$test_name\"" "$log_file" | grep -oP 'testing time: [0-9]+us' | grep -oP '[0-9]+'
}

generate_results() {
  local log_file=$1
  local label=$2

  results=""
  for test_name in "${TEST_NAMES[@]}"; do
    test_times=$(process_log "$log_file" "$test_name")

    if [[ -n "$test_times" ]]; then
      stats=$(echo "$test_times" | datamash min 1 max 1 mean 1 sstdev 1 median 1 trimmean 1 geomean 1 harmmean 1)
      results+="### $label Results for $test_name\n\n"
      results+="| Min | Max | Mean | StdDev | Median | TrimMean | GeoMean | HarmMean |\n"
      results+="| --- | --- | ---- | ------ | ------ | -------- | ------- | -------- |\n"
      results+="| $stats |\n\n"
    else
      results+="### $label Results for $test_name\n\nNo results found for $test_name in $log_file.\n\n"
    fi
  done
  echo "$results"
}

markdown="# Benchmark and Reference Results\n\n"
markdown+="## Benchmark Results\n\n"
markdown+="$(generate_results "$BENCHMARK_LOG" "Benchmark")\n"
markdown+="## Reference Results\n\n"
markdown+="$(generate_results "$REFERENCE_LOG" "Reference")\n"

echo -e "$markdown" > benchmark_results.md
echo "Benchmark and reference results saved to benchmark_results.md"