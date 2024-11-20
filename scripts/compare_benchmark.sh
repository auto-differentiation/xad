#!/bin/bash

REFERENCE_LOG="QuantLib/build/test-suite/reference.log"
BENCHMARK_LOG="QuantLib/benchmark-build/test-suite/benchmark.log"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <test_name1> [<test_name2> ... <test_nameN>]"
  exit 1
fi

TEST_NAMES=("$@")

echo "Benchmark Log Content"
cat "$BENCHMARK_LOG" || echo "Benchmark log not found or empty."

echo "Reference Log Content"
cat "$REFERENCE_LOG" || echo "Reference log not found or empty."

process_log() {
  local log_file=$1
  local test_name=$2

  awk -v test_name="$test_name" '
    $0 ~ test_name && /Leaving test case/ {
      match($0, /testing time: ([0-9]+)us/, arr);
      if (arr[1] != "") print arr[1];
    }
  ' "$log_file"
}

process_tests() {
  local log_file=$1
  local label=$2

  echo "*$label Results*"
  for test_name in "${TEST_NAMES[@]}"; do
    echo "$label results for $test_name:"
    test_times=$(process_log "$log_file" "$test_name")

    if [[ -n "$test_times" ]]; then
      echo "$test_times" | datamash min 1 max 1 mean 1 sstdev 1 median 1 trimmean 1 geomean 1 harmmean 1
    else
      echo "No results found for $test_name in $log_file."
    fi
  done
}

process_tests "$BENCHMARK_LOG" "Benchmark"
process_tests "$REFERENCE_LOG" "Reference"

echo
echo "Overall Comparison:"
for test_name in "${TEST_NAMES[@]}"; do
  benchmark_median=$(process_log "$BENCHMARK_LOG" "$test_name" | datamash median 1)
  reference_median=$(process_log "$REFERENCE_LOG" "$test_name" | datamash median 1)

  if [[ -n "$benchmark_median" && -n "$reference_median" ]]; then
    difference=$(echo "scale=9; $reference_median - $benchmark_median" | bc | awk '{printf "%.3f\n", $1}')
    percentage=$(echo "scale=9; (($reference_median - $benchmark_median) / $reference_median) * 100.0" | bc | awk '{printf "%.3f\n", $1}')

    echo "Test: $test_name"
    echo "  Difference: $difference us"
    echo "  Percentage: $percentage%"
  else
    echo "Test: $test_name - Insufficient data for comparison."
  fi
done