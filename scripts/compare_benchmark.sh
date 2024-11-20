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

  grep -P ".*?Leaving test case \"$test_name\"; testing time: [0-9]+us" "$log_file" | grep -oP "[0-9]+(?=us)"
}

compute_difference() {
  local reference=$1
  local benchmark=$2

  local diff=$(echo "$reference - $benchmark" | bc)
  local percentage_change=$(awk "BEGIN { printf \"%.2f\", ($diff / $reference) * 100 }")

  echo "$diff $percentage_change"
}

generate_results() {
  local ref_log=$1
  local bench_log=$2

  local results=""
  local total_diff=0
  local total_ref=0
  local total_bench=0
  local test_count=0
  local total_runs=0

  results+="| Test Name                  | Runs  | Reference (us) | Benchmark (us) | Difference (us) | % Change |\n"
  results+="| -------------------------- | ----- | -------------- | -------------- | --------------- | -------- |\n"

  for test_name in "${TEST_NAMES[@]}"; do
    ref_times=$(process_log "$ref_log" "$test_name")
    bench_times=$(process_log "$bench_log" "$test_name")
    runs=$(echo "$ref_times" | wc -l)

    if [[ -n "$ref_times" && -n "$bench_times" ]]; then
      ref_median=$(echo "$ref_times" | datamash median 1)
      bench_median=$(echo "$bench_times" | datamash median 1)

      diff_and_percent=$(compute_difference "$ref_median" "$bench_median")
      diff=$(echo "$diff_and_percent" | awk '{print $1}')
      percent=$(echo "$diff_and_percent" | awk '{print $2}')

      total_diff=$(echo "$total_diff + $diff" | bc)
      total_ref=$(echo "$total_ref + $ref_median" | bc)
      total_bench=$(echo "$total_bench + $bench_median" | bc)
      total_runs=$((total_runs + runs))
      test_count=$((test_count + 1))

      results+="| $test_name                | $runs | $ref_median      | $bench_median      | $diff           | $percent% |\n"
    else
      results+="| $test_name                | N/A   | N/A              | N/A              | N/A             | N/A      |\n"
    fi
  done

  if [[ $test_count -gt 0 ]]; then
    overall_diff=$(echo "$total_ref - $total_bench" | bc)
    overall_percent=$(awk "BEGIN { printf \"%.2f\", ($overall_diff / $total_ref) * 100 }")

    results+="| **Total**                 | $total_runs | $total_ref      | $total_bench      | $overall_diff   | $overall_percent% |\n"
  fi

  echo "$results"
}

markdown="# QuantLib Benchmarks\n\n"
markdown+="$(generate_results "$REFERENCE_LOG" "$BENCHMARK_LOG" "")\n"

echo -e "Generated Markdown Content:\n"
echo -e "$markdown"
echo -e "$markdown" > benchmark_results.md
echo "Benchmark and reference results saved to benchmark_results.md"

echo "RESULTS_FILE=$(pwd)/benchmark_results.md" >> $GITHUB_ENV