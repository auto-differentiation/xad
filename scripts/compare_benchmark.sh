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
  local percentage_change=$(echo "scale=2; ($diff / $reference) * 100" | bc)

  echo "$diff $percentage_change"
}

generate_results() {
  local ref_log=$1
  local bench_log=$2
  local label=$3

  local results=""
  local total_diff=0
  local total_ref=0
  local total_bench=0
  local test_count=0

  for test_name in "${TEST_NAMES[@]}"; do
    ref_times=$(process_log "$ref_log" "$test_name")
    bench_times=$(process_log "$bench_log" "$test_name")

    if [[ -n "$ref_times" && -n "$bench_times" ]]; then
      ref_median=$(echo "$ref_times" | datamash median 1)
      bench_median=$(echo "$bench_times" | datamash median 1)

      diff_and_percent=$(compute_difference "$ref_median" "$bench_median")
      diff=$(echo "$diff_and_percent" | awk '{print $1}')
      percent=$(echo "$diff_and_percent" | awk '{print $2}')

      stats_ref=$(echo "$ref_times" | datamash min 1 max 1 mean 1 sstdev 1 median 1 trimmean 1 geomean 1 harmmean 1)
      stats_bench=$(echo "$bench_times" | datamash min 1 max 1 mean 1 sstdev 1 median 1 trimmean 1 geomean 1 harmmean 1)

      results+="### $test_name\n\n"
      results+="| Metric     | Reference | Benchmark | Difference | % Change |\n"
      results+="| ---------- | --------- | --------- | ---------- | -------- |\n"
      results+="| Median     | $ref_median | $bench_median | $diff | $percent% |\n"
      results+="| Min        | $(echo "$stats_ref" | awk '{print $1}') | $(echo "$stats_bench" | awk '{print $1}') | - | - |\n"
      results+="| Max        | $(echo "$stats_ref" | awk '{print $2}') | $(echo "$stats_bench" | awk '{print $2}') | - | - |\n"
      results+="| Mean       | $(echo "$stats_ref" | awk '{print $3}') | $(echo "$stats_bench" | awk '{print $3}') | - | - |\n"
      results+="| StdDev     | $(echo "$stats_ref" | awk '{print $4}') | $(echo "$stats_bench" | awk '{print $4}') | - | - |\n\n"

      total_diff=$(echo "$total_diff + $diff" | bc)
      total_ref=$(echo "$total_ref + $ref_median" | bc)
      total_bench=$(echo "$total_bench + $bench_median" | bc)
      test_count=$((test_count + 1))
    else
      results+="### $label Results for $test_name\n\nNo results found for $test_name in one or both logs.\n\n"
    fi
  done

  # Overall metrics
  if [[ $test_count -gt 0 ]]; then
    overall_diff=$(echo "$total_ref - $total_bench" | bc)
    overall_percent=$(awk "BEGIN { printf \"%.2f\", ($overall_diff / $total_ref) * 100 }")
    results+="### Overall Results\n\n"
    results+="| Metric     | Reference | Benchmark | Difference | % Change |\n"
    results+="| ---------- | --------- | --------- | ---------- | -------- |\n"
    results+="| Total      | $total_ref | $total_bench | $overall_diff | $overall_percent% |\n\n"
  fi

  echo "$results"
}

markdown="# QuantLib Benchmark and Reference Results\n\n"
markdown+="$(generate_results "$REFERENCE_LOG" "$BENCHMARK_LOG" "")\n"

echo -e "Generated Markdown Content:\n"
echo -e "$markdown"
echo -e "$markdown" > benchmark_results.md
echo "Benchmark and reference results saved to benchmark_results.md"

echo "RESULTS_FILE=$(pwd)/benchmark_results.md" >> $GITHUB_ENV