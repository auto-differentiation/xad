#!/bin/bash

LOG_DIR="$(pwd)/logs"
REFERENCE_LOG="$LOG_DIR/reference.log"
BENCHMARK_LOG="$LOG_DIR/benchmark.log"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <test_name1> [<test_name2> ... <test_nameN>]"
  exit 1
fi

TEST_NAMES=("$@")

if [[ ! -f "$REFERENCE_LOG" || ! -f "$BENCHMARK_LOG" ]]; then
  echo "Error: One or both log files do not exist in $LOG_DIR."
  exit 1
fi

normalize_time() {
  local raw_time=$1
  local unit=$2

  case "$unit" in
    "us") echo "$raw_time" ;;
    "ms") echo "$(echo "$raw_time * 1000" | bc)" ;;
    "s")  echo "$(echo "$raw_time * 1000000" | bc)" ;;
    *) echo "Error: Unknown time unit $unit" >&2; exit 1 ;;
  esac
}

process_log() {
  local log_file=$1
  local test_name=$2

  if [[ "$test_name" == test* ]]; then
    grep -P ".*?Leaving test case \"$test_name\"; testing time: [0-9]+[a-z]+" "$log_file" | \
      awk -F 'testing time: ' '{print $2}' | \
      awk '{raw_time=$1; sub(/[a-z]+$/, "", raw_time); unit=$1; sub(/^[0-9]+/, "", unit); print raw_time, unit}'
  else
    grep -P ".*?Run for \"$test_name\".*: [0-9]+[a-z]+" "$log_file" | \
      awk -F ': ' '{print $2}' | \
      awk '{raw_time=$1; sub(/[a-z]+$/, "", raw_time); unit=$1; sub(/^[0-9]+/, "", unit); print raw_time, unit}'
  fi
}

extract_normalized_times() {
  local log_file=$1
  local test_name=$2

  process_log "$log_file" "$test_name" | \
    while read -r raw_time unit; do
      normalize_time "$raw_time" "$unit"
    done
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

  results+="| Test/Example Name          | Runs  | Reference (us) | Benchmark (us) | Difference (us) | % Change |\n"
  results+="| -------------------------- | ----: | -------------: | -------------: | --------------: | -------: |\n"

  for test_name in "${TEST_NAMES[@]}"; do
    ref_times=$(extract_normalized_times "$ref_log" "$test_name")
    bench_times=$(extract_normalized_times "$bench_log" "$test_name")
    runs=$(echo "$ref_times" | wc -l)

    if [[ -n "$ref_times" && -n "$bench_times" ]]; then
      ref_median=$(echo "$ref_times" | datamash median 1 | awk '{printf "%.2f", $1}')
      bench_median=$(echo "$bench_times" | datamash median 1 | awk '{printf "%.2f", $1}')

      diff_and_percent=$(compute_difference "$ref_median" "$bench_median")
      diff=$(echo "$diff_and_percent" | awk '{printf "%.2f", $1}')
      percent=$(echo "$diff_and_percent" | awk '{printf "%.2f", $2}')

      total_diff=$(echo "$total_diff + $diff" | bc)
      total_ref=$(echo "$total_ref + $ref_median" | bc)
      total_bench=$(echo "$total_bench + $bench_median" | bc)
      total_runs=$((total_runs + runs))
      test_count=$((test_count + 1))

      results+="| $test_name                | $runs | $ref_median      | $bench_median      | $diff           | $percent% |\n"
    else
      results+="| $test_name                |    N/A |            N/A |            N/A |             N/A |      N/A |\n"
    fi
  done

  if [[ $test_count -gt 0 ]]; then
    overall_diff=$(echo "$total_ref - $total_bench" | bc | awk '{printf "%.2f", $1}')
    overall_percent=$(awk "BEGIN { printf \"%.2f\", ($overall_diff / $total_ref) * 100 }")

    results+="| **Total**                 | $total_runs | $total_ref      | $total_bench      | $overall_diff   | $overall_percent% |\n"
  fi

  echo "$results"
}

markdown="# QuantLib Benchmarks\n\n"
markdown+="$(generate_results "$REFERENCE_LOG" "$BENCHMARK_LOG")\n"

echo -e "Generated Markdown Content:\n"
echo -e "$markdown"
echo -e "$markdown" > "$LOG_DIR/benchmark_results.md"
echo "Benchmark and reference results saved to $LOG_DIR/benchmark_results.md"

echo "RESULTS_FILE=$LOG_DIR/benchmark_results.md" >> $GITHUB_ENV