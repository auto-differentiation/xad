#!/bin/bash
REFERENCE_LOG="QuantLib/build/test-suite/reference.log"
BENCHMARK_LOG="QuantLib/benchmark-build/test-suite/benchmark.log"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <test_name1> [<test_name2> ... <test_nameN>]"
  exit 1
fi

TEST_NAMES=("$@")

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
      stats=$(echo "$test_times" | datamash min 1 max 1 mean 1 sstdev 1 median 1 trimmean 1 geomean 1 harmmean 1)
      echo "| Min | Max | Mean | StdDev | Median | TrimMean | GeoMean | HarmMean |"
      echo "| --- | --- | ---- | ------ | ------ | -------- | ------- | -------- |"
      echo "| $stats |"
      echo
    else
      echo "No results found for $test_name in $log_file."
    fi
  done
}

generate_markdown() {
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

process_tests "$BENCHMARK_LOG" "Benchmark"

process_tests "$REFERENCE_LOG" "Reference"

markdown="# Benchmark and Reference Results\n\n"
markdown+="## Benchmark Results\n\n"
markdown+="$(generate_markdown "$BENCHMARK_LOG" "Benchmark")\n"
markdown+="## Reference Results\n\n"
markdown+="$(generate_markdown "$REFERENCE_LOG" "Reference")\n"

echo -e "$markdown" > benchmark_results.md
echo "Benchmark and reference results saved to benchmark_results.md"