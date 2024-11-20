#!/bin/bash
set -e

REFERENCE_LOG="QuantLib/build/test-suite/reference.log"
BENCHMARK_LOG="QuantLib/benchmark-build/test-suite/benchmark.log"

echo "Min   Max   Mean  StdDev   Median  TrimMean  GeoMean  HarmMean"

process_log() {
  local log_file=$1
  local test_name=$2

  awk -v test_name="$test_name" '
    $0 ~ test_name && /Leaving test case/ {
      match($0, /testing time: ([0-9]+)us/, arr);
      print arr[1];
    }
  ' "$log_file"
}

process_tests() {
  local log_file=$1
  local label=$2

  tests=$(awk '/Entering test case/ {print $6}' "$log_file" | sort | uniq)

  for test in $tests; do
    echo "$label results for $test:"
    test_times=$(process_log "$log_file" "$test")

    if [[ -n "$test_times" ]]; then
      echo "$test_times" | datamash min 1 max 1 mean 1 sstdev 1 median 1 trimmean 1 geomean 1 harmmean 1
    else
      echo "No results found for $test in $log_file."
    fi
  done
}

echo "*Benchmark Results*"
process_tests "$BENCHMARK_LOG" "Benchmark"

echo "*Reference Results*"
process_tests "$REFERENCE_LOG" "Reference"

benchmark_median=$(awk '$0 ~ /Leaving test case/ {match($0, /testing time: ([0-9]+)us/, arr); print arr[1]}' "$BENCHMARK_LOG" | datamash median 1)
reference_median=$(awk '$0 ~ /Leaving test case/ {match($0, /testing time: ([0-9]+)us/, arr); print arr[1]}' "$REFERENCE_LOG" | datamash median 1)

if [[ -n "$benchmark_median" && -n "$reference_median" ]]; then
  difference=$(echo "scale=9; $reference_median - $benchmark_median" | bc | awk '{printf "%.3f\n", $1}')
  percentage=$(echo "scale=9; (($reference_median - $benchmark_median) / $reference_median) * 100.0" | bc | awk '{printf "%.3f\n", $1}')

  echo "Difference: $difference"
  echo "Percentage: $percentage"
  echo "result=| ${reference_median}us | ${benchmark_median}us | ${percentage}% |" >> "$GITHUB_OUTPUT"
fi

cat <<EOF > benchmark_results.md
# Benchmark and Reference Results

## Benchmark Results
$(process_tests "$BENCHMARK_LOG" "Benchmark" | sed 's/^/  /')

## Reference Results
$(process_tests "$REFERENCE_LOG" "Reference" | sed 's/^/  /')

## Overall Difference
Difference: $difference
Percentage: $percentage%
EOF

echo "RESULTS_FILE=$(pwd)/benchmark_results.md" >> $GITHUB_ENV