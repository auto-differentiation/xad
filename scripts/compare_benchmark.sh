#!/bin/bash
set -e

# File paths
REFERENCE_LOG="QuantLib/build/test-suite/reference.log"
BENCHMARK_LOG="QuantLib/benchmark-build/test-suite/benchmark.log"

# Print headers
echo "Min   Max   Mean  StdDev   Median  TrimMean  GeoMean  HarmMean"

# Process benchmark results
echo "*Output*"
OUTPUT_RESULTS=$(awk '$1 == "For" && $7 == "average" { print $8 }' "$BENCHMARK_LOG" | datamash min 1 max 1 mean 1 sstdev 1 median 1 trimmean 1 geomean 1 harmmean 1)
OUT_TIME=$(awk '$1 == "For" && $7 == "average" { print $8 }' "$BENCHMARK_LOG" | datamash median 1)

# Process reference results
echo "*Reference*"
REFERENCE_RESULTS=$(awk '$1 == "For" && $7 == "average" { print $8 }' "$REFERENCE_LOG" | datamash min 1 max 1 mean 1 sstdev 1 median 1 trimmean 1 geomean 1 harmmean 1)
REF_TIME=$(awk '$1 == "For" && $7 == "average" { print $8 }' "$REFERENCE_LOG" | datamash median 1)

# Calculate differences
DIFFERENCE=$(echo "scale=9; $REF_TIME - $OUT_TIME" | bc | awk '{printf "%.3f\n", $1}')
echo "Difference: $DIFFERENCE"
PERCENTAGE=$(echo "scale=9; (($REF_TIME - $OUT_TIME) / $REF_TIME) * 100.0" | bc | awk '{printf "%.3f\n", $1}')
echo "Percentage: $PERCENTAGE"

# Debugging: Print logs
echo "Benchmark Log:"
cat "$BENCHMARK_LOG"
echo "Reference Log:"
cat "$REFERENCE_LOG"

# Write results to GitHub Actions output
echo "result=| ${REF_TIME}ms | ${OUT_TIME}ms | ${PERCENTAGE}% |" >> "$GITHUB_OUTPUT"

# Write markdown results to a file
cat <<EOF > benchmark_results.md

**Output Results:**
$OUTPUT_RESULTS

**Output Time:**
$OUT_TIME

**Reference Results:**
$REFERENCE_RESULTS

**Reference Time:**
$REF_TIME

**Difference:**
$DIFFERENCE
$PERCENTAGE %
EOF

# Export the results file path for GitHub Actions
echo "RESULTS_FILE=$(pwd)/benchmark_results.md" >> $GITHUB_ENV