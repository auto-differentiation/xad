#!/bin/bash
set -e

REFERENCE_LOG="QuantLib/build/test-suite/reference.log"
BENCHMARK_LOG="QuantLib/benchmark-build/test-suite/benchmark.log"

echo "Min   Max   Mean  StdDev   Median  TrimMean  GeoMean  HarmMean"

echo "*Output*"
OUTPUT_RESULTS=$(awk '$1 == "For" && $7 == "average" { print $8 }' "$BENCHMARK_LOG" | datamash min 1 max 1 mean 1 sstdev 1 median 1 trimmean 1 geomean 1 harmmean 1)
OUT_TIME=$(awk '$1 == "For" && $7 == "average" { print $8 }' "$BENCHMARK_LOG" | datamash median 1)

echo "*Reference*"
REFERENCE_RESULTS=$(awk '$1 == "For" && $7 == "average" { print $8 }' "$REFERENCE_LOG" | datamash min 1 max 1 mean 1 sstdev 1 median 1 trimmean 1 geomean 1 harmmean 1)
REF_TIME=$(awk '$1 == "For" && $7 == "average" { print $8 }' "$REFERENCE_LOG" | datamash median 1)

DIFFERENCE=$(echo "scale=9; $REF_TIME - $OUT_TIME" | bc | awk '{printf "%.3f\n", $1}')
echo "Difference: $DIFFERENCE"
PERCENTAGE=$(echo "scale=9; (($REF_TIME - $OUT_TIME) / $REF_TIME) * 100.0" | bc | awk '{printf "%.3f\n", $1}')
echo "Percentage: $PERCENTAGE"

echo "Benchmark Log:"
cat "$BENCHMARK_LOG"
echo "Reference Log:"
cat "$REFERENCE_LOG"

echo "result=| ${REF_TIME}ms | ${OUT_TIME}ms | ${PERCENTAGE}% |" >> "$GITHUB_OUTPUT"

cat <<EOF > benchmark_results.md

EOF

echo "RESULTS_FILE=$(pwd)/benchmark_results.md" >> $GITHUB_ENV