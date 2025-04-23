#!/bin/bash

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <run_type> <repetitions> <test1> [<test2> ... <testN>]"
  echo "run_type: 'reference' or 'benchmark'"
  exit 1
fi

RUN_TYPE=$1
REPETITIONS=$2
shift 2
tests=("$@")

if [ "$RUN_TYPE" != "reference" ] && [ "$RUN_TYPE" != "benchmark" ]; then
  echo "Error: run_type must be either 'reference' or 'benchmark'."
  exit 1
fi

BASE_DIR="$(pwd)/xad/build/benchmarks"
if [ "$RUN_TYPE" == "reference" ]; then
  DIR="$BASE_DIR"
elif [ "$RUN_TYPE" == "benchmark" ]; then
  DIR="$BASE_DIR"
fi

echo "Running $RUN_TYPE runs for tests/examples: ${tests[*]}"

FORMAT="json"

for TEST_NAME in "${tests[@]}"; do

    echo "Running $RUN_TYPE test: $TEST_NAME"
    cd "$DIR/$TEST_NAME" || exit 1

    BIN_NAME="$(echo "$TEST_NAME" | tr '[:upper:]' '[:lower:]')_benchmark"
    FILE_OUT_NAME="${RUN_TYPE}_${BIN_NAME}.${FORMAT}"

    ./$BIN_NAME --benchmark_out="$FILE_OUT_NAME" --benchmark_out_format=$FORMAT
    cp "$FILE_OUT_NAME" "$DIR/"
done

echo "Completed $RUN_TYPE runs for tests/examples: ${tests[*]}"
echo "Results saved in $DIR/${RUN_TYPE}_${BIN_NAME}.${FORMAT}"

if [ "$RUN_TYPE" != "reference" ]; then
  echo "Collecting benchmark JSON logs..."
  RESULTS_MD="$DIR/benchmark_results.md"
  {
    echo "| Test Name | Reference (ns) | Benchmark (ns) | Difference (ns) | % Change |"
    echo "| --------- | --------------:| -------------: | ---------------:| -------: |"
    for ref in "$BASE_DIR"/*/reference_*_benchmark.json; do
      [ -e "$ref" ] || continue
      ref_dir=$(dirname "$ref")
      test_name=$(basename "$ref" | sed -E 's/reference_(.*)_benchmark\.json/\1/')
      bench_file="$BASE_DIR/${test_name}/benchmark_${test_name}_benchmark.json"
      for benchmark in $(jq -c '.benchmarks[]' "$ref"); do
          bm_name=$(echo "$benchmark" | jq -r '.name')
          raw_ref_time=$(echo "$benchmark" | jq '.real_time')
          ref_time=$(awk "BEGIN {printf \"%f\", $raw_ref_time}")
          if [ -f "$bench_file" ]; then
            raw_bench_time=$(jq -r --arg name "$bm_name" '.benchmarks[] | select(.name==$name) | .real_time' "$bench_file")
            if [ -z "$raw_bench_time" ]; then
              bench_time="N/A"
              diff="N/A"
              pct="N/A"
            else
              bench_time=$(awk "BEGIN {printf \"%f\", $raw_bench_time}")
              diff=$(echo "$ref_time - $bench_time" | bc -l)
              pct=$(echo "scale=3; ($diff / $ref_time) * 100" | bc -l)
              ref_time=$(awk "BEGIN {printf \"%.2f\", $ref_time}")
              bench_time=$(awk "BEGIN {printf \"%.2f\", $bench_time}")
              diff=$(awk "BEGIN {printf \"%.2f\", $diff}")
            fi
          else
            bench_time="N/A"
            diff="N/A"
            pct="N/A"
          fi
          echo "| $bm_name | $ref_time | $bench_time | $diff | ${pct}% |"
      done
    done
  } > "$RESULTS_MD"
  echo "Benchmark comparisons saved in $RESULTS_MD"
  
  cp "$RESULTS_MD" "$(pwd)/results.md" # for workflow
fi