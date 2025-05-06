#!/bin/bash

# Run reference from main repo
# - Generates reference.json at main/build/benchmarks
#
# Run benchmark from pr repo
# - Generates benchmark.json at xad/build/benchmarks
# - Compares with reference.json
# - Generates benchmark_results.md at xad/build/benchmarks

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <run_type: 'benchmark' | 'reference'> [<local: bool=false>] <test1> [<test2> ... <testN>]"
  echo "run_type: 'reference' or 'benchmark'"
  exit 1
fi

RUN_TYPE=$1
LOCAL=$2
shift 2
tests=("$@")

if [ "$RUN_TYPE" != "reference" ] && [ "$RUN_TYPE" != "benchmark" ]; then
  echo "Error: run_type must be either 'reference' or 'benchmark'."
  exit 1
fi

echo "$(pwd) is current directory"

if [ ! -d "../main" ]; then
    echo "Reference repo not found. You must have the main repo checked out at ../main to run the benchmarks locally."
    exit 0
fi

echo "Running $RUN_TYPE runs for tests/examples: ${tests[*]}"

FORMAT="json"

if [ "$RUN_TYPE" == "reference" ]; then
    if [ "$LOCAL" == "true" ]; then
        DIR="$(pwd)/build/benchmarks"
        MAIN_DIR="$(pwd)/build/benchmarks"
    else
        DIR="/__w/xad/xad/main/build/benchmarks"
        MAIN_DIR="/__w/xad/xad/main/build/benchmarks"
    fi

    COMBINED_FILE="$MAIN_DIR/reference.json"
    if [ -f "$COMBINED_FILE" ]; then
        rm -f "$COMBINED_FILE"
    fi
    echo "[" > "$COMBINED_FILE"
    COMMA_NEEDED=0
elif [ "$RUN_TYPE" == "benchmark" ]; then
    if [ "$LOCAL" == "true" ]; then
        DIR="$(pwd)/build/benchmarks"
        MAIN_DIR="$(pwd)/../main/build/benchmarks"
    else
        DIR="/__w/xad/xad/xad/build/benchmarks"
        MAIN_DIR="/__w/xad/xad/main/build/benchmarks"
    fi


    COMBINED_FILE="$DIR/benchmark.json"
    if [ -f "$COMBINED_FILE" ]; then
        rm -f "$COMBINED_FILE"
    fi
    echo "[" > "$COMBINED_FILE"
    COMMA_NEEDED=0
fi

for TEST_NAME in "${tests[@]}"; do

    echo "Running $RUN_TYPE test: $TEST_NAME"
    cd "$DIR/$TEST_NAME" || exit 1

    BIN_NAME="$(echo "$TEST_NAME" | tr '[:upper:]' '[:lower:]')_benchmark"
    FILE_OUT_NAME="${RUN_TYPE}_${BIN_NAME}.${FORMAT}"

    ./$BIN_NAME --benchmark_out="$FILE_OUT_NAME" --benchmark_out_format=$FORMAT

    # Append to combined file and remove the individual file
    if [ $COMMA_NEEDED -eq 1 ]; then
        echo "," >> "$COMBINED_FILE"
    fi
    cat "$FILE_OUT_NAME" >> "$COMBINED_FILE"
    COMMA_NEEDED=1
    rm -f "$FILE_OUT_NAME"
done

echo "]" >> "$COMBINED_FILE"

echo "Completed $RUN_TYPE runs for tests/examples: ${tests[*]}"
echo "Results saved in $DIR/$RUN_TYPE.json"

if [ "$RUN_TYPE" != "reference" ]; then
  echo "Collecting benchmark JSON logs..."
  RESULTS_MD="$DIR/benchmark_results.md"
  {
    echo "| Test Name | Reference (ns) | Benchmark (ns) | Difference (ns) | % Change |"
    echo "| --------- | --------------:| -------------: | ---------------:| -------: |"
    for benchmark in $(jq -c '.[] | .benchmarks[]' "$MAIN_DIR/reference.json"); do
          bm_name=$(echo "$benchmark" | jq -r '.name')
          raw_ref_time=$(echo "$benchmark" | jq '.real_time')
          ref_time=$(awk "BEGIN {printf \"%f\", $raw_ref_time}")
          raw_bench_time=$(jq -r --arg name "$bm_name" '.[] | .benchmarks[] | select(.name==$name) | .real_time' "$DIR/benchmark.json")
          if [ -z "$raw_bench_time" ]; then
              bench_time="N/A"
              diff="N/A"
              pct="N/A"
          else
              bench_time=$(awk "BEGIN {printf \"%f\", $raw_bench_time}")
              diff=$(awk "BEGIN { print $ref_time - $bench_time }")
              pct=$(awk "BEGIN { print ($diff / $ref_time) * 100 }")
              ref_time=$(awk "BEGIN {printf \"%.2f\", $ref_time}")
              bench_time=$(awk "BEGIN {printf \"%.2f\", $bench_time}")
              diff=$(awk "BEGIN {printf \"%.2f\", $diff}")
          fi
          echo "| $bm_name | $ref_time | $bench_time | $diff | ${pct}% |"
    done
  } > "$RESULTS_MD"
  echo "Benchmark comparisons saved in $RESULTS_MD"
  
  cp "$RESULTS_MD" "$(pwd)/results.md" # for workflow
fi