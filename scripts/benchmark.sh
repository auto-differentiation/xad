#!/bin/bash

# this script requires the following directory structure:
# - $(pwd)
#   - xad
#     - scripts
#   - main
#     - scripts
#
# First, ./scripts/benchmark.sh reference <...> should be run from the _main_ repo to generate
# the reference.json file.
# Then, ./scripts/benchmark.sh benchmark <...> should be run from the _xad_ repo to generate
# the benchmark.json file. This will also generate:
# - benchmark_results.md, a table of the results
# - benchmark.json, the benchmark results as JSON, useful for plotting


if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <run_type: 'benchmark' | 'reference'> [<DIR>] [<MAIN_DIR>] <test1> [<test2> ...]"
  exit 1
fi

RUN_TYPE=$1; shift

# check if first argument is a directory override for benchmarks directory:
if [ -n "$1" ] && [ -d "$1" ]; then
  DIR="$1"
  shift
else
  DIR="$(pwd)/build/benchmarks"
fi

# check if next argument is a directory override for main/build directory:
if [ -n "$1" ] && [ -d "$1" ]; then
  MAIN_DIR="$1"
  shift
else
  if [ "$RUN_TYPE" == "reference" ]; then
    MAIN_DIR="$(pwd)/build/benchmarks"
  else
    MAIN_DIR="$(pwd)/../main/build/benchmarks"
  fi
fi

tests=("$@")

echo "$(pwd) is current directory"

# ensure required directories exist based on run type
if [ "$RUN_TYPE" == "reference" ]; then
    if [ ! -d "$MAIN_DIR" ]; then
        echo "Reference repo not found at $MAIN_DIR."
        exit 1
    fi
else
    if [ ! -d "$MAIN_DIR" ]; then
        echo "Main repo not found at $MAIN_DIR."
        exit 1
    fi
fi

echo "Running $RUN_TYPE runs for tests/examples: ${tests[*]}"

FORMAT="json"

if [ "$RUN_TYPE" == "reference" ]; then
    COMBINED_FILE="$MAIN_DIR/reference.json"
    mkdir -p "$(dirname "$COMBINED_FILE")"
    [ -f "$COMBINED_FILE" ] && rm -f "$COMBINED_FILE"
    echo "[" > "$COMBINED_FILE"
    COMMA_NEEDED=0
elif [ "$RUN_TYPE" == "benchmark" ]; then
    COMBINED_FILE="$DIR/benchmark.json"
    [ -f "$COMBINED_FILE" ] && rm -f "$COMBINED_FILE"
    echo "[" > "$COMBINED_FILE"
    COMMA_NEEDED=0
fi

for TEST_NAME in "${tests[@]}"; do

    echo "Running $RUN_TYPE test: $TEST_NAME"
    cd "$DIR/$TEST_NAME" || exit 1

    BIN_NAME="$(echo "$TEST_NAME" | tr '[:upper:]' '[:lower:]')_benchmark"
    FILE_OUT_NAME="${RUN_TYPE}_${BIN_NAME}.${FORMAT}"

    ./$BIN_NAME --benchmark_out="$FILE_OUT_NAME" --benchmark_out_format="$FORMAT"

    # append to combined file and remove the individual file
    if [ "$COMMA_NEEDED" -eq 1 ]; then
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
    echo "# Benchmark Comparison Report"
    echo ""
    FULL_DATE=$(jq -r '.[0].context.date' "$DIR/benchmark.json")
    RUN_DATE=$(echo "$FULL_DATE" | cut -d'T' -f1)
    echo "**Run Date:** $RUN_DATE"
    echo ""
    echo "| Test Name | Reference (ns) | Benchmark (ns) | Difference (ns) | % Change |"
    echo "| --------- | --------------:| --------------:| ---------------:| --------:|"
    for benchmark in $(jq -c '.[] | .benchmarks[]' "$MAIN_DIR/reference.json"); do
      bm_name=$(echo "$benchmark" | jq -r '.name')
      ref_time=$(echo "$benchmark" | jq '.real_time')
      ref_time_fmt=$(awk "BEGIN {printf \"%.2f\", $ref_time}")

      raw_bench=$(jq -r --arg name "$bm_name" '.[] | .benchmarks[] | select(.name==$name) | .real_time' "$DIR/benchmark.json")

      if [ -z "$raw_bench" ] || [ "$raw_bench" = "null" ]; then
        echo "| $bm_name | $ref_time_fmt | N/A | N/A | N/A |"
      else
        bench_time_fmt=$(awk "BEGIN {printf \"%.2f\", $raw_bench}")
        diff=$(awk "BEGIN {printf \"%.2f\", $ref_time - $raw_bench}")
        pct=$(awk "BEGIN {printf \"%.2f\", (($ref_time - $raw_bench) / $ref_time) * 100}")
        echo "| $bm_name | $ref_time_fmt | $bench_time_fmt | $diff | ${pct}% |"
      fi
    done
  } > "$RESULTS_MD"
  echo "Benchmark comparisons saved in $RESULTS_MD"
  cp "$RESULTS_MD" "$(pwd)/results.md" # for workflow
fi