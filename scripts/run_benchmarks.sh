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

if [ "$RUN_TYPE" == "reference" ]; then
  TEST_SUITE_DIR="$(pwd)/QuantLib/build/test-suite"
  EXAMPLES_DIR="$(pwd)/QuantLib/build/QuantLib-Risks-Cpp/Examples"
  LOG_FILE="$(pwd)/reference.log"
elif [ "$RUN_TYPE" == "benchmark" ]; then
  TEST_SUITE_DIR="$(pwd)/QuantLib/benchmark-build/test-suite"
  EXAMPLES_DIR="$(pwd)/QuantLib/benchmark-build/QuantLib-Risks-Cpp/Examples"
  LOG_FILE="$(pwd)/benchmark.log"
fi

echo "Running $RUN_TYPE runs for tests/examples: ${tests[*]}"

for TEST_NAME in "${tests[@]}"; do
  if [[ "$TEST_NAME" == test* ]]; then
    # Running as a test
    echo "Running $RUN_TYPE test: $TEST_NAME"
    cd "$TEST_SUITE_DIR" || exit 1

    ./quantlib-test-suite --log_level=test_suite --run_test="QuantLibTests/*/$TEST_NAME" | grep -v "is skipped because"

    for i in $(seq 1 $REPETITIONS); do
      ./quantlib-test-suite --log_level=test_suite --run_test="QuantLibTests/*/$TEST_NAME" | grep -v "is skipped because" | tee -a "$LOG_FILE"
    done

  else
    # Running as an example
    echo "Running $RUN_TYPE example: $TEST_NAME"
    cd "$EXAMPLES_DIR/$TEST_NAME" || exit 1

    ls -l
    file ./"$TEST_NAME"
    chmod +x ./"$TEST_NAME"
    export LD_LIBRARY_PATH=.
    echo "Warmup run for $TEST_NAME"
    ./"$TEST_NAME" 200

    for i in $(seq 1 $REPETITIONS); do
      ./"$TEST_NAME" 200 | tee -a "$LOG_FILE"
    done
    ls -l ..
  fi
done

echo "$RUN_TYPE runs completed. Combined log saved to $LOG_FILE."