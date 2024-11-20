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
  TEST_SUITE_DIR="QuantLib/build/test-suite"
  LOG_PREFIX="reference"
elif [ "$RUN_TYPE" == "benchmark" ]; then
  TEST_SUITE_DIR="QuantLib/benchmark-build/test-suite"
  LOG_PREFIX="benchmark"
fi

echo "Running $RUN_TYPE runs for tests: ${tests[*]}"

cd "$TEST_SUITE_DIR" || exit 1
rm -f "${LOG_PREFIX}.log" || true

for TEST_NAME in "${tests[@]}"; do
  echo "Running $RUN_TYPE for: $TEST_NAME"

  ./quantlib-test-suite --log_level=test_suite --run_test="QuantLibTests/*/$TEST_NAME" | grep -v "is skipped because"

  for i in $(seq 1 $REPETITIONS); do
    ./quantlib-test-suite --log_level=test_suite --run_test="QuantLibTests/*/$TEST_NAME" | grep -v "is skipped because" | tee -a "${LOG_PREFIX}_${TEST_NAME}.log"
  done
done

cat "${LOG_PREFIX}_*.log" > "${LOG_PREFIX}.log"

echo "$RUN_TYPE runs completed. Combined log saved to ${LOG_PREFIX}.log."