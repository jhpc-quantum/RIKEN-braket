#!/bin/bash

INCLUDE_PATH="-I ../../sample"
INPUT_QASM="../qasm/test_004.qasm"
LIBRARY_PATH="../../build/qasminterpreter"
OUTPUT_RESULT="../result/UnitTest004.log"
OUTPUT_JSON="test_004.json"
ANSWER_FILE="../answer/test_004.json"
MPI_OPT="-n 1"

mpiexec $MPI_OPT -std-proc $OUTPUT_RESULT $LIBRARY_PATH $INCLUDE_PATH $INPUT_QASM

if diff -q "$OUTPUT_JSON" "$ANSWER_FILE" > /dev/null; then
    echo "Test success"
    exit 0
else
    echo "Test faild"
    exit 1 
fi
