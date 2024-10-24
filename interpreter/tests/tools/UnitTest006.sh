#!/bin/bash

INCLUDE_PATH="-I ../../sample"
INPUT_QASM="../qasm/test_006.qasm"
LIBRARY_PATH="../../build/qasminterpreter"
OUTPUT_RESULT="../result/UnitTest006.log"
ANSWER_FILE="../answer/test_006.json"
MPI_OPT="-n 1"

mpiexec $MPI_OPT -std-proc $OUTPUT_RESULT $LIBRARY_PATH $INCLUDE_PATH $INPUT_QASM

if diff -q "$OUTPUT_RESULT" "$ANSWER_FILE" > /dev/null; then
    echo "Test success"
else
    echo "Test faild"
    exit 1 
fi
