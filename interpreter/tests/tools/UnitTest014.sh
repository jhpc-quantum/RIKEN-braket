INCLUDE_PATH="-I ../../sample"
INPUT_QASM="../qasm/test_014.qasm"
LIBRARY_PATH="../../build/qasminterpreter"
OUTPUT_RESULT="-std-proc ../../result/UnitTest014.log"
MPI_OPT="-n 1"

export LD_LIBRARY_PATH=./qe-qasm/build/lib:$LD_LIBRARY_PATH

mpiexec $MPI_OPT $OUTPUT_RESULT $LIBRARY_PATH $INCLUDE_PATH $INPUT_QASM

if grep -q 'Unsupported : "Bit"' "$OUTPUT_RESULT" && \
   grep -q 'Unsupported : "Measure"' "$OUTPUT_RESULT" && \
   grep -q 'Error detected in parse. Processing is aborted.' "$OUTPUT_RESULT"; then
    echo "Test success"
else
    echo "Test faild"
    exit 1
fi