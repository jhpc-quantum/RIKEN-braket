INCLUDE_PATH="-I ../../sample"
INPUT_QASM="../qasm/test_015.qasm"
LIBRARY_PATH="../../build/qasminterpreter"
OUTPUT_RESULT="../result/UnitTest015.log"
MPI_OPT="-n 1"

mpiexec $MPI_OPT -std-proc $OUTPUT_RESULT $LIBRARY_PATH $INCLUDE_PATH $INPUT_QASM

if grep -q 'Unsupported : "t"' $OUTPUT_RESULT* && \
   grep -q 'Error detected in parse. Processing is aborted.' $OUTPUT_RESULT*; then
    echo "Test success"
    exit 0
else
    echo "Test faild"
    exit 1
fi