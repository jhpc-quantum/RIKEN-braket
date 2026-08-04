OPENQASM 3.0;
include "stdgates.inc";

bit[2] meas;
qubit[2] q;

h q[0];
cx q[0], q[1];

meas = measure q;
