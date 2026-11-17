OPENQASM 3.0;
include "stdgates.inc";
	
qubit q1;
	
int i = 3;
	
if (i == 3) {
  h q1;
} else {
  x q1;
}
