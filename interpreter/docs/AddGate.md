# How to add gates

## Description of directories
This section describes the directory where the files to be updated when adding a gate are stored.

```
interpreter
├─src
└─tests
    ├─answer
    ├─qasm
    ├─qcx
    └─tools
```
* `interpreter/src`:Interpreter Source Code  
* `interpreter/tests`:Unit test set  
* `interpreter/tests/answer`:Collection of answers to be used in the test  
* `interpreter/tests/qasm`:Programs for testing  
* `interpreter/tests/qcx`:bra code for comparison  
* `interpreter/tests/tools`:Scripts for unit testing  

## Interpreter Source Code
Modify files under `interpreter/src`.  
The RX gate is used as an example.

### qipTypes.h
Add as a member of `enum enumGates`.  

Example: Add `RXGate` to `enumGates`
```
enum enumGates{
    HGate,
    CXGate,
    CZGate,
    RXGate,
    RYGate,
    RZGate,
    SGate,
    SdgGate,
    XGate,
    U1Gate,
    Measure,
    NGates /// Number of gates
};
```

### IRGenQASM3Visitor.cpp
Add gate determination process to the visit function that processes `ASTGenericGateOpNode`.  

Example: If the gate name is the string “rx”, `RXGate` is set.
```
void IRGenQASM3Visitor::visit(const ASTGenericGateOpNode *node) {
  const ASTGateNode *gateNode = node->GetGateNode();
  const std::string &gateName = gateNode->GetName();

  if (gateName == "cz") {
    qasmir.gate[qasmir.ngates].id = CZGate;
  }
       :
       :
       :
  else if (gateName == "rx") {
    qasmir.gate[qasmir.ngates].id = RXGate;
  }
       :
       :
       :
}
```

### qipKet.hpp, qipKet.cpp
Define a function to process the gate and call the ket function.

Example: Function to call ket on RX gate: `qip::addRXGate()`

Add Caller
```
void qip::addGate() {
  int n = qip::qasmir.ngates;

  // Apply all gates
  for (int i = 0; i < n; i++) {
    switch (qip::qasmir.gate[i].id) {
    case HGate:
      addHGate(&qip::qasmir.gate[i]);
      break;
       :
       :
       :
    case RXGate:
      addRXGate(&qip::qasmir.gate[i]);
      break;
       :
       :
       :
    default:
      assert(0 && "Unsupported Gate");
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
}
```
Definition of `addRXGate`
```
void qip::addRXGate(gateInfoTy *ginfo) {
       :
       :
       :
}
```

## Test
Add unit tests under `interpreter/tests`.

### Unit test
Add OpenQASM file for unit test.

### Scripts
Add scripts to run unit tests.

### Run method
See `interpreter/tests/README.md` for running instructions.
