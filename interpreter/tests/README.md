# openqasm-interpreter tests
## Introduction
This is the directory that contains the openqasm-interpreter test relations.

## Directory file structure
The directory structure assumed in this script is as follows
```
├─spack
├─qe-qasm
│  └─build
│      └─lib
└─RIKEN-braket
    └─interpreter
        ├─build
        └─tests
            ├─answer
            ├─qasm
            ├─qcx
            ├─result
            └─tools
```

## Description of directories
* `spack/`:Spack environment for private instances
* `qe-qasm/`:OpenQASM v3.0 parser library used by the interpreter
* `RIKEN-braket/`:RIKEN's public bra/ket repository
* `RIKEN-braket/interpreter`:Interpreter for OpenQASM v3.0
* `RIKEN-braket/interpreter/build`:Where to build the interpreter
* `RIKEN-braket/interpreter/tests`:Interpreter tests are stored.
* `RIKEN-braket/interpreter/tests/answer`:A collection of unit test justifications
* `RIKEN-braket/interpreter/tests/qcx/`: Bra programs corresponding to qasm
* `RIKEN-braket/interpreter/tests/result/`:Test Result Output Destination
* `RIKEN-braket/interpreter/tests/tools`:Unit tests and working scripts
## About Scripts
#### `tools/UnitTest*.sh/`：Individual unit tests
#### Example：
```
$ bash UnitTest001.sh
```
#### `RIKEN-braket/interpreter/tests/tools/UnitTest_all.sh/`：Script to run all unit tests
#### Example：
```
$ bash UnitTest_all.sh
```
#### `RIKEN-braket/interpreter/tests/tools/run_compute_node.sh`：Script to run UnitTest_all.sh in Fugaku
#### Example：
```
$ pjsub run_compute_node.sh
```