# openqasm-interpreter

## Introduction

*openqasm-interpreter* is an interpreter of OpenQASM code for running quantum computer simulations.

## Requirements

*openqasm-interpreter* requires a C++11 compliant compiler.  The following libraries are required.  
- gmp
- mpfr
- mpc
- bison(>= 3.6.2)
- flex(>= 2.6.1)
- [Boost C++ library](https://www.boost.org/)
- [MPI](https://www.mpi-forum.org/)

Also, *openqasm-interpreter* uses [qe-qasm](https://github.com/openqasm/qe-qasm) as a parser for OpenQASM code.

## How to build

This section describes the build procedure at Fugaku (compute node).  
In this explanation, we show the procedure for using private instances.

### Prepare Package
Prepare the package using Spack.

1) spack environment settings
```bash
$ git clone https://github.com/RIKEN-RCCS/spack.git
$ git checkout fugaku-v0.21.0
$ . ./spack/share/spack/setup-env.sh
```

2) compiler setup
```bash
$ spack compilers
```
3) installation of packages
```bash
$ spack install cmake
$ spack install gmp
$ spack install mpfr
$ spack install mpc@1.2.1
$ spack install bison
$ spack install flex
```

### Building qe-qasm
Clone the repository to get qe-qasm.  
In this explanation, the procedure is described using the procedure conducted on Fugaku.

```bash
$ git clone https://github.com/openqasm/qe-qasm.git
```

Building qe-qasm.
1) install build dependencies  
Build the virtual environment qe-qasm_env and install build dependencies.
```bash
$ python3 -m venv ./qe-qasm_env
$ source ./qe-qasm_env/bin/activat
(qe-qasm_env) $ pip install -r requirements-dev.txt
(qe-qasm_env) $ deactivate
```
2) loading packages
```bash
# gmp
$ spack load /rx544si
# mpfr
$ spack load /7whj32d
# mpc
$ spack load /5w5gp5k
# bison
$ spack load /7d5m4dq
# flex
$ spack load /o4lwh46
# mpi
$ spack load fujitsu-mpi%gcc@12.2.0
# boost
$ spack load /epjk46e
```
3) setting environment variables
```bash
$ export LD_LIBRARY_PATH=/lib64:$LD_LIBRARY_PATH
```
4) cmake
```bash
$ cd build/
$ cmake -G "Unix Makefiles" ..
```
5) Building
```bash
$ make
```

If the following is performed and a dump of the AST is output, the build has been successful.
```bash
$ cd ./bin
$ ./QasmParser -I../../tests/include ../../tests/src/test-void-measure.qasm > ~/test-void-measure.xml
```

### Building *openqasm-interpreter*

Clone the repository and obtain *openqasm-interpreter*.

```bash
$ git clone --recursive https://github.com/jhpc-quantum/RIKEN-braket.git
```

Because MPI's command for compilation (mpicxx) is available only on compute nodes and cannot be cross-compiled, builds are performed by conversational job execution.
1) loading packages
```bash
# gmp
$ spack load /rx544si
# mpfr
$ spack load /7whj32d
# mpc
$ spack load /5w5gp5k
# bison
$ spack load /7d5m4dq
# flex
$ spack load /o4lwh46
# mpi
$ spack load fujitsu-mpi%gcc@12.2.0
# boost
$ spack load /epjk46e

```
2) setting environment variables
```bash
$ export LD_LIBRARY_PATH=./qe-qasm/build/lib:$LD_LIBRARY_PATH 
```
3) Building
```bash
$ cd ./RIKEN-braket/interpreter/
$ make
```

## Usage
The following is a description of the execution procedure at Fugaku (computation node).

1) spack environment settings
```bash
$ . ./spack/share/spack/setup-env.sh
```

2) loading packages
```bash
# gmp
$ spack load /rx544si
# mpfr
$ spack load /7whj32d
# mpc
$ spack load /5w5gp5k
# bison
$ spack load /7d5m4dq
# flex
$ spack load /o4lwh46
# mpi
$ spack load fujitsu-mpi%gcc@12.2.0
# boost
$ spack load /epjk46e

```
3) setting environment variables
```bash
$ export LD_LIBRARY_PATH=./qe-qasm/build/lib:$LD_LIBRARY_PATH 
```
4) execution
```bash
$ cd ./RIKEN-braket/interpreter/
$ mpiexec -n 1 ./qasminterpreter -I ../../qe-qasm/tests/include ./sample/test_h_cx.qasm
```

