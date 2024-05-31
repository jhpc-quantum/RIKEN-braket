# bra

## Introduction

*bra* is an interpreter of "quantum assembler" codes to perform simulation of quantum computers.

## Requirements

*bra* requires a C++11 compliant compiler and the [Boost C++ library](https://www.boost.org/).
Any [MPI](https://www.mpi-forum.org/) libaries are also required if you would like to use *bra* in massively parallel supercomputers.

## How to build *bra*

### Using Makefile

It is not difficult to build *bra* if you have [GCC](https://gcc.gnu.org/) in your environment.
Usually *bra* can be built just by using `make`:

```bash
$ cd /path/to/braket/bra
$ make <command>
$ ls bin/
bra
$
```

You can specify `<command>` in the form of either `[nompi-]<build>[-<fp>]` or `nompi[-<fp>]`, where `[]` means optional.
For example, the realease version of *bra* in which single-precision floating-point numbers are used is built by the comand `make release-float`.

* `<build>` is to specify if the release version or the debug version is built. Possible values are "release" and "debug".
* `<fp>` is to specify which tpye of floating-point numbers is used in the simulator *bra* to represent real and imaginary parts of complex numbers. Possible values are "float" and "long", and `float` and `long double` is used, respectively. If `<fp>` is unspecified (e.g., `make release`), `double` is used by default.
* `nompi-*` represents *bra* without using an MPI library is built. MPI version is built by default if `nompi-` is unspecified (e.g., `make release`).

Note that, if no `<command>` is specified (just `make`), the release version with linking an MPI library and with using `double` as a type of floating-point numbers are built, which means `make release` is the default command.

You can build *bra* on (the login servers of) the supercomputer Fugaku in the same way, without editing Makefile.
Note that the Fujitsu C++ compiler `FCCpx` is used in this case.

### Editing Makefile

You would like to edit Makefile to build *bra* if, for example,
* you want to use a compiler other than GCC,
* you use the Fujitsu compiler, but you use other than the supercomputer Fugaku,
* you don't have the Boost library on your environment, so you download its tarball or any kinds of archive file from the [official web page](https://www.boost.org/users/download/), or
* you would like to add other compile/link options to build *bra*.

#### Other compiler

The compiler to build *bra* is specified via `CXX`.
The easiest way to change to your compiler is to replace `$(CXX)` to the command you want to use.

https://github.com/naoki-yoshioka/braket/blob/ec5460b24455de307949ad1289752af94415ef29/bra/Makefile#L174-L180

#### Fujitsu compiler, but you use other than the supercomputer Fugaku

The Makefile uses the command `uname -n` to know the hostname of the login server.
After that, it uses the `findstring` function to understand if the login server is on the supercomputer Fugaku or not.

https://github.com/naoki-yoshioka/braket/blob/ec5460b24455de307949ad1289752af94415ef29/bra/Makefile#L44-L50

You can build by using the Makefile if you change the conditional directives in the Makefile appropriately.

#### Adding include path of Boost library

The directories of libraries are specified at the line starts with `library_dirs`.

https://github.com/naoki-yoshioka/braket/blob/ec5460b24455de307949ad1289752af94415ef29/bra/Makefile#L18-L20

Add the directory of the Boost library you expanded, `/path/to/boost`, to the line.
Don't specify the *include path* itself `/path/to/boost/include` there.

#### Adding other options

Edit the lines start with `common_flags` or `cxx_flags` to add other options.

https://github.com/naoki-yoshioka/braket/blob/ec5460b24455de307949ad1289752af94415ef29/bra/Makefile#L171-L172

If you would like to add options only to the linker, edit the line starts with `LDFLAGS`.

https://github.com/naoki-yoshioka/braket/blob/ec5460b24455de307949ad1289752af94415ef29/bra/Makefile#L34-L36

## Usage

### Nompi version

*bra* can be used in the following way:

```bash
$ ./bin/bra --path <file> --threads <threads> --seed <seed>
```

* `--file <path>`: specifies the path of "quantum assembler" file. If this option is omitted, "quantum assembler" code is read from the standard input. Therefore `./bin/bra < <file>` and `/path/to/script_generating_my_excellent_quantum_circuit | ./bin/bra` are OK.
* `--threads <threads>`: specifies the number of threads. The default value is `1` if this option is omitted.
* `--seed <seed>`: specifies the initial seed of the random number generator. You can omit this option, too.

### MPI version

There are additional options other than ones of the nompi version of *bra*.

```bash
$ mpiexec -n <processes> ./bin/bra --path <file> --threads <threads> --seed <seed> --mode <mode> --unit-qubits <unit-qubits> --unit-processes <unit-processes> --page-qubits <page-qubits>
```

## Quantum assembler

So-called "quantum assembler" code is required to use *bra*.
*bra* supports all instructions presented in Appendix of [our paper](https://doi.org/10.1016/j.cpc.2018.11.005), but more gates are also supported.

The instruction set supported by *bra* is as follows.

* `I i`: identity gate operated on qubit $i$, $\hat{I} (a_0 \ket{0} + a_1 \ket{1}) = a_0 \ket{0} + a_1 \ket{1}$
* `H i`: the Hadamard gate operated on qubit $i$, $\hat{H} (a_0 \ket{0} + a_1 \ket{1}) = \dfrac{a_0 + a_1}{\sqrt{2}} \ket{0} + \dfrac{a_0 - a_1}{\sqrt{2}} \ket{1}$
* `CCCCCH c1 c2 c3 c4 c5 t` or `C5H c1 c2 c3 c4 c5 t`: the controlled Hadamard gate. Qubits $c_1$, ..., $c_5$ are control qubits, and qubit $t$ is a target qubit. You can specify upto six qubits totally. If you use three control qubits, use `CCCH c1 c2 c3 t` or `C3H c1 c2 c3 t` instead.
* `NOT i`: the NOT gate operated on qubit $i$. Note that `NOT` is an alias of `X`.
* `CCCCCNOT c1 c2 c3 c4 c5 t` or `C5NOT c1 c2 c3 c4 c5 t`: the controlled NOT gate. This is an alias of `CCCCCX c1 c2 c3 c4 c5 t` or `C5X c1 c2 c3 c4 c5 t`.
* `X i`: the Pauli $\hat{X}$ gate operated on qubit $i$, $\hat{X} (a_0 \ket{0} + a_1 \ket{1}) = a_1 \ket{0} + a_0 \ket{1}$
* `Y i`: the Pauli $\hat{Y}$ gate operated on qubit $i$, $\hat{Y} (a_0 \ket{0} + a_1 \ket{1}) = -\mathrm{i}a_1 \ket{0} + \mathrm{i}a_0 \ket{1}$
* `Z i`: the Pauli $\hat{Z}$ gate operated on qubit $i$, $\hat{Z} (a_0 \ket{0} + a_1 \ket{1}) = a_0 \ket{0} - a_1 \ket{1}$
* `XXXXXX i j k l m n` or `X6 i j k l m n`: the Pauli $\hat{X}$ gates operated on qubits $i$, ..., $n$. You can specify upto six qubits. If you use two qubits, use `XX i j` or `X2 i j` instead. The Pauli $\hat{Y}$ and $\hat{Z}$ versions are also supported.
* `CCCXXX c1 c2 c3 t1 t2 t3` or `C3X3 c1 c2 c3 t1 t2 t3`: the controlled Pauli $\hat{X}$ gates. Qubits $c_1$, $c_2$, $c_3$ are control qubits, and qubits $t_1$, $t_2$, and $t_3$ are target qubits. You can specify upto six qubits totally. If you use two target qubits and two control qubits, use `CCXX c1 c2 t1 t2` or `C2X2 c1 c2 t1 t2` instead. The Pauli $\hat{Y}$ and $\hat{Z}$ versions are also supported.
* `SWAP i j`: the SWAP gate operated on qubits $i$ and $i$, $\hat{P} (a_{00} \ket{00} + a_{01} \ket{01} + a_{10} \ket{10} + a_{11} \ket{11}) = a_{00} \ket{00} + a_{10} \ket{01} + a_{01} \ket{10} + a_{11} \ket{11}$
* `CCCCSWAP c1 c2 c3 c4 t1 t2` or `C4SWAP c1 c2 c3 c4 t1 t2`: the controlled SWAP gate. Qubits $c_1$, ..., $c_4$ are control qubits, and qubits $t_1$ and $t_2$ are target qubits. You can specify upto six qubits totally. If you use two control qubits, use `CCSWAP c1 c2 t1 t2` or `C2SWAP c1 c2 t1 t2` instead.
* `S i`: the $\hat{S}$ gate operated on qubit $i$, $\hat{S} (a_0 \ket{0} + a_1 \ket{1}) = a_0 \ket{0} + \mathrm{i}a_1 \ket{1}$
* `S+ i`: the $\hat{S}^\dagger$ gate operated on qubit $i$, $\hat{S}^\dagger (a_0 \ket{0} + a_1 \ket{1}) = a_0 \ket{0} - \mathrm{i}a_1 \ket{1}$
* `T i`: the $\hat{T}$ gate operated on qubit $i$, $\hat{T} (a_0 \ket{0} + a_1 \ket{1}) = a_0 \ket{0} + \dfrac{1 + \mathrm{i}}{\sqrt{2}} a_1 \ket{1}$
* `T+ i`: the $\hat{T}^\dagger$ gate operated on qubit $i$, $\hat{T}^\dagger (a_0 \ket{0} + a_1 \ket{1}) = a_0 \ket{0} + \dfrac{1 - \mathrm{i}}{\sqrt{2}} a_1 \ket{1}$
* `CCCCCS c1 c2 c3 c4 c5 t` or `C5S c1 c2 c3 c4 c5 t`: the controlled $\hat{S}$ gate. Qubits $c_1$, ..., $c_5$ are control qubits, and qubit $t$ is a target qubit. You can specify upto six qubits totally. If you use three control qubits, use `CCCS c1 c2 c3 t` or `C3S c1 c2 c3 t` instead. The $\hat{S}^\dagger$, $\hat{T}$, and $\hat{T}^\dagger$ versions are also supported.
* `U1 i lambda`: the phase-shift gate, or $\hat{U}_1(\lambda)$ operation, on qubit $i$, $\hat{U}_1(\lambda) (a_0 \ket{0} + a_1 \ket{1}) = a_0 \ket{0} + \mathrm{e}^{\mathrm{i} \lambda} a_1 \ket{1}$
* `U2 i phi lambda`: $\hat{U}_2(\phi, \lambda)$ operation on qubit $i$, $\hat{U}_2(\phi, \lambda) (a_0 \ket{0} + a_1 \ket{1}) = \dfrac{a_0 - \mathrm{e}^{\mathrm{i} \lambda} a_1}{\sqrt{2}} \ket{0} + \mathrm{e}^{\mathrm{i} \phi} \dfrac{a_0 + \mathrm{e}^{\mathrm{i} \lambda} a_1}{\sqrt{2}} \ket{1}$
* `U3 i theta phi lambda`: $\hat{U}_3(\theta, \phi, \lambda)$ operation on qubit $i$, $\hat{U}_3(\theta, \phi, \lambda) (a_0 \ket{0} + a_1 \ket{1}) = [\cos(\theta/2) a_0 - \sin(\theta/2) \mathrm{e}^{\mathrm{i} \lambda} a_1] \ket{0} + \mathrm{e}^{\mathrm{i} \phi} [\sin(\theta/2) a_0 + \cos(\theta/2) \mathrm{e}^{\mathrm{i} \lambda} a_1] \ket{1}$
* `CCCCCU1 c1 c2 c3 c4 c5 t lambda` or `C5U1 c1 c2 c3 c4 c5 t lambda`: the controlled phase-shift gate. Qubits $c_1$, ..., $c_5$ are control qubits, and qubit $t$ is a target qubit. You can specify upto six qubits totally. If you use three control qubits, use `CCCU1 c1 c2 c3 t lambda` or `C3U1 c1 c2 c3 t lambda` instead. The $\hat{U}_2$ and $\hat{U}_3$ versions are also supported.
* `+X i`: the +X gate which rotates qubit $i$ by $-\pi/2$ about the $x$-axis, $\hat{X}_+ (a_0 \ket{0} + a_1 \ket{1}) = \dfrac{a_0 + \mathrm{i} a_1}{\sqrt{2}} \ket{0} + \dfrac{\mathrm{i} a_0 + a_1}{\sqrt{2}} \ket{1}$
* `-X i`: the -X gate which rotates qubit $i$ by $+\pi/2$ about the $x$-axis, $\hat{X}_- (a_0 \ket{0} + a_1 \ket{1}) = \dfrac{a_0 - \mathrm{i} a_1}{\sqrt{2}} \ket{0} + \dfrac{-\mathrm{i} a_0 + a_1}{\sqrt{2}} \ket{1}$
* `+Y i`: the +Y gate which rotates qubit $i$ by $-\pi/2$ about the $y$-axis, $\hat{Y}_+ (a_0 \ket{0} + a_1 \ket{1}) = \dfrac{a_0 + a_1}{\sqrt{2}} \ket{0} + \dfrac{-a_0 + a_1}{\sqrt{2}} \ket{1}$
* `-Y i`: the -Y gate which rotates qubit $i$ by $+\pi/2$ about the $y$-axis, $\hat{Y}_- (a_0 \ket{0} + a_1 \ket{1}) = \dfrac{a_0 - a_1}{\sqrt{2}} \ket{0} + \dfrac{a_0 + a_1}{\sqrt{2}} \ket{1}$
* `CCCCC+X c1 c2 c3 c4 c5 t` or `C5+X c1 c2 c3 c4 c5 t`: the controlled +X gate. Qubits $c_1$, ..., $c_5$ are control qubits, and qubit $t$ is a target qubit. You can specify upto six qubits totally. If you use three control qubits, use `CCC+X c1 c2 c3 t` or `C3+X c1 c2 c3 t` instead. The -X, +Y, -Y versions are also supported.
* `R i k`: the phase-shift gate which changes the phase of qubit $i$ by an angle $2\pi/2^k$, $\hat{R}(k) (a_0 \ket{0} + a_1 \ket{1}) = a_0 \ket{0} + \mathrm{e}^{2\pi\mathrm{i}/2^k} a_1 \ket{1}$. Note that `R i -k` is an alias of `R+ i k`.
* `R+ i k`: the phase-shift gate which changes the phase of qubit $i$ by an angle $-2\pi/2^k$, $\hat{R}^\dagger(k) (a_0 \ket{0} + a_1 \ket{1}) = a_0 \ket{0} + \mathrm{e}^{-2\pi\mathrm{i}/2^k} a_1 \ket{1}$. Note that `R+ i -k` is an alias of `R i k`.
* `CCCCCR c1 c2 c3 c4 c5 t k` or `C5R c1 c2 c3 c4 c5 t k`: the controlled phase-shift gate which changes the phase of target qubit $t$ by an angle $2\pi/2^k$. Note that qubits $c_1$, ..., $c_5$ are control qubits. You can specify upto six qubits totally. If you use three control qubits, use `CCCR c1 c2 c3 t k` or `C3R c1 c2 c3 t k` instead. The `R+` version is also supported.
* `U c t k`: the controlled phase-shift gate. This is an alias of `CR c t k`.
* `U+ c t k`: the controlled phase-shift gate. This is an alias of `CR+ c t k`.
* `CCCCCU c1 c2 c3 c4 c5 t k` or `C5U c1 c2 c3 c4 c5 t k`: the controlled phase-shift gate. This is an alias of `CCCCCR c t k` or `C5R c t k`. Note that `CU c t k` and `C1U c t k` are aliases of `U c t k`, and therefore of `CR c t k`. The `U+` version is also supported.
* `TOFFOLI c1 c2 t1`: the TOFFOLI gate. This is an alias of `CCNOT c1 c2 t1`.
* `EX i theta`: the exponential Pauli $\hat{X}$ gate $\exp(\mathrm{i} \theta \hat{X}) = \hat{I} \cos \theta + \mathrm{i} \hat{X} \sin \theta$ operated on qubit $i$.
* `EY i theta`: the exponential Pauli $\hat{Y}$ gate $\exp(\mathrm{i} \theta \hat{Y}) = \hat{I} \cos \theta + \mathrm{i} \hat{Y} \sin \theta$ operated on qubit $i$.
* `EZ i theta`: the exponential Pauli $\hat{Z}$ gate $\exp(\mathrm{i} \theta \hat{Z}) = \hat{I} \cos \theta + \mathrm{i} \hat{Z} \sin \theta$ operated on qubit $i$.
* `EXXXXXX i j k l m n theta` or `EX6 i j k l m n theta`: the exponential Pauli $\hat{X}$ gates operated on qubits $i$, ..., $n$. You can specify upto six qubits. If you use two qubits, use `EXX i j theta` or `EX2 i j theta` instead. The Pauli $\hat{Y}$ and $\hat{Z}$ versions are also supported.
* `CCCEXXX c1 c2 c3 t1 t2 t3 theta` or `C3EX3 c1 c2 c3 t1 t2 t3 theta`: the controlled exponential Pauli $\hat{X}$ gates. Qubits $c_1$, $c_2$, $c_3$ are control qubits, and qubits $t_1$, $t_2$, and $t_3$ are target qubits. You can specify upto six qubits totally. If you use two target qubits and two control qubits, use `CCEXX c1 c2 t1 t2 theta` or `C2EX2 c1 c2 t1 t2 theta` instead. The Pauli $\hat{Y}$ and $\hat{Z}$ versions are also supported.
* `ESWAP i j theta`: the exponential SWAP gate $\exp(\mathrm{i} \theta \hat{P}) = \hat{I} \cos \theta + \mathrm{i} \hat{P} \sin \theta$ operated on qubit $i$.
* `CCCCESWAP c1 c2 c3 c4 t1 t2 theta` or `C4ESWAP c1 c2 c3 c4 t1 t2 theta`: the controlled exponential SWAP gate. Qubits $c_1$, ..., $c_4$ are control qubits, and qubits $t_1$ and $t_2$ are target qubits. You can specify upto six qubits totally. If you use two control qubits, use `CCESWAP c1 c2 t1 t2 theta` or `C2ESWAP c1 c2 t1 t2 theta` instead.
* `BEGIN MEASUREMENT`: computes and prints out the expectation values of all qubits.
* `GENERATE EVENTS n seed`: computes the probabilities of each of the basis states and exits. It generates $n$ events by using random number generator with the initial seed `seed` and prints out the states according to these probabilites.
* `M i`: projective measurement on qubit $i$.
* `QUBITS n`: specifies the number of qubits. This must be the first instruction.
* `BIT ASSIGNMENT i j k...`: specifies the initial permutation of qubits. The number of qubits specified as arguments of this instruction must be equal to the number of qubits specified in the `QUBITS n` instruction.
* `SHORBOX nx G y`
* `CLEAR i`: projects the state of qubit $i$ to $\ket{0}$.
* `SET i`: projects the state of qubit $i$ to $\ket{1}$.
* `DEPOLARIZING CHANNEL P_X=px,P_Y=py,P_Z=pz,SEED=seed`: inserts the Pauli $\hat{X}$, $\hat{Y}$, and $\hat{Z}$ gates with specified probabilities to all qubits. For example, the Pauli $\hat{X}$ gate is inserted with probability $p_x$. The random number generator uses the `seed` value as its initial seed. If the specified `seed` is negative, the value specified in the command line option of *bra* is used as the initial seed.
* `EXIT`: measures all qubits and terminate execution.

