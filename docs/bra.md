# bra

## Introduction

*bra* is an interpreter of "quantum assembler" codes to perform simulation of quantum computers.

## Requirements

*bra* requires a C++14 compliant compiler and the [Boost C++ library](https://www.boost.org/).
Any [MPI](https://www.mpi-forum.org/) libaries are also required if you would like to use *bra* in massively parallel supercomputers.

## How to build *bra*

### Using Makefile

It is not difficult to build *bra* if you have [GCC](https://gcc.gnu.org/) in your environment.
Usually *bra* can be built just by using `make`:

```bash
$ cd /path/to/RIKEN-braket/bra
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

https://github.com/jhpc-quantum/RIKEN-braket/blob/ec5460b24455de307949ad1289752af94415ef29/bra/Makefile#L174-L180

#### Fujitsu compiler, but you use other than the supercomputer Fugaku

The Makefile uses the command `uname -n` to know the hostname of the login server.
After that, it uses the `findstring` function to understand if the login server is on the supercomputer Fugaku or not.

https://github.com/jhpc-quantum/RIKEN-braket/blob/ec5460b24455de307949ad1289752af94415ef29/bra/Makefile#L44-L50

You can build by using the Makefile if you change the conditional directives in the Makefile appropriately.

#### Adding include path of Boost library

The directories of libraries are specified at the line starts with `library_dirs`.

https://github.com/jhpc-quantum/RIKEN-braket/blob/ec5460b24455de307949ad1289752af94415ef29/bra/Makefile#L18-L20

Add the directory of the Boost library you expanded, `/path/to/boost`, to the line.
Don't specify the *include path* itself `/path/to/boost/include` there.

#### Adding other options

Edit the lines start with `common_flags` or `cxx_flags` to add other options.

https://github.com/jhpc-quantum/RIKEN-braket/blob/ec5460b24455de307949ad1289752af94415ef29/bra/Makefile#L171-L172

If you would like to add options only to the linker, edit the line starts with `LDFLAGS`.

https://github.com/jhpc-quantum/RIKEN-braket/blob/ec5460b24455de307949ad1289752af94415ef29/bra/Makefile#L34-L36

## Usage

### Nompi version

*bra* can be used in the following way:

```bash
$ ./bin/bra --file <path> --threads <threads> --seed <seed>
```

* `--file <path>`: specifies the path of "quantum assembler" file. If this option is omitted, "quantum assembler" code is read from the standard input. Therefore `./bin/bra < <path>` and `/path/to/script_generating_my_excellent_quantum_circuit | ./bin/bra` are OK.
* `--threads <threads>`: specifies the number of threads. The default value is `1` if this option is omitted.
* `--seed <seed>`: specifies the initial seed of the random number generator. You can omit this option, too.

### MPI version

There are additional options other than ones of the nompi version of *bra*.

```bash
$ mpiexec -n <processes> ./bin/bra --file <path> --threads <threads> --seed <seed> --mode <mode> --unit-qubits <unit-qubits> --unit-processes <unit-processes> --page-qubits <page-qubits>
```

## Quantum assembler

So-called "quantum assembler" code is required to use *bra*.
*bra* supports all instructions presented in Appendix of [our paper](https://doi.org/10.1016/j.cpc.2018.11.005), but more gates are also supported.

The instruction set supported by *bra* is as follows[^1].

[^1]: There are some multi-qubit gates in this instruction set. The number of qubits (sum of target and control qubits) which can be specified is determined by the macro `BRA_MAX_NUM_OPERATED_QUBITS`, which can be set when building *bra*. Its default value is 10.

* `I i`: identity gate operated on qubit $i$, $I (a_0 \ket{0} + a_1 \ket{1}) = a_0 \ket{0} + a_1 \ket{1}$. This gate performs MPI communication of a gate with one qubit, e.g., `X i`.
* `IC c`: identity gate operated on control qubit $c$, $I_\mathrm{C} (a_0 \ket{0} + a_1 \ket{1}) = a_0 \ket{0} + a_1 \ket{1}$. This gate performs MPI communication of a gate with one control qubit, e.g., `Z c`.
* `CCCIII c1 c2 c3 t1 t2 t3` or `C3I3 c1 c2 c3 t1 t2 t3`: the controlled identity gate. Qubits $c_1$, ..., $c_3$ are control qubits and $t_1$, ..., $t_3$ are target qubits. If you use two control qubits and two target qubits, use `CCII c1 c2 t1 t2` or `C2I2 c1 c2 t1 t2` instead. This gate performs MPI comunication of a gate with control and target qubits, e.g., `CCCZZZ c1 c2 c3 t1 t2 t3`.
* `CCCCCIC c1 c2 c3 c4 c5 c6` or `C5IC c1 c2 c3 c4 c5 c6`: the controlled identity gate. Qubits $c_1$, ..., $c_6$ are control qubits. If you use three control qubits, use `CCIC c1 c2 c3` or `C2IC c1 c2 c3 c3` instead. This gate performs MPI comunication of a gate with control qubits, e.g., `CCCCCZ c1 c2 c3 c4 c5 c6`.
* `H t`: the Hadamard gate operated on qubit $t$, $H (a_0 \ket{0} + a_1 \ket{1}) = \dfrac{a_0 + a_1}{\sqrt{2}} \ket{0} + \dfrac{a_0 - a_1}{\sqrt{2}} \ket{1}$
* `CCCCCH c1 c2 c3 c4 c5 t` or `C5H c1 c2 c3 c4 c5 t`: the controlled Hadamard gate. Qubits $c_1$, ..., $c_5$ are control qubits, and qubit $t$ is a target qubit. If you use three control qubits, use `CCCH c1 c2 c3 t` or `C3H c1 c2 c3 t` instead.
* `NOT t`: the NOT gate operated on qubit $t$. Note that `NOT` is an alias of `X`.
* `CCCCCNOT c1 c2 c3 c4 c5 t` or `C5NOT c1 c2 c3 c4 c5 t`: the controlled NOT gate. This is an alias of `CCCCCX c1 c2 c3 c4 c5 t` or `C5X c1 c2 c3 c4 c5 t`.
* `X t`: the Pauli $X$ gate operated on qubit $t$, $X (a_0 \ket{0} + a_1 \ket{1}) = a_1 \ket{0} + a_0 \ket{1}$
* `Y t`: the Pauli $Y$ gate operated on qubit $t$, $Y (a_0 \ket{0} + a_1 \ket{1}) = -\mathrm{i}a_1 \ket{0} + \mathrm{i}a_0 \ket{1}$
* `Z c`: the Pauli $Z$ gate operated on control qubit $c$, $Z (a_0 \ket{0} + a_1 \ket{1}) = a_0 \ket{0} - a_1 \ket{1}$
* `XXXXXX t1 t2 t3 t4 t5 t6` or `X6 t1 t2 t3 t4 t5 t6`: the Pauli $X$ gates operated on qubits $t1$, ..., $t6$. If you use two qubits, use `XX t1 t2` or `X2 t1 t2` instead. The Pauli $Y$ and $Z$ versions are also supported.
* `CCCXXX c1 c2 c3 t1 t2 t3` or `C3X3 c1 c2 c3 t1 t2 t3`: the controlled Pauli $X$ gates. Qubits $c_1$, $c_2$, $c_3$ are control qubits, and qubits $t_1$, $t_2$, and $t_3$ are target qubits. If you use two target qubits and two control qubits, use `CCXX c1 c2 t1 t2` or `C2X2 c1 c2 t1 t2` instead. The Pauli $Y$ and $Z$ versions are also supported. Note that qubits of `CCCCCZ` or any `CnZ` gates are control ones.
* `SWAP t1 t2`: the SWAP gate operated on qubits $t1$ and $t2$, $P (a_{00} \ket{00} + a_{01} \ket{01} + a_{10} \ket{10} + a_{11} \ket{11}) = a_{00} \ket{00} + a_{10} \ket{01} + a_{01} \ket{10} + a_{11} \ket{11}$
* `CCCCSWAP c1 c2 c3 c4 t1 t2` or `C4SWAP c1 c2 c3 c4 t1 t2`: the controlled SWAP gate. Qubits $c_1$, ..., $c_4$ are control qubits, and qubits $t_1$ and $t_2$ are target qubits. If you use two control qubits, use `CCSWAP c1 c2 t1 t2` or `C2SWAP c1 c2 t1 t2` instead.
* `SX t`: the square-root Pauli $\sqrt{X}$ gate operated on qubit $t$, $\sqrt{X} (a_0 \ket{0} + a_1 \ket{1}) = \biggl( \frac{1 + \mathrm{i}}{2} a_0 + \frac{1 - \mathrm{i}}{2} a_1 \biggr) \ket{0} + \biggl( \frac{1 - \mathrm{i}}{2} a_0 + \frac{1 + \mathrm{i}}{2} a_1 \biggr) \ket{1}$
* `SX+ t`: the square-root Pauli $\sqrt{X}^\dagger$ gate operated on qubit $t$, $\sqrt{X}^\dagger (a_0 \ket{0} + a_1 \ket{1}) = \biggl( \frac{1 - \mathrm{i}}{2} a_0 + \frac{1 + \mathrm{i}}{2} a_1 \biggr) \ket{0} + \biggl( \frac{1 + \mathrm{i}}{2} a_0 + \frac{1 - \mathrm{i}}{2} a_1 \biggr) \ket{1}$
* `SY t`: the square-root Pauli $\sqrt{Y}$ gate operated on qubit $t$, $\sqrt{Y} (a_0 \ket{0} + a_1 \ket{1}) = \biggl( \frac{1 + \mathrm{i}}{2} a_0 - \frac{1 + \mathrm{i}}{2} a_1 \biggr) \ket{0} + \biggl( \frac{1 + \mathrm{i}}{2} a_0 + \frac{1 + \mathrm{i}}{2} a_1 \biggr) \ket{1}$
* `SY+ t`: the square-root Pauli $\sqrt{Y}^\dagger$ gate operated on qubit $t$, $\sqrt{Y}^\dagger (a_0 \ket{0} + a_1 \ket{1}) = \biggl( \frac{1 - \mathrm{i}}{2} a_0 + \frac{1 - \mathrm{i}}{2} a_1 \biggr) \ket{0} + \biggl( -\frac{1 - \mathrm{i}}{2} a_0 + \frac{1 - \mathrm{i}}{2} a_1 \biggr) \ket{1}$
* `SZ c`: the square-root Pauli $\sqrt{Z}$ gate operated on control qubit $c$, $\sqrt{Z} (a_0 \ket{0} + a_1 \ket{1}) = a_0 \ket{0} + \mathrm{i} a_1 \ket{1}$
* `SZ+ c`: the square-root Pauli $\sqrt{Z}^\dagger$ gate operated on control qubit $c$, $\sqrt{Z}^\dagger (a_0 \ket{0} + a_1 \ket{1}) = a_0 \ket{0} - \mathrm{i} a_1 \ket{1}$
* `CCCCCSX c1 c2 c3 c4 c5 t` or `C5SX c1 c2 c3 c4 c5 t`: the controlled $\sqrt{X}$ gate. Qubits $c_1$, ..., $c_5$ are control qubits, and qubit $t$ is a target qubit. If you use three control qubits, use `CCCSX c1 c2 c3 t` or `C3SX c1 c2 c3 t` instead. Similar instructions with control qubits for `SX+`, `SY`, `SY+` are also supported.
* `CCCSZZZ c1 c2 c3 t1 t2 t3` or `C3SZ3 c1 c2 c3 t1 t2 t3`: the controlled $\sqrt{Z}$ gate. Qubits $c_1$, ..., $c_3$ are control qubits, and qubits $t_1$, ..., $t_3$ is target qubits. If you use two target qubits and two control qubits, use `CCSZZ c1 c2 t1 t2` or `C2SZ2 c1 c2 t1 t2` instead. Similar instructions with control qubits for `SZ+` are also supported.
* `S c`: the $S$ gate operated on control qubit $c$. Note that `S` is an alias of `SZ`.
* `S+ c`: the $S^\dagger$ gate operated on control qubit $c$. Note that `S+` is an alias of `SZ+`.
* `T c`: the $T$ gate operated on control qubit $c$, $T (a_0 \ket{0} + a_1 \ket{1}) = a_0 \ket{0} + \dfrac{1 + \mathrm{i}}{\sqrt{2}} a_1 \ket{1}$
* `T+ c`: the $T^\dagger$ gate operated on control qubit $c$, $T^\dagger (a_0 \ket{0} + a_1 \ket{1}) = a_0 \ket{0} + \dfrac{1 - \mathrm{i}}{\sqrt{2}} a_1 \ket{1}$
* `CCCCCS c1 c2 c3 c4 c5 c6` or `C5S c1 c2 c3 c4 c5 c6`: the controlled $S$ gate. Qubits $c_1$, ..., $c_6$ are control qubits. If you use three control qubits, use `CCS c1 c2 c3` or `C2S c1 c2 c3` instead. The $S^\dagger$, $T$, and $T^\dagger$ versions are also supported.
* `U1 c lambda`: the phase-shift gate, or $U_1(\lambda)$ operation, on control qubit $c$, $U_1(\lambda) (a_0 \ket{0} + a_1 \ket{1}) = a_0 \ket{0} + \mathrm{e}^{\mathrm{i} \lambda} a_1 \ket{1}$
* `U2 t phi lambda`: $U_2(\phi, \lambda)$ operation on qubit $t$, $U_2(\phi, \lambda) (a_0 \ket{0} + a_1 \ket{1}) = \dfrac{a_0 - \mathrm{e}^{\mathrm{i} \lambda} a_1}{\sqrt{2}} \ket{0} + \mathrm{e}^{\mathrm{i} \phi} \dfrac{a_0 + \mathrm{e}^{\mathrm{i} \lambda} a_1}{\sqrt{2}} \ket{1}$
* `U3 t theta phi lambda`: $U_3(\theta, \phi, \lambda)$ operation on qubit $t$, $U_3(\theta, \phi, \lambda) (a_0 \ket{0} + a_1 \ket{1}) = [\cos(\theta/2) a_0 - \sin(\theta/2) \mathrm{e}^{\mathrm{i} \lambda} a_1] \ket{0} + \mathrm{e}^{\mathrm{i} \phi} [\sin(\theta/2) a_0 + \cos(\theta/2) \mathrm{e}^{\mathrm{i} \lambda} a_1] \ket{1}$
* `CCCCCU1 c1 c2 c3 c4 c5 c6 lambda` or `C5U1 c1 c2 c3 c4 c5 c6 lambda`: the controlled phase-shift gate. Qubits $c_1$, ..., $c_6$ are control qubits. If you use three control qubits, use `CCU1 c1 c2 c3 lambda` or `C2U1 c1 c2 c3 lambda` instead.
* `CCCCCU2 c1 c2 c3 c4 c5 t phi lambda` or `C5U2 c1 c2 c3 c4 c5 t phi lambda`: the controlled $U_2(\phi, \lambda)$ gate. Qubits $c_1$, ..., $c_5$ are control qubits, and qubit $t$ is a target qubit. If you use three control qubits, use `CCCU2 c1 c2 c3 t phi lambda` or `C3U2 c1 c2 c3 t phi lambda` instead. The $U_3$ versions are also supported.
* `+X t`: the +X gate which rotates qubit $t$ by $-\pi/2$ about the $x$-axis, $X_+ (a_0 \ket{0} + a_1 \ket{1}) = \dfrac{a_0 + \mathrm{i} a_1}{\sqrt{2}} \ket{0} + \dfrac{\mathrm{i} a_0 + a_1}{\sqrt{2}} \ket{1}$
* `-X t`: the -X gate which rotates qubit $t$ by $+\pi/2$ about the $x$-axis, $X_- (a_0 \ket{0} + a_1 \ket{1}) = \dfrac{a_0 - \mathrm{i} a_1}{\sqrt{2}} \ket{0} + \dfrac{-\mathrm{i} a_0 + a_1}{\sqrt{2}} \ket{1}$
* `+Y t`: the +Y gate which rotates qubit $t$ by $-\pi/2$ about the $y$-axis, $Y_+ (a_0 \ket{0} + a_1 \ket{1}) = \dfrac{a_0 + a_1}{\sqrt{2}} \ket{0} + \dfrac{-a_0 + a_1}{\sqrt{2}} \ket{1}$
* `-Y t`: the -Y gate which rotates qubit $t$ by $+\pi/2$ about the $y$-axis, $Y_- (a_0 \ket{0} + a_1 \ket{1}) = \dfrac{a_0 - a_1}{\sqrt{2}} \ket{0} + \dfrac{a_0 + a_1}{\sqrt{2}} \ket{1}$
* `CCCCC+X c1 c2 c3 c4 c5 t` or `C5+X c1 c2 c3 c4 c5 t`: the controlled +X gate. Qubits $c_1$, ..., $c_5$ are control qubits, and qubit $t$ is a target qubit. If you use three control qubits, use `CCC+X c1 c2 c3 t` or `C3+X c1 c2 c3 t` instead. The -X, +Y, -Y versions are also supported.
* `R c k`: the phase-shift gate which changes the phase of control qubit $c$ by an angle $2\pi/2^k$, $R(k) (a_0 \ket{0} + a_1 \ket{1}) = a_0 \ket{0} + \mathrm{e}^{2\pi\mathrm{i}/2^k} a_1 \ket{1}$. Note that `R c -k` is an alias of `R+ c k`.
* `R+ c k`: the phase-shift gate which changes the phase of control qubit $c$ by an angle $-2\pi/2^k$, $R^\dagger(k) (a_0 \ket{0} + a_1 \ket{1}) = a_0 \ket{0} + \mathrm{e}^{-2\pi\mathrm{i}/2^k} a_1 \ket{1}$. Note that `R+ c -k` is an alias of `R c k`.
* `CCCCCR c1 c2 c3 c4 c5 c6 k` or `C5R c1 c2 c3 c4 c5 c6 k`: the controlled phase-shift gate which changes the phase by an angle $2\pi/2^k$. Note that qubits $c_1$, ..., $c_6$ are control qubits. If you use three control qubits, use `CCR c1 c2 c3 k` or `C2R c1 c2 c3 k` instead. The `R+` version is also supported.
* `U c1 c2 k`: the controlled phase-shift gate. This is an alias of `CR c1 c2 k`.
* `U+ c1 c2 k`: the controlled phase-shift gate. This is an alias of `CR+ c1 c2 k`.
* `TOFFOLI c1 c2 t1`: the TOFFOLI gate. This is an alias of `CCNOT c1 c2 t1`.
* `EX t theta`: the exponential Pauli $X$ gate $\exp(\mathrm{i} \theta X) = I \cos \theta + \mathrm{i} X \sin \theta$ operated on qubit $t$.
* `EY t theta`: the exponential Pauli $Y$ gate $\exp(\mathrm{i} \theta Y) = I \cos \theta + \mathrm{i} Y \sin \theta$ operated on qubit $t$.
* `EZ t theta`: the exponential Pauli $Z$ gate $\exp(\mathrm{i} \theta Z) = I \cos \theta + \mathrm{i} Z \sin \theta$ operated on qubit $t$.
* `EXXXXXX t1 t2 t3 t4 t5 t6 theta` or `EX6 t1 t2 t3 t4 t5 t6 theta`: the exponential Pauli $X$ gates operated on qubits $t_1$, ..., $t_6$. If you use two qubits, use `EXX t1 t2 theta` or `EX2 t1 t2 theta` instead. The Pauli $Y$ and $Z$ versions are also supported.
* `CCCEXXX c1 c2 c3 t1 t2 t3 theta` or `C3EX3 c1 c2 c3 t1 t2 t3 theta`: the controlled exponential Pauli $X$ gates. Qubits $c_1$, $c_2$, $c_3$ are control qubits, and qubits $t_1$, $t_2$, and $t_3$ are target qubits. If you use two target qubits and two control qubits, use `CCEXX c1 c2 t1 t2 theta` or `C2EX2 c1 c2 t1 t2 theta` instead. The Pauli $Y$ and $Z$ versions are also supported.
* `ESWAP t1 t2 theta`: the exponential SWAP gate $\exp(\mathrm{i} \theta P) = I \cos \theta + \mathrm{i} P \sin \theta$ operated on qubits $t_1$ and $t_2$.
* `CCCCESWAP c1 c2 c3 c4 t1 t2 theta` or `C4ESWAP c1 c2 c3 c4 t1 t2 theta`: the controlled exponential SWAP gate. Qubits $c_1$, ..., $c_4$ are control qubits, and qubits $t_1$ and $t_2$ are target qubits. If you use two control qubits, use `CCESWAP c1 c2 t1 t2 theta` or `C2ESWAP c1 c2 t1 t2 theta` instead.
* `DO MEASUREMENT`: computes and prints the expectation values of all qubits. `BEGIN MEASUREMENT` can be used but it is deprecated.
* `DO AMPLITUDES`: prints the amplitudes of the state vector.
* `GENERATE EVENTS n seed`: computes the probabilities of each of the basis states and exits. It generates $n$ events by using random number generator with the initial seed `seed` and prints out the states according to these probabilites.
* `M i`: projective measurement on qubit $i$. Its result is assigned to `:OUTCOME` AND `:OUTCOME:i`.
* `CIRCUITS n`: specifies the number of quantum circuits. This should be placed before the `QUBITS` instruction. If this `CIRCUITS` instruction is omitted, the number of circuits is assumed to be 1.
* `QUBITS n`: specifies the number of qubits. This should be placed before any insstructions except for the `CIRCUITS` instruction.
* `BIT ASSIGNMENT i j k...`: specifies the initial permutation of qubits. The number of qubits specified as arguments of this instruction must be equal to the number of qubits specified in the `QUBITS n` instruction.
* `SHORBOX nx G y`
* `CLEAR i`: projects the state of qubit $i$ to $\ket{0}$.
* `SET i`: projects the state of qubit $i$ to $\ket{1}$.
* `DEPOLARIZING CHANNEL P_X=px,P_Y=py,P_Z=pz,SEED=seed`: inserts the Pauli $X$, $Y$, and $Z$ gates with specified probabilities to all qubits. For example, the Pauli $X$ gate is inserted with probability $p_x$. The random number generator uses the `seed` value as its initial seed. If the specified `seed` is negative, the value specified in the command line option of *bra* is used as the initial seed.
* `EXIT`: measures all qubits and terminate execution.
* `BEGIN FUSION q1 q2 q3 q4`/`END FUSION`: starts/ends gate fusion[^2]. Qubits $q_1$, $q_2$, ... should be appeared gate instructions between `BEGIN FUSION` and `END FUSION`.
* `BEGIN CIRCUIT n`/`END CIRCUIT`: starts/ends description of quantum gates in the quantum circuit with specified circuit index $n$. The index $n$ should be less than the number of quantum circuits specified in the `CIRCUITS` instruction. The gates out of `BEGIN CIRCUIT`/`END CIRCUIT` are assumed to be gates in the quantum circuit $0$.
* `VAR name type [size]`: declares classical variable/array whose name is `name`, type is `type`, and size is `size`. Possible `type`'s are `INT`, `REAL`, `COMPLEX`, or `PAULISS`. If `size` is not specified, array size is assumed to be 1. Note that classical variable is just an array whose size is 1. In order to get an element of array `XS`, specify such as `XS:0`, which corresponds to `XS[0]` in C/C++. In this quantum assembly language, `A:B:C:D` is for example a valid expression, and corresponding to `A[B[C[D]]]` in C/C++.
* There are built-in constants/immutable variables; `:INT:x`, `:OUTDOME`, `:OUTCOMES:n` for `INT` type, `:REAL:x`, `:IMAG:x`, `:PI`, `:HALF_PI`, `:TWO_PI`, `:ROOT_TWO`, `:HALF_ROOT_TWO` for `REAL` type, `:COMPLEX:x`, `:I`, `:MINUS_I`, `:RESULT` for `COMPLEX` type, and `:PAULIS:x` for `PAULISS` type. Note that `PAULISS` stands for Pauli string space, whose elements are linear combinations of Pauli strings with complex scalars.
* `LET lhs op rhs`: applies an operation `op` to `lhs` and `rhs`, and assign the value to `lhs`. Possible `op`'s are `:=`, `+=`, `-=`, `*=`, and `/=`. While only classical variables can be specified to `lhs`, any variables and literals can be set to `rhs`.
* `@label`: declares label `label`. Do not insert any spaces between `@` and `label`.
* `JUMP label`/`JUMPIF label lhs op rhs`: jumps to the label `label`. In the case of `JUMPIF`, one can specify a condition to jump by `lhs op rhs`, where possible `op`'s are `==`, `\=`, `>`, `<`, `<=`, and `>=`.
* `EXPECTATION operator q1 q2 q3 q4`: calculates expectation value of `operator` for the present circuit[^3]. This `operator` should be a classical variable of type `PAULISS`, and its length of Pauli string should be equal to the number of qubits specified in this instruction (`q1 q2 q3 q4` in this example). The result of this instruction is assigned to `:RESULT` whose type is `COMPLEX`.

[^2]: The number of qubits which can be specified in `BEGIN FUSION` instruction is determined by the macro `BRA_MAX_NUM_FUSED_QUBITS`, which can be set when building *bra*. Its default value is 10.
[^3]: To be more precise, `EXPECTATION H ...` calculates $\bra{\Psi} (H \ket{\Psi})$ for given operator $H$ and state $\ket{\Psi}$. It becomes the expectation value of $H$ if $H$ is Hermitian.

