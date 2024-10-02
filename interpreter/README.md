# openqasm-interpreter

## Introduction

*openqasm-interpreter*は、量子コンピュータのシミュレーションを実行するためのOpenQASMコードのインタプリタです。

## Requirements

*openqasm-interpreter* にはC++11 準拠のコンパイラが必要です。  
以下のライブラリが必要です。  
- gmp
- mpfr
- mpc
- bison(3.6.2以降)
- flex(2.6.1以降)
- [Boost C++ library](https://www.boost.org/)
- [MPI](https://www.mpi-forum.org/)

また、*openqasm-interpreter*では、[qe-qasm](https://github.com/openqasm/qe-qasm) をOpenQASMコードのパーサとして利用します。

## How to build

富岳（計算ノード）でのビルド手順を説明します。  
本説明においては、プライベート・インスタンスを利用する手順を示します。

### パッケージの準備
Spackを用いて、パッケージを準備します。

1) spackの環境設定
```bash
$ git clone https://github.com/RIKEN-RCCS/spack.git
$ git checkout fugaku-v0.21.0
$ . ./spack/share/spack/setup-env.sh
```

2) コンパイラの設定

```bash
$ spack compilers
```
3) パッケージのインストール
```bash
$ spack install cmake
$ spack install gmp
$ spack install mpfr
$ spack install mpc@1.2.1
$ spack install bison
$ spack install flex
```

### qe-qasmのビルド
リポジトリをクローンすることで、qe-qasmを取得できます。
本説明においては、富岳上で実施した手順で説明します。

```bash
$ git clone https://github.com/openqasm/qe-qasm.git
```

qe-qasmをビルドします。
1) ビルドの依存関係をインストール  
仮想環境 qe-qasm_env を構築し、ビルド依存関係をインストールします。
```bash
$ python3 -m venv ./qe-qasm_env
$ source ./qe-qasm_env/bin/activat
(qe-qasm_env) $ pip install -r requirements-dev.txt
(qe-qasm_env) $ deactivate
```
2) パッケージのロード
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
3) 環境変数の設定
```bash
$ export LD_LIBRARY_PATH=/lib64:$LD_LIBRARY_PATH
```
4) cmake
```bash
$ cd build/
$ cmake -G "Unix Makefiles" ..
```
5) ビルド
```bash
$ make
```

以下を実施し、ASTのダンプが出力されれば、正常にビルドできています。
```bash
$ cd ./bin
$ ./QasmParser -I../../tests/include ../../tests/src/test-void-measure.qasm > ~/test-void-measure.xml
```

### openqasm-interpreterのビルド

リポジトリをクローンすることで、*openqasm-interpreter*を取得できます。

```bash
$ git clone --recursive https://github.com/jhpc-quantum/RIKEN-braket.git
```

MPIのコンパイル用コマンド（mpicxx）は計算ノードのみで使用可能でクロスコンパイルはできないため、ビルドは会話型ジョブ実行で行います。
1) パッケージのロード
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
2) 環境変数の設定
```bash
$ export LD_LIBRARY_PATH=./qe-qasm/build/lib:$LD_LIBRARY_PATH 
```
3) ビルド
```bash
$ cd ./RIKEN-braket/interpreter/
$ make
```

## Usage
富岳（計算ノード）での実行手順を説明します。  

1) spackの環境設定
```bash
$ . ./spack/share/spack/setup-env.sh
```

2) パッケージのロード
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
3) 環境変数の設定
```bash
$ export LD_LIBRARY_PATH=./qe-qasm/build/lib:$LD_LIBRARY_PATH 
```
4) 実行
```bash
$ cd ./RIKEN-braket/interpreter/
$ mpiexec -n 1 ./qasminterpreter -I ../../qe-qasm/tests/include ./sample/test_h_cx.qasm
```

