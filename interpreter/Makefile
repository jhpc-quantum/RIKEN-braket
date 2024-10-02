CC = mpicxx

CFLAGS = -fopenmp -DKET_USE_OPENMP -DKET_PRINT_LOG -I../ket/include/ -I../yampi/include/ -I../../qe-qasm/include -O3 -std=c++17
LIBS = -fopenmp -lm -L../../qe-qasm/build/lib -lqasmAST -lqasmDIAG -lqasmFrontend -lqasmParser
LIBFLAGS = -Wl,--rpath=../../qe-qasm/build/lib

.PHONY : all clean build

all: build

build: qasminterpreter

qasminterpreter: qasminterpreter.o BaseQASM3Visitor.o IRGenQASM3Visitor.o  qipKet.o
	$(CC) qasminterpreter.o BaseQASM3Visitor.o IRGenQASM3Visitor.o qipKet.o $(LIBS) $(LIBFLAGS) -o qasminterpreter

qasminterpreter.o: src/qasminterpreter.cpp
	$(CC) $(CFLAGS) -c src/qasminterpreter.cpp 

BaseQASM3Visitor.o: src/BaseQASM3Visitor.cpp
	$(CC) $(CFLAGS) -c src/BaseQASM3Visitor.cpp

IRGenQASM3Visitor.o: src/IRGenQASM3Visitor.cpp
	$(CC) $(CFLAGS) -c src/IRGenQASM3Visitor.cpp

qipKet.o: src/qipKet.cpp
	$(CC) $(CFLAGS) -c src/qipKet.cpp

clean:
	rm *.o qasminterpreter
