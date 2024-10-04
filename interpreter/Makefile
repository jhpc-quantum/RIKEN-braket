CC = mpicxx

CFLAGS = -fopenmp -DKET_USE_OPENMP -DKET_PRINT_LOG -I../ket/include/ -I../yampi/include/ -I../../qe-qasm/include -O3 -std=c++17
LIBS = -fopenmp -lm -L../../qe-qasm/build/lib -lqasmAST -lqasmDIAG -lqasmFrontend -lqasmParser
LIBFLAGS = -Wl,--rpath=../../qe-qasm/build/lib

.PHONY : all clean
VPATH=./build

OBJDIR=./build

all: $(OBJDIR) qasminterpreter

$(OBJDIR): 
	mkdir -p $(OBJDIR)

qasminterpreter: qasminterpreter.o BaseQASM3Visitor.o IRGenQASM3Visitor.o  qipKet.o
	$(CC) $(OBJDIR)/qasminterpreter.o $(OBJDIR)/BaseQASM3Visitor.o $(OBJDIR)/IRGenQASM3Visitor.o $(OBJDIR)/qipKet.o $(LIBS) $(LIBFLAGS) -o $(OBJDIR)/qasminterpreter

qasminterpreter.o: src/qasminterpreter.cpp
	$(CC) $(CFLAGS) -c src/qasminterpreter.cpp  -o $(OBJDIR)/qasminterpreter.o

BaseQASM3Visitor.o: src/BaseQASM3Visitor.cpp
	$(CC) $(CFLAGS) -c src/BaseQASM3Visitor.cpp -o $(OBJDIR)/BaseQASM3Visitor.o

IRGenQASM3Visitor.o: src/IRGenQASM3Visitor.cpp
	$(CC) $(CFLAGS) -c src/IRGenQASM3Visitor.cpp -o $(OBJDIR)/IRGenQASM3Visitor.o

qipKet.o: src/qipKet.cpp
	$(CC) $(CFLAGS) -c src/qipKet.cpp -o $(OBJDIR)/qipKet.o

clean:
	rm $(OBJDIR)/*.o $(OBJDIR)/qasminterpreter
