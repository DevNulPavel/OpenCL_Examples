# Custom compiler
# CC=gcc

# Input output
OUT_GPU=OpenCL_Test

# Library Linux
# OPEN_CL_LIBRARY="-l OpenCL"

# Library OSX
COMPILE_FLAGS=-O2 -std=gnu++14 -stdlib=libc++ -framework opencl


all: MathCode.o main.o
	$(CXX) MathCode.o main.o $(COMPILE_FLAGS) -o $(OUT_GPU)

MathCode.o:
	$(CXX) -c MathCode.cpp $(COMPILE_FLAGS)

main.o:
	$(CXX) -c main.cpp $(COMPILE_FLAGS)

clean:
	rm -f *.o $(OUT_GPU)

