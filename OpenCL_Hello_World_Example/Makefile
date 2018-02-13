# Custom compiler
# CC=gcc

# Input output
SRC_GPU=hello.c
OUT_GPU=gpu

# Library Linux
OPEN_CL_LIBRARY="-l OpenCL"

# Library OSX
# OPEN_CL_LIBRARY=-framework opencl


all:
	$(CC) $(SRC_GPU) -O2 -o $(OUT_GPU) $(OPEN_CL_LIBRARY)

clean:
	rm -f $(OUT_GPU) $(OUT_CPU)

