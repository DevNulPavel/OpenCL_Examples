# Custom compiler
# CC=gcc

# Input output
SRC_GPU=hello.c
OUT_GPU=gpu

# Library Linux
OPEN_CL_LIBRARY=-L/usr/lib/x86_64-linux-gnu/ -lOpenCL -lm

# Library OSX
# OPEN_CL_LIBRARY=-framework opencl


all:
	$(CC) $(SRC_GPU) -O2 $(OPEN_CL_LIBRARY) -o $(OUT_GPU)

clean:
	rm -f $(OUT_GPU)

