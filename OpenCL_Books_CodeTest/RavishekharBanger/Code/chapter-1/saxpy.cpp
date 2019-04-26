//If you want to build the file directly at the command prompt then use the following commands. 
//AMD commands
//cl /c saxpy.cpp /I"%AMDAPPSDKROOT%\include"
//link  /OUT:"saxpy.exe" "%AMDAPPSDKROOT%\lib\x86_64\OpenCL.lib" saxpy.obj

//nVIDIA commands
//cl /c saxpy.cpp /I"%NVSDKCOMPUTE_ROOT%\OpenCL\common\inc"
//link  /OUT:"saxpy.exe" "%NVSDKCOMPUTE_ROOT%\OpenCL\common\lib\x64\OpenCL.lib" saxpy.obj

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include <ocl_macros.h>

//Common defines 
#define VENDOR_NAME "AMD"
#define DEVICE_TYPE CL_DEVICE_TYPE_GPU

#define VECTOR_SIZE 1024

//OpenCL kernel which is run for every work item created.
//The below const char string is compiled by the runtime complier
//when a program object is created with clCreateProgramWithSource 
//and built with clBuildProgram.
const char *saxpy_kernel =
"__kernel                                   \n"
"void saxpy_kernel(const float alpha,       \n"
"                  __global float *A,       \n"
"                  __global float *B,       \n"
"                  __global float *C)       \n"
"{                                          \n"
"    //Get the index of the work-item       \n"
"    int index = get_global_id(0);          \n"
"    C[index] = alpha* A[index] + B[index]; \n"
"}                                          \n";

int main(void) {

    cl_int clStatus; //Keeps track of the error values returned. 

    // Get platform and device information
    cl_platform_id * platforms = NULL;

    // Set up the Platform. Take a look at the MACROs used in this file. 
    // These are defined in common/ocl_macros.h
    OCL_CREATE_PLATFORMS( platforms );

    // Get the devices list and choose the type of device you want to run on
    cl_device_id *device_list = NULL;
    OCL_CREATE_DEVICE( platforms[0], DEVICE_TYPE, device_list);

    // Create OpenCL context for devices in device_list
    cl_context context;
    cl_context_properties props[3] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platforms[0],
        0
    };
    // An OpenCL context can be associated to multiple devices, either CPU or GPU
    // based on the value of DEVICE_TYPE defined above.
    context = clCreateContext( NULL, num_devices, device_list, NULL, NULL, &clStatus);
    LOG_OCL_ERROR(clStatus, "clCreateContext Failed..." );

    // Create a command queue for the first device in device_list
    cl_command_queue command_queue = clCreateCommandQueue(context, device_list[0], 0, &clStatus);
    LOG_OCL_ERROR(clStatus, "clCreateCommandQueue Failed..." );

    float alpha = 2.0;
    // Allocate space for vectors A, B and C    
    float *A = (float*)malloc(sizeof(float)*VECTOR_SIZE);
    float *B = (float*)malloc(sizeof(float)*VECTOR_SIZE);
    float *C = (float*)malloc(sizeof(float)*VECTOR_SIZE);
    for(int i = 0; i < VECTOR_SIZE; i++)
    {
        A[i] = (float)i;
        B[i] = (float)(VECTOR_SIZE - i);
        C[i] = 0;
    }

    // Create memory buffers on the device for each vector
    cl_mem A_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY,
            VECTOR_SIZE * sizeof(float), NULL, &clStatus);
    cl_mem B_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY,
            VECTOR_SIZE * sizeof(float), NULL, &clStatus);
    cl_mem C_clmem = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
            VECTOR_SIZE * sizeof(float), NULL, &clStatus);

    // Copy the Buffer A and B to the device. We do a blocking write to the device buffer.
    clStatus = clEnqueueWriteBuffer(command_queue, A_clmem, CL_TRUE, 0,
            VECTOR_SIZE * sizeof(float), A, 0, NULL, NULL);
    LOG_OCL_ERROR(clStatus, "clEnqueueWriteBuffer Failed..." );
    clStatus = clEnqueueWriteBuffer(command_queue, B_clmem, CL_TRUE, 0,
            VECTOR_SIZE * sizeof(float), B, 0, NULL, NULL);
    LOG_OCL_ERROR(clStatus, "clEnqueueWriteBuffer Failed..." );

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1,
            (const char **)&saxpy_kernel, NULL, &clStatus);
    LOG_OCL_ERROR(clStatus, "clCreateProgramWithSource Failed..." );

    // Build the program
    clStatus = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);
    if(clStatus != CL_SUCCESS)
        LOG_OCL_COMPILER_ERROR(program, device_list[0]);

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "saxpy_kernel", &clStatus);

    // Set the arguments of the kernel. Take a look at the kernel definition in saxpy_kernel 
    // variable. First parameter is a constant and the other three are buffers.
    clStatus = clSetKernelArg(kernel, 0, sizeof(float),  (void *)&alpha);
    clStatus |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&A_clmem);
    clStatus |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&B_clmem);
    clStatus |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&C_clmem);
    LOG_OCL_ERROR(clStatus, "clSetKernelArg Failed..." );

    // Execute the OpenCL kernel on the list
    size_t global_size = VECTOR_SIZE; // Process one vector element in each work item
    size_t local_size = 64;           // Process in work groups of size 64.
    cl_event saxpy_event;
    clStatus = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
            &global_size, &local_size, 0, NULL, &saxpy_event);
    LOG_OCL_ERROR(clStatus, "clEnqueueNDRangeKernel Failed..." );

    // Read the memory buffer C_clmem on the device to the host allocated buffer C
    // This task is invoked only after the completion of the event saxpy_event
    clStatus = clEnqueueReadBuffer(command_queue, C_clmem, CL_TRUE, 0,
            VECTOR_SIZE * sizeof(float), C, 1, &saxpy_event, NULL);
    LOG_OCL_ERROR(clStatus, "clEnqueueReadBuffer Failed..." );

    // Clean up and wait for all the comands to complete.
    clStatus = clFinish(command_queue);

    // Display the result to the screen
    for(int i = 0; i < VECTOR_SIZE; i++)
        printf("%f * %f + %f = %f\n", alpha, A[i], B[i], C[i]);

    // Finally release all OpenCL objects and release the host buffers.
    clStatus = clReleaseKernel(kernel);
    clStatus = clReleaseProgram(program);
    clStatus = clReleaseMemObject(A_clmem);
    clStatus = clReleaseMemObject(B_clmem);
    clStatus = clReleaseMemObject(C_clmem);
    clStatus = clReleaseCommandQueue(command_queue);
    clStatus = clReleaseContext(context);
    free(A);
    free(B);
    free(C);
    free(platforms);
    free(device_list);

    return 0;
}