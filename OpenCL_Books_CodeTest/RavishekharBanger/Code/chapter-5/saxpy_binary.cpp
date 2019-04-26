
#include <stdio.h>
#include <stdlib.h>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include <ocl_macros.h>

#define VECTOR_SIZE 1024

int main(void) {
    int i;
    FILE *fp;
    float alpha = 2.0;
    // Allocate space for vectors A, B and C
    float *A = (float*)malloc(sizeof(float)*VECTOR_SIZE);
    float *B = (float*)malloc(sizeof(float)*VECTOR_SIZE);
    float *C = (float*)malloc(sizeof(float)*VECTOR_SIZE);
    
    for(i = 0; i < VECTOR_SIZE; i++)
    {
        A[i] = (float)i;
        B[i] = (float)(VECTOR_SIZE - i);
        C[i] = 0;
    }
    
    // Get platform and device information
    cl_platform_id * platforms = NULL;
    cl_uint     num_platforms;
    //Set up the Platform
    cl_int clStatus = clGetPlatformIDs(0, NULL, &num_platforms);
    LOG_OCL_ERROR(clStatus, "clGetPlatformIDs Failed while getting number of platforms available" );

    platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id)*num_platforms);
    clStatus = clGetPlatformIDs(num_platforms, platforms, NULL);
    LOG_OCL_ERROR(clStatus, "clGetPlatformIDs Failed" );

    //Get the devices list and choose the type of device you want to run on
    cl_device_id     *device_list = NULL;
    cl_uint       num_devices;

    clStatus = clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_GPU, 0,
            NULL, &num_devices);
    LOG_OCL_ERROR(clStatus, "clGetDeviceIDs Failed" );

    device_list = (cl_device_id *)malloc(sizeof(cl_device_id)*num_devices);
    clStatus = clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_GPU, num_devices,
            device_list, NULL);
    LOG_OCL_ERROR(clStatus, "clGetDeviceIDs Failed" );

    // Create one OpenCL context for each device in the platform
    cl_context context;
    context = clCreateContext( NULL, num_devices, 
                               device_list, NULL, 
                               NULL, &clStatus);
    LOG_OCL_ERROR(clStatus, "clCreateContext Failed" );

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, 
                                        device_list[0], 0, &clStatus);
    LOG_OCL_ERROR(clStatus, "clCreateCommandQueue Failed" );

    // Create memory buffers on the device for each vector
    cl_mem A_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY,
            VECTOR_SIZE * sizeof(float), NULL, &clStatus);
    cl_mem B_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY,
            VECTOR_SIZE * sizeof(float), NULL, &clStatus);
    cl_mem C_clmem = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
            VECTOR_SIZE * sizeof(float), NULL, &clStatus);

    // Copy the Buffer A and B to the device
    clStatus = clEnqueueWriteBuffer(command_queue, A_clmem, CL_TRUE, 0,
            VECTOR_SIZE * sizeof(float), A, 0, NULL, NULL);
    LOG_OCL_ERROR(clStatus, "clEnqueueWriteBuffer of A_clmem Failed" );
    clStatus = clEnqueueWriteBuffer(command_queue, B_clmem, CL_TRUE, 0,
            VECTOR_SIZE * sizeof(float), B, 0, NULL, NULL);
    LOG_OCL_ERROR(clStatus, "clEnqueueWriteBuffer of B_clmem Failed" );

    //Open the file for reading
    fp = fopen("saxpy_kernel_binary_gpu.clbin", "rb");
    fseek(fp,0L,SEEK_END);
    size_t fileSize = ftell(fp);
    rewind(fp);
    unsigned char * saxpy_kernel = new unsigned char [fileSize];
    fread(saxpy_kernel,fileSize,1,fp);
    // Create a program from the kernel source
    cl_int binary_status;
    cl_program program = clCreateProgramWithBinary(context, 1, 
                         &device_list[0], &fileSize,
                         (const unsigned char **)&saxpy_kernel, 
                         &binary_status, &clStatus);
    LOG_OCL_ERROR(clStatus, "clCreateProgramWithBinary Failed" );

    // Build the program
    clStatus = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);
    LOG_OCL_ERROR(clStatus, "clBuildProgram Failed" );

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "saxpy_kernel", &clStatus);
    LOG_OCL_ERROR(clStatus, "clCreateKernel Failed" );

    // Set the arguments of the kernel
    clStatus = clSetKernelArg(kernel, 0, sizeof(float), (void *)&alpha);
    clStatus |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&A_clmem);
    clStatus |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&B_clmem);
    clStatus |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&C_clmem);
    LOG_OCL_ERROR(clStatus, "clSetKernelArg Failed" );

    // Execute the OpenCL kernel on the list
    size_t global_size = VECTOR_SIZE; // Process the entire lists
    size_t local_size = 64;           // Process one item at a time
    clStatus = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
            &global_size, &local_size, 0, NULL, NULL);
    LOG_OCL_ERROR(clStatus, "clEnqueueNDRangeKernel Failed" );

    // Read the memory buffer C_clmem on the device to the local variable C
    clStatus = clEnqueueReadBuffer(command_queue, C_clmem, CL_TRUE, 0,
            VECTOR_SIZE * sizeof(float), C, 0, NULL, NULL);
    LOG_OCL_ERROR(clStatus, "clEnqueueReadBuffer Failed" );

    // Clean up and wait for all the comands to complete.
    clStatus = clFinish(command_queue);
    LOG_OCL_ERROR(clStatus, "clFinish Failed" );

    // Display the result to the screen
    for(i = 0; i < VECTOR_SIZE; i++)
        printf("%f * %f + %f = %f\n", alpha, A[i], B[i], C[i]);

    // Finally release all OpenCL allocated objects and host buffers.
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