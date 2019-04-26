#include <stdio.h>
#include <stdlib.h>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include <ocl_macros.h>

#define VECTOR_SIZE 1024

double get_event_exec_time(cl_event event)
{
    cl_ulong start_time, end_time;
    /*Get start device counter for the event*/
    clGetEventProfilingInfo (event,
                    CL_PROFILING_COMMAND_START,
                    sizeof(cl_ulong),
                    &start_time,
                    NULL);
    /*Get end device counter for the event*/
    clGetEventProfilingInfo (event,
                    CL_PROFILING_COMMAND_END,
                    sizeof(cl_ulong),
                    &end_time,
                    NULL);
    /*Convert the counter values to milli seconds*/
    double total_time = (end_time - start_time) * 1e-6;
    return total_time;
}

//OpenCL kernel which is run for every work item created.
const char *saxpy_kernel =
"__kernel                                   \n"
"void saxpy_kernel(float alpha,     \n"
"                  __global float *A,       \n"
"                  __global float *B,       \n"
"                  __global float *C)       \n"
"{                                          \n"
"    //Get the index of the work-item       \n"
"    int index = get_global_id(0);          \n"
"    C[index] = alpha* A[index] + B[index]; \n"
"}                                          \n";

int main(void) {
    int i;
    // Allocate space for vectors A, B and C
    float alpha = 2.0;
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
    LOG_OCL_ERROR(clStatus, "clGetPlatformIDs Failed" );

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
    cl_context_properties props[3] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platforms,
        0
    };
    context = clCreateContext( NULL, num_devices, device_list, NULL, NULL, &clStatus);
    LOG_OCL_ERROR(clStatus, "clCreateContext Failed" );

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_list[0], 
                                                          CL_QUEUE_PROFILING_ENABLE, &clStatus);
    LOG_OCL_ERROR(clStatus, "clCreateCommandQueue Failed" );

    // Create memory buffers on the device for each vector
    cl_mem A_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY,
            VECTOR_SIZE * sizeof(float), NULL, &clStatus);
    cl_mem B_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY,
            VECTOR_SIZE * sizeof(float), NULL, &clStatus);
    cl_mem C_clmem = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
            VECTOR_SIZE * sizeof(float), NULL, &clStatus);

    // Copy the Buffer A and B to the device
    cl_event write_event[2];
    clStatus = clEnqueueWriteBuffer(command_queue, A_clmem, CL_TRUE, 0,
            VECTOR_SIZE * sizeof(float), A, 0, NULL, &write_event[0]);
    LOG_OCL_ERROR(clStatus, "clEnqueueWriteBuffer Failed" );
    clStatus = clEnqueueWriteBuffer(command_queue, B_clmem, CL_TRUE, 0,
            VECTOR_SIZE * sizeof(float), B, 0, NULL, &write_event[1]);
    LOG_OCL_ERROR(clStatus, "clEnqueueWriteBuffer Failed" );

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1,
            (const char **)&saxpy_kernel, NULL, &clStatus);
    LOG_OCL_ERROR(clStatus, "clCreateProgramWithSource Failed" );

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
    
    cl_event kernel_event;
    clStatus = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
            &global_size, &local_size, 2, write_event, &kernel_event);
    LOG_OCL_ERROR(clStatus, "clEnqueueNDRangeKernel Failed" );

    // Read the memory buffer C_clmem on the device to the local variable C
    cl_event read_event;
    clStatus = clEnqueueReadBuffer(command_queue, C_clmem, CL_TRUE, 0,
            VECTOR_SIZE * sizeof(float), C, 1, &kernel_event, &read_event);
    LOG_OCL_ERROR(clStatus, "clEnqueueReadBuffer Failed" );

    /*Get all the event statistics and display the timings*/
    double exec_time;
    exec_time = get_event_exec_time(write_event[0]);
    printf("Time taken to transfer Matrix A = %lf ms\n",exec_time);
    exec_time = get_event_exec_time(write_event[1]);
    printf("Time taken to transfer Matrix B = %lf ms\n",exec_time);
    exec_time = get_event_exec_time(kernel_event);
    printf("Time taken to execute the SAXPY kernel = %lf ms\n",exec_time);
    exec_time = get_event_exec_time(read_event);
    printf("Time taken to read the result Matrix C = %lf ms\n",exec_time);

    // Clean up and wait for all the comands to complete.
    clStatus = clFinish(command_queue);
    LOG_OCL_ERROR(clStatus, "clFinish Failed" );

    // Display the result on stdout 
    //for(i = 0; i < VECTOR_SIZE; i++)
    //    printf("%f * %f + %f = %f\n", alpha, A[i], B[i], C[i]);

    // Finally release all OpenCL allocated objects and host buffers.
    clStatus = clReleaseKernel(kernel);
    clStatus |= clReleaseProgram(program);
    clStatus |= clReleaseMemObject(A_clmem);
    clStatus |= clReleaseMemObject(B_clmem);
    clStatus |= clReleaseMemObject(C_clmem);
    clStatus |= clReleaseCommandQueue(command_queue);
    clStatus |= clReleaseContext(context);
    LOG_OCL_ERROR(clStatus, "OpenCL release Failed" );

    free(A);
    free(B);
    free(C);
    free(platforms);
    free(device_list);

    return 0;
}