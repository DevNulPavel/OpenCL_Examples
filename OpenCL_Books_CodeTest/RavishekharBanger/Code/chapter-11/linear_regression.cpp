#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <ocl_macros.h>
#define NUM_OF_POINTS 1024

//OpenCL kernel which is run for every work item created.
const char *linear_regression_kernel =
"#define DATA_TYPE float                                                                            \n"
"#define SUM_STEP(LENGTH, INDEX, _W)                                                              \\\n"
"    if ((INDEX < _W) && ((INDEX + _W) < LENGTH)) {                                               \\\n"
"      localSumX[INDEX] = localSumX[INDEX] + localSumX[INDEX + _W];                               \\\n"
"      localSumY[INDEX] = localSumY[INDEX] + localSumY[INDEX + _W];                               \\\n"
"      localSumXX[INDEX] = localSumXX[INDEX] + localSumXX[INDEX + _W];                            \\\n"
"      localSumXY[INDEX] = localSumXY[INDEX] + localSumXY[INDEX + _W];                            \\\n"
"    }                                                                                            \\\n"
"    barrier(CLK_LOCAL_MEM_FENCE);                                                                  \n"
"                                                                                                   \n"
"__kernel                                                                                           \n"
"void linear_regression_kernel(                                                                     \n"
"                  __global DATA_TYPE *X,                                                           \n"
"                  __global DATA_TYPE *Y,                                                           \n"
"                  __global DATA_TYPE *sumX,                                                        \n"
"                  __global DATA_TYPE *sumY,                                                        \n"
"                  __global DATA_TYPE *sumXX,                                                       \n"
"                  __global DATA_TYPE *sumXY,                                                       \n"
"                  __local  DATA_TYPE *localSumX,                                                   \n"
"                  __local  DATA_TYPE *localSumY,                                                   \n"
"                  __local  DATA_TYPE *localSumXX,                                                  \n"
"                  __local  DATA_TYPE *localSumXY,                                                  \n"
"                           int        length )                                                     \n"
"{                                                                                                  \n"
"    //Get the index of the work-item                                                               \n"
"    int index = get_global_id(0);                                                                  \n"
"    int gx = get_global_id (0);                                                                    \n"
"    int gloId = gx;                                                                                \n"
"                                                                                                   \n"
"    //  Initialize the accumulator private variable with data from the input array                 \n"
"    //  This essentially unrolls the loop below at least once                                      \n"
"    DATA_TYPE accumulatorX;                                                                        \n"
"    DATA_TYPE accumulatorY;                                                                        \n"
"    if(gloId < length){                                                                            \n"
"       accumulatorX = X[gx];                                                                       \n"
"       accumulatorY = Y[gx];                                                                       \n"
"    }                                                                                              \n"
"                                                                                                   \n"
"                                                                                                   \n"
"    //  Initialize local data store                                                                \n"
"    int local_index = get_local_id(0);                                                             \n"
"    localSumX[local_index] = accumulatorX;                                                            \n"
"    localSumY[local_index] = accumulatorY;                                                            \n"
"    localSumXY[local_index] = accumulatorX*accumulatorY;                                                            \n"
"    localSumXX[local_index] = accumulatorX*accumulatorX;                                                            \n"
"    barrier(CLK_LOCAL_MEM_FENCE);                                                                  \n"
"                                                                                                   \n"
"    //  Tail stops the last workgroup from reading past the end of the input vector                \n"
"    uint tail = length - (get_group_id(0) * get_local_size(0));                                    \n"
"                                                                                                   \n"
"    // Parallel reduction within a given workgroup using local data store                          \n"
"    // to share values between workitems                                                           \n"
"    SUM_STEP(tail, local_index, 32);                                                               \n"
"    SUM_STEP(tail, local_index, 16);                                                               \n"
"    SUM_STEP(tail, local_index,  8);                                                               \n"
"    SUM_STEP(tail, local_index,  4);                                                               \n"
"    SUM_STEP(tail, local_index,  2);                                                               \n"
"    SUM_STEP(tail, local_index,  1);                                                               \n"
"                                                                                                   \n"
"     //  Abort threads that are passed the end of the input vector                                 \n"
"    if( gloId >= length )                                                                          \n"
"        return;                                                                                    \n"
"                                                                                                   \n"
"    //  Write only the single reduced value for the entire workgroup                               \n"
"    if (local_index == 0) {                                                                        \n"
"        sumX[get_group_id(0)] = localSumX[0];                                                      \n"
"        sumY[get_group_id(0)] = localSumY[0];                                                      \n"
"        sumXX[get_group_id(0)] = localSumXX[0];                                                      \n"
"        sumXY[get_group_id(0)] = localSumXY[0];                                                      \n"
"    }                                                                                              \n"
"}                                                                                                  \n";                                


#define WORK_GROUP_SIZE 64

int main(void) {
    int i;
    // Allocate space for vectors of 
    //  x Axis, y Axis 
    cl_int clStatus;
    float *pX = (float*)malloc(sizeof(float)*NUM_OF_POINTS);
    float *pY = (float*)malloc(sizeof(float)*NUM_OF_POINTS);
    float A0 = 2.0f;
    float A1 = 0.5f;
    for (i = 0; i < NUM_OF_POINTS; i++)
    {
        pX[i] = (float)i;
        pY[i] = (float)(i - rand()%5);
        //std::cout << "pX,pY = " << pX[i] << "," << pY[i] << "\n";
    }

    // Get platform and device information
    cl_platform_id * platforms = NULL;
    //Set up the Platform
    OCL_CREATE_PLATFORMS( platforms );
    
    //Get the devices list and choose the type of device you want to run on
    cl_device_id     *device_list = NULL;
    OCL_CREATE_DEVICE( platforms[0], CL_DEVICE_TYPE_GPU, device_list);

    // Create one OpenCL context for each device in the platform
    cl_context context;
    cl_context_properties props[3] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platforms,
        0
    };

    context = clCreateContext( NULL, num_devices, device_list, NULL, NULL, &clStatus);

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_list[0], 0, &clStatus);
    // Execute the OpenCL kernel on the list
    size_t global_size = NUM_OF_POINTS; // Process the entire lists
    size_t local_size = WORK_GROUP_SIZE;           // Process one item at a time
    size_t num_of_work_groups = global_size/local_size;
    int    points=NUM_OF_POINTS;
    //Allocate memory for storing the sumations
    float *psumX = (float*)malloc(sizeof(float)*num_of_work_groups);
    float *psumY = (float*)malloc(sizeof(float)*num_of_work_groups);
    float *psumXY = (float*)malloc(sizeof(float)*num_of_work_groups);
    float *psumXX = (float*)malloc(sizeof(float)*num_of_work_groups);

    // Create memory buffers on the device for each vector
    cl_mem pX_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY,
            NUM_OF_POINTS * sizeof(float), NULL, &clStatus);
    cl_mem pY_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY,
            NUM_OF_POINTS * sizeof(float), NULL, &clStatus);
    cl_mem psumX_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY,
            num_of_work_groups * sizeof(float), NULL, &clStatus);
    cl_mem psumY_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY,
            num_of_work_groups * sizeof(float), NULL, &clStatus);
    cl_mem psumXX_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY,
            num_of_work_groups * sizeof(float), NULL, &clStatus);
    cl_mem psumXY_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY,
            num_of_work_groups * sizeof(float), NULL, &clStatus);

    // Copy the Buffer A and B to the device
    clStatus = clEnqueueWriteBuffer(command_queue, pX_clmem, CL_TRUE, 0,
            NUM_OF_POINTS * sizeof(float), pX, 0, NULL, NULL);
    clStatus = clEnqueueWriteBuffer(command_queue, pY_clmem, CL_TRUE, 0,
            NUM_OF_POINTS * sizeof(float), pY, 0, NULL, NULL);

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1,
            (const char **)&linear_regression_kernel, NULL, &clStatus);

    // Build the program
    clStatus = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);
    if(clStatus != CL_SUCCESS)
        LOG_OCL_COMPILER_ERROR(program, device_list[0]);

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "linear_regression_kernel", &clStatus);

    // Set the arguments of the kernel
    clStatus = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&pX_clmem);
    clStatus |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&pY_clmem);
    clStatus |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&psumX_clmem);
    clStatus |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&psumY_clmem);
    clStatus |= clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&psumXX_clmem);
    clStatus |= clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&psumXY_clmem);
    clStatus |= clSetKernelArg(kernel, 6, WORK_GROUP_SIZE*sizeof(float), NULL);
    clStatus |= clSetKernelArg(kernel, 7, WORK_GROUP_SIZE*sizeof(float), NULL);
    clStatus |= clSetKernelArg(kernel, 8, WORK_GROUP_SIZE*sizeof(float), NULL);
    clStatus |= clSetKernelArg(kernel, 9, WORK_GROUP_SIZE*sizeof(float), NULL);
    clStatus |= clSetKernelArg(kernel, 10, sizeof(int), &points);
    LOG_OCL_ERROR(clStatus, "Kernel Arguments setting failed." );
    
    clStatus = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
            &global_size, &local_size, 0, NULL, NULL);
    LOG_OCL_ERROR(clStatus, "NDRange Failed." );

    // Read the device memory buffer which stores the sums of each work group
    clStatus = clEnqueueReadBuffer(command_queue, psumX_clmem, CL_TRUE, 0,
            num_of_work_groups * sizeof(float), psumX, 0, NULL, NULL);
    clStatus = clEnqueueReadBuffer(command_queue, psumY_clmem, CL_TRUE, 0,
            num_of_work_groups * sizeof(float), psumY, 0, NULL, NULL);
    clStatus = clEnqueueReadBuffer(command_queue, psumXX_clmem, CL_TRUE, 0,
            num_of_work_groups * sizeof(float), psumXX, 0, NULL, NULL);
    clStatus = clEnqueueReadBuffer(command_queue, psumXY_clmem, CL_TRUE, 0,
            num_of_work_groups * sizeof(float), psumXY, 0, NULL, NULL);

    float sumX  = 0.0f;
    float sumY  = 0.0f;
    float sumXX = 0.0f;
    float sumXY = 0.0f;
    for(int i=0;i<num_of_work_groups;i++)
    {
        sumX  += psumX[i];
        sumY  += psumY[i];
        sumXY += psumXY[i];
        sumXX += psumXX[i];
    }
    std::cout << "SUM_X = " << sumX << "\n"; 
    std::cout << "SUM_Y = " << sumY << "\n";
    std::cout << "SUM_XX = " << sumXX << "\n";
    std::cout << "SUM_XY = " << sumXY << "\n";
    
    // Clean up and wait for all the comands to complete.
    clStatus = clFlush(command_queue);
    clStatus = clFinish(command_queue);
    A0 = (sumY*sumXX - sumX*sumXY)/(NUM_OF_POINTS*sumXX - sumX*sumX );
    A1 = (NUM_OF_POINTS*sumXY - sumX*sumY)/(NUM_OF_POINTS*sumXX - sumX*sumX );
    std::cout << "A0 = " << A0 << "\n";
    std::cout << "A1 = " << A1 << "\n";
    
    // Display the result to the screen
    /*for(i = 0; i < VECTOR_SIZE; i++)
        printf("%f * %f + %f = %f\n", alpha, A[i], B[i], C[i]);*/

    // Finally release all OpenCL allocated objects and host buffers.
    clStatus = clReleaseKernel(kernel);
    clStatus = clReleaseProgram(program);
    clStatus = clReleaseMemObject(pX_clmem);
    clStatus = clReleaseMemObject(pY_clmem);
    clStatus = clReleaseMemObject(psumX_clmem);
    clStatus = clReleaseMemObject(psumY_clmem);
    clStatus = clReleaseMemObject(psumXX_clmem);
    clStatus = clReleaseMemObject(psumXY_clmem);
    clStatus = clReleaseCommandQueue(command_queue);
    clStatus = clReleaseContext(context);
    /*Free all Memory Allocations*/
    free(pX);
    free(pY);
    free(psumX);
    free(psumXX);
    free(psumY);
    free(psumXY);

    OCL_RELEASE_PLATFORMS( platforms );   
    OCL_RELEASE_DEVICES( device_list );    

    return 0;
}
