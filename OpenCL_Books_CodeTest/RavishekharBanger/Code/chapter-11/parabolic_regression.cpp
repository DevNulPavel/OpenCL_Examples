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

#define NUM_OF_POINTS 4096
#define WORK_GROUP_SIZE 64

//OpenCL kernel which is run for every work item created.
const char *parabolic_regression_kernel =
"#define DATA_TYPE float                                                                            \n"
"#define SUM_STEP(LENGTH, INDEX, _W)                                                              \\\n"
"    if ((INDEX < _W) && ((INDEX + _W) < LENGTH)) {                                               \\\n"
"      localSumX[INDEX] = localSumX[INDEX] + localSumX[INDEX + _W];                               \\\n"
"      localSumY[INDEX] = localSumY[INDEX] + localSumY[INDEX + _W];                               \\\n"
"      localSumXY[INDEX] = localSumXY[INDEX] + localSumXY[INDEX + _W];                            \\\n"
"      localSumXXY[INDEX] = localSumXXY[INDEX] + localSumXXY[INDEX + _W];                         \\\n"
"      localSumXX[INDEX] = localSumXX[INDEX] + localSumXX[INDEX + _W];                            \\\n"
"      localSumXXX[INDEX] = localSumXXX[INDEX] + localSumXXX[INDEX + _W];                         \\\n"
"      localSumXXXX[INDEX] = localSumXXXX[INDEX] + localSumXXXX[INDEX + _W];                      \\\n"
"    }                                                                                            \\\n"
"    barrier(CLK_LOCAL_MEM_FENCE);                                                                  \n"
"                                                                                                   \n"
"__kernel                                                                                           \n"
"void parabolic_regression_kernel(                                                                  \n"
"                  __global DATA_TYPE *X,                                                           \n"
"                  __global DATA_TYPE *Y,                                                           \n"
"                  __global DATA_TYPE *sumX,                                                        \n"
"                  __global DATA_TYPE *sumY,                                                        \n"
"                  __global DATA_TYPE *sumXY,                                                       \n"
"                  __global DATA_TYPE *sumXXY,                                                      \n"
"                  __global DATA_TYPE *sumXX,                                                       \n"
"                  __global DATA_TYPE *sumXXX,                                                      \n"
"                  __global DATA_TYPE *sumXXXX,                                                     \n"
"                  __local  DATA_TYPE *localSumX,                                                   \n"
"                  __local  DATA_TYPE *localSumY,                                                   \n"
"                  __local  DATA_TYPE *localSumXX,                                                  \n"
"                  __local  DATA_TYPE *localSumXY,                                                  \n"
"                  __local  DATA_TYPE *localSumXXY,                                                 \n"
"                  __local  DATA_TYPE *localSumXXX,                                                 \n"
"                  __local  DATA_TYPE *localSumXXXX,                                                \n"
"                           int        length )                                                     \n"
"{                                                                                                  \n"
"    //Get the index of the work-item                                                               \n"
"    int index = get_global_id(0);                                                                  \n"
"    int gx = get_global_id (0);                                                                    \n"
"    int gloId = gx;                                                                                \n"
"    DATA_TYPE XX;                                                                                  \n"
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
"    //  Initialize local data store                                                                \n"
"    int local_index = get_local_id(0);                                                             \n"
"    localSumX[local_index] = accumulatorX;                                                         \n"
"    localSumY[local_index] = accumulatorY;                                                         \n"
"    XX = accumulatorX*accumulatorX;                                                                \n"
"    localSumXY[local_index]   = accumulatorX*accumulatorY;                                         \n"
"    localSumXXY[local_index]  = XX*accumulatorY;                                                   \n"
"    localSumXX[local_index]   = XX;                                                                \n"
"    localSumXXX[local_index]  = XX*accumulatorX;                                                   \n"
"    localSumXXXX[local_index] = XX*XX;                                                             \n"
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
"        sumX[get_group_id(0)]    = localSumX[0];                                                      \n"
"        sumY[get_group_id(0)]    = localSumY[0];                                                      \n"
"        sumXY[get_group_id(0)]   = localSumXY[0];                                                    \n"
"        sumXXY[get_group_id(0)]  = localSumXXY[0];                                                    \n"
"        sumXX[get_group_id(0)]   = localSumXX[0];                                                    \n"
"        sumXXX[get_group_id(0)]  = localSumXXX[0];                                                  \n"
"        sumXXXX[get_group_id(0)] = localSumXXXX[0];                                                \n"
"    }                                                                                              \n"
"}                                                                                                  \n";                                

float determinant3By3(
                      float a1,
                      float b1,
                      float c1,
                      float a2,
                      float b2,
                      float c2,
                      float a3,
                      float b3,
                      float c3
                      )
{
    float det = a1*b2*c3 - a1*b3*c2;
    det += a3*b1*c2 - a2*b1*c3;
    det += a2*b3*c1 - a3*b2*c1;
    return det;
}

void findParabolla( 
                   float* pA0, 
                   float* pA1, 
                   float* pA2,
                   //Input parameters
                   int    N, 
                   float sumX, 
                   float sumXX, 
                   float sumXXX, 
                   float sumXXXX, 
                   float sumY, 
                   float sumXY, 
                   float sumXXY,
                   bool* resultValid
                   )
{
    //compute detA
    float detA = determinant3By3((float)N, sumX,   sumXX,
                                  sumX,    sumXX,  sumXXX,
                                  sumXX,   sumXXX, sumXXXX);


    if( 0.f == detA)
    {
        *resultValid = false;
        return;
    }

    float detA0 = determinant3By3(sumY, sumX,   sumXX,
                                  sumXY,    sumXX,  sumXXX,
                                  sumXXY,   sumXXX, sumXXXX);
    std::cout <<"KOushik " << detA0;
    float detA1 = determinant3By3((float)N, sumY,   sumXX,
                                  sumX,    sumXY,  sumXXX,
                                  sumXX,   sumXXY, sumXXXX);

    float detA2 = determinant3By3((float)N, sumX,   sumY,
                                  sumX,    sumXX,  sumXY,
                                  sumXX,   sumXXX, sumXXY);


    *pA0 = detA0/detA;
    *pA1 = detA1/detA;
    *pA2 = detA2/detA;
}



int main(void) {
    int i;
    // Allocate space for vectors of 
    //  x Axis, y Axis 
    cl_int clStatus;
    float *pX = (float*)malloc(sizeof(float)*NUM_OF_POINTS);
    float *pY = (float*)malloc(sizeof(float)*NUM_OF_POINTS);
    float A0 = 50.0f;
    float A1 = 3.5f;
    float A2 = 0.89f;
    for (i = 0; i < NUM_OF_POINTS; i++)
    {
        pX[i] = (float)i;
        pY[i] = (float)(A2*i*i + A1*i + A0);
    }
    //Verification Code
    float vSum = 0.0f;
    for (i = 0; i < NUM_OF_POINTS; i++)
    {
        vSum += pX[i]*pX[i]*pY[i];
    }
    std::cout << "sumxxy = " << vSum << "\n";
vSum = 0.0f;
    for (i = 0; i < NUM_OF_POINTS; i++)
    {
        vSum += pY[i];
    }
    std::cout << "sumy = " << vSum << "\n";
vSum = 0.0f;
    for (i = 0; i < NUM_OF_POINTS; i++)
    {
        vSum += pX[i];
    }
    std::cout << "sumx = " << vSum << "\n";
vSum = 0.0f;
    for (i = 0; i < NUM_OF_POINTS; i++)
    {
        vSum += pX[i]*pY[i];
    }
    std::cout << "sumxy = " << vSum << "\n";
vSum = 0.0f;
    for (i = 0; i < NUM_OF_POINTS; i++)
    {
        vSum += pX[i]*pX[i];
    }
    std::cout << "sumxx = " << vSum << "\n";
vSum = 0.0f;
    for (i = 0; i < NUM_OF_POINTS; i++)
    {
        vSum += pX[i]*pX[i]*pX[i];
    }
    std::cout << "sumxxx = " << vSum << "\n";
vSum = 0.0f;
    for (i = 0; i < NUM_OF_POINTS; i++)
    {
        vSum += pX[i]*pX[i]*pX[i]*pX[i];
    }
    std::cout << "sumxxxx = " << vSum << "\n";
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
    float *psumX    = (float*)malloc(sizeof(float)*num_of_work_groups);
    float *psumY    = (float*)malloc(sizeof(float)*num_of_work_groups);
    float *psumXY   = (float*)malloc(sizeof(float)*num_of_work_groups);
    float *psumXXY  = (float*)malloc(sizeof(float)*num_of_work_groups);
    float *psumXX   = (float*)malloc(sizeof(float)*num_of_work_groups);
    float *psumXXX  = (float*)malloc(sizeof(float)*num_of_work_groups);
    float *psumXXXX = (float*)malloc(sizeof(float)*num_of_work_groups);

    // Create memory buffers on the device for each vector
    cl_mem pX_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY,
            NUM_OF_POINTS * sizeof(float), NULL, &clStatus);
    cl_mem pY_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY,
            NUM_OF_POINTS * sizeof(float), NULL, &clStatus);
    cl_mem psumX_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY,
            num_of_work_groups * sizeof(float), NULL, &clStatus);
    cl_mem psumY_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY,
            num_of_work_groups * sizeof(float), NULL, &clStatus);
    cl_mem psumXY_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY,
            num_of_work_groups * sizeof(float), NULL, &clStatus);
    cl_mem psumXXY_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY,
            num_of_work_groups * sizeof(float), NULL, &clStatus);
    cl_mem psumXX_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY,
            num_of_work_groups * sizeof(float), NULL, &clStatus);
    cl_mem psumXXX_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY,
            num_of_work_groups * sizeof(float), NULL, &clStatus);
    cl_mem psumXXXX_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY,
            num_of_work_groups * sizeof(float), NULL, &clStatus);

    // Copy the Buffer A and B to the device
    clStatus = clEnqueueWriteBuffer(command_queue, pX_clmem, CL_TRUE, 0,
            NUM_OF_POINTS * sizeof(float), pX, 0, NULL, NULL);
    clStatus = clEnqueueWriteBuffer(command_queue, pY_clmem, CL_TRUE, 0,
            NUM_OF_POINTS * sizeof(float), pY, 0, NULL, NULL);

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1,
            (const char **)&parabolic_regression_kernel, NULL, &clStatus);

    // Build the program
    clStatus = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);
    if(clStatus != CL_SUCCESS)
        LOG_OCL_COMPILER_ERROR(program, device_list[0]);

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "parabolic_regression_kernel", &clStatus);

    // Set the arguments of the kernel
    clStatus = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&pX_clmem);
    clStatus |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&pY_clmem);
    clStatus |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&psumX_clmem);
    clStatus |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&psumY_clmem);
    clStatus |= clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&psumXY_clmem);
    clStatus |= clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&psumXXY_clmem);
    clStatus |= clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&psumXX_clmem);
    clStatus |= clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *)&psumXXX_clmem);
    clStatus |= clSetKernelArg(kernel, 8, sizeof(cl_mem), (void *)&psumXXXX_clmem);
    clStatus |= clSetKernelArg(kernel, 9, WORK_GROUP_SIZE*sizeof(float), NULL);
    clStatus |= clSetKernelArg(kernel, 10, WORK_GROUP_SIZE*sizeof(float), NULL);
    clStatus |= clSetKernelArg(kernel, 11, WORK_GROUP_SIZE*sizeof(float), NULL);
    clStatus |= clSetKernelArg(kernel, 12, WORK_GROUP_SIZE*sizeof(float), NULL);
    clStatus |= clSetKernelArg(kernel, 13, WORK_GROUP_SIZE*sizeof(float), NULL);
    clStatus |= clSetKernelArg(kernel, 14, WORK_GROUP_SIZE*sizeof(float), NULL);
    clStatus |= clSetKernelArg(kernel, 15, WORK_GROUP_SIZE*sizeof(float), NULL);
    clStatus |= clSetKernelArg(kernel, 16, sizeof(int), &points);
    LOG_OCL_ERROR(clStatus, "Kernel Arguments setting failed." );
    
    clStatus = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
            &global_size, &local_size, 0, NULL, NULL);
    LOG_OCL_ERROR(clStatus, "NDRange Failed." );

    // Read the device memory buffer which stores the sums of each work group
    clStatus = clEnqueueReadBuffer(command_queue, psumX_clmem, CL_TRUE, 0,
            num_of_work_groups * sizeof(float), psumX, 0, NULL, NULL);
    clStatus = clEnqueueReadBuffer(command_queue, psumY_clmem, CL_TRUE, 0,
            num_of_work_groups * sizeof(float), psumY, 0, NULL, NULL);
    clStatus = clEnqueueReadBuffer(command_queue, psumXY_clmem, CL_TRUE, 0,
            num_of_work_groups * sizeof(float), psumXY, 0, NULL, NULL);
    clStatus = clEnqueueReadBuffer(command_queue, psumXXY_clmem, CL_TRUE, 0,
            num_of_work_groups * sizeof(float), psumXXY, 0, NULL, NULL);
    clStatus = clEnqueueReadBuffer(command_queue, psumXX_clmem, CL_TRUE, 0,
            num_of_work_groups * sizeof(float), psumXX, 0, NULL, NULL);
    clStatus = clEnqueueReadBuffer(command_queue, psumXXX_clmem, CL_TRUE, 0,
            num_of_work_groups * sizeof(float), psumXXX, 0, NULL, NULL);
    clStatus = clEnqueueReadBuffer(command_queue, psumXXXX_clmem, CL_TRUE, 0,
            num_of_work_groups * sizeof(float), psumXXXX, 0, NULL, NULL);

    float sumX    = 0.0f;
    float sumY    = 0.0f;
    float sumXY   = 0.0f;
    float sumXXY  = 0.0f;
    float sumXX   = 0.0f;
    float sumXXX  = 0.0f;
    float sumXXXX = 0.0f;
    for(int i=0;i<num_of_work_groups;i++)
    {
        sumX    += psumX[i];
        sumY    += psumY[i];
        sumXY   += psumXY[i];
        sumXXY  += psumXXY[i];
        sumXX   += psumXX[i];
        sumXXX  += psumXXX[i];
        sumXXXX += psumXXXX[i];
    }
    std::cout << "SUM_X    = " << sumX << "\n";
    std::cout << "SUM_Y    = " << sumY << "\n";
    std::cout << "SUM_XY   = " << sumXY << "\n";
    std::cout << "SUM_XXY  = " << sumXXY << "\n";
    std::cout << "SUM_XX   = " << sumXX << "\n";
    std::cout << "SUM_XXX  = " << sumXXX << "\n";
    std::cout << "SUM_XXXX = " << sumXXXX << "\n";
    // Clean up and wait for all the comands to complete.
    clStatus = clFlush(command_queue);
    clStatus = clFinish(command_queue);
    /*Below is the call two function which solves a 3 variable set of equations*/
    //A0 = (sumY*sumXX - sumX*sumXY)/(NUM_OF_POINTS*sumXX - sumX*sumX );
    //A1 = (NUM_OF_POINTS*sumXY - sumX*sumY)/(NUM_OF_POINTS*sumXX - sumX*sumX );
    //std::cout << "A0 = " << A0 << "\n";
    //std::cout << "A1 = " << A1 << "\n";
    bool resultValid;
    findParabolla( &A0, &A1, &A2, NUM_OF_POINTS, 
                   sumX, sumXX, sumXXX, sumXXXX, sumY, sumXY, sumXXY,
                   &resultValid );
    if(resultValid)
    {
        std::cout << "The values A0, A1 and A2 are :\n";
        std::cout << "A0 = " << A0 << "\n";
        std::cout << "A1 = " << A1 << "\n";
        std::cout << "A2 = " << A2 << "\n";
    }
    else
    {
        std::cout << "The Set of equation cannot be solved";
    }

    // Finally release all OpenCL allocated objects and host buffers.
    clStatus = clReleaseKernel(kernel);
    clStatus = clReleaseProgram(program);
    clStatus = clReleaseMemObject(pX_clmem);
    clStatus = clReleaseMemObject(pY_clmem);
    clStatus = clReleaseMemObject(psumX_clmem);
    clStatus = clReleaseMemObject(psumY_clmem);
    clStatus = clReleaseMemObject(psumXY_clmem);
    clStatus = clReleaseMemObject(psumXXY_clmem);
    clStatus = clReleaseMemObject(psumXX_clmem);
    clStatus = clReleaseMemObject(psumXXX_clmem);
    clStatus = clReleaseMemObject(psumXXXX_clmem);
    clStatus = clReleaseCommandQueue(command_queue);
    clStatus = clReleaseContext(context);
    /*Free all Memory Allocations*/
    free(pX);
    free(pY);
    free(psumX);
    free(psumY);
    free(psumXY);
    free(psumXXY);
    free(psumXX);
    free(psumXXX);
    free(psumXXXX);
    OCL_RELEASE_PLATFORMS( platforms );   
    OCL_RELEASE_DEVICES( device_list );    

    return 0;
}
