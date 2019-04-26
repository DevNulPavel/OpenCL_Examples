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

#define DATA_TYPE int   //The kernel define should also be changed.
#define DATA_SIZE 1024   //DATA_SIZE should be a multiple of 64
#define WORK_GROUP_SIZE 64


//OpenCL kernel which is run for every work item created.
const char *bitonic_sort_kernel =
"#define DATA_TYPE int                                                                  \n"
"                                                                                       \n"
"//The bitonic sort kernel does an ascending sort                                       \n"
"kernel                                                                                 \n"
"void bitonic_sort_kernel(__global DATA_TYPE * input_ptr,                               \n"
"                 const uint stage,                                                     \n"
"                 const uint passOfStage )                                              \n"
"{                                                                                      \n"
"    uint threadId = get_global_id(0);                                                  \n"
"    uint pairDistance = 1 << (stage - passOfStage);                                    \n"
"    uint blockWidth   = 2 * pairDistance;                                              \n"
"    uint temp;                                                                         \n"
"    bool compareResult;                                                                \n"
"    uint leftId = (threadId & (pairDistance -1))                                       \n"
"                       + (threadId >> (stage - passOfStage) ) * blockWidth;            \n"
"    uint rightId = leftId + pairDistance;                                              \n"
"                                                                                       \n"
"    DATA_TYPE leftElement, rightElement;                                               \n"
"    DATA_TYPE greater, lesser;                                                         \n"
"    leftElement  = input_ptr[leftId];                                                  \n"
"    rightElement = input_ptr[rightId];                                                 \n"
"                                                                                       \n"
"    uint sameDirectionBlockWidth = threadId >> stage;                                  \n"
"    uint sameDirection = sameDirectionBlockWidth & 0x1;                                \n"
"                                                                                       \n"
"    temp    = sameDirection?rightId:temp;                                              \n"
"    rightId = sameDirection?leftId:rightId;                                            \n"
"    leftId  = sameDirection?temp:leftId;                                               \n"
"                                                                                       \n"
"    compareResult = (leftElement < rightElement) ;                                     \n"
"                                                                                       \n"
"    greater      = compareResult?rightElement:leftElement;                             \n"
"    lesser       = compareResult?leftElement:rightElement;                             \n"
"                                                                                       \n"
"    input_ptr[leftId]  = lesser;                                                       \n"
"    input_ptr[rightId] = greater;                                                      \n"
"}                                                                                      \n";


int main(void) {
    cl_int clStatus;

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

    // Create a command queue for the selected device
    cl_command_queue command_queue = clCreateCommandQueue(context, device_list[0], 0, &clStatus);

    // Execute the OpenCL kernel on the list
    size_t global_size = DATA_SIZE/2;                   // Each work item shall compare two elements.
    size_t local_size  = WORK_GROUP_SIZE;               // This is the size of the work group.
    size_t num_of_work_groups = global_size/local_size; // Calculate the Number of work groups.

    //Allocate memory and initialize the input buffer.
    DATA_TYPE *pInputBuffer = (DATA_TYPE*)malloc(sizeof(DATA_TYPE)*DATA_SIZE);
    for(int i =0; i< DATA_SIZE; i++)
    {
        pInputBuffer[i] = DATA_SIZE - i;//(DATA_TYPE)rand();
        printf("pInputBuffer[i] = %4d\n",pInputBuffer[i]);
    }
    //Create memory buffers on the device for each vector
    cl_mem pInputBuffer_clmem = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR,
            DATA_SIZE * sizeof(DATA_TYPE), pInputBuffer, &clStatus);

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1,
            (const char **)&bitonic_sort_kernel, NULL, &clStatus);

    // Build the program
    clStatus = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);
    if(clStatus != CL_SUCCESS)
        LOG_OCL_COMPILER_ERROR(program, device_list[0]);

    // Create the OpenCL kernel
    cl_kernel bitonic_sort_kernel = clCreateKernel(program, "bitonic_sort_kernel", &clStatus);
    LOG_OCL_ERROR(clStatus, "CreateKernel Failed on kernel knn_bitonic_sort_kernel." );

    //Sort the input buffer using Bitonic Sort.
    clStatus = clSetKernelArg(bitonic_sort_kernel, 0, sizeof(cl_mem), (void *)&pInputBuffer_clmem);
    LOG_OCL_ERROR(clStatus, "SetKernelArg failed for bitonic_sort_kernel." );

    unsigned int stage, passOfStage, numStages, temp;
    stage = passOfStage = numStages = 0;
    for(temp = DATA_SIZE; temp > 1; temp >>= 1)
        ++numStages;
    global_size = DATA_SIZE>>1;
    local_size  = WORK_GROUP_SIZE;
    for(stage = 0; stage < numStages; ++stage)
    {
        // stage of the algorithm
        std::cout << "---------------------------------------------"<<"Stage no "<< stage << "---------------------------------------------" <<std::endl;
        clStatus = clSetKernelArg(bitonic_sort_kernel, 1, sizeof(int), (void *)&stage);
        // Every stage has stage + 1 passes
        for(passOfStage = 0; passOfStage < stage + 1; ++passOfStage) {
            // pass of the current stage
            std::cout << "Pass no "<< passOfStage << std::endl;
            clStatus = clSetKernelArg(bitonic_sort_kernel, 2, sizeof(int), (void *)&passOfStage);
            //
            // Enqueue a kernel run call.
            // Each thread writes a sorted pair.
            // So, the number of  threads (global) should be half the length of the input buffer.
            //
            clStatus = clEnqueueNDRangeKernel(command_queue, bitonic_sort_kernel, 1, NULL,
                                              &global_size, &local_size, 0, NULL, NULL);
            LOG_OCL_ERROR(clStatus, "enqueueNDRangeKernel() failed for sort() kernel." );
            clFinish(command_queue);
        }//end of for passStage = 0:stage-1
    }//end of for stage = 0:numStage-1

    DATA_TYPE *mapped_input_buffer = (DATA_TYPE *)clEnqueueMapBuffer(command_queue, pInputBuffer_clmem, true, CL_MAP_READ, 0, sizeof(DATA_TYPE) * DATA_SIZE, 0, NULL, NULL, &clStatus);
    // Display the Sorted data on the screen
    for(int i = 0; i < DATA_SIZE; i++)
        printf( "%d  ", mapped_input_buffer[i] );

    // Finally release all OpenCL allocated objects and host buffers.
    clStatus = clReleaseKernel(bitonic_sort_kernel);
    clStatus = clReleaseProgram(program);
    clStatus = clReleaseMemObject(pInputBuffer_clmem);
    clStatus = clReleaseCommandQueue(command_queue);
    clStatus = clReleaseContext(context);

    // Free all Memory Allocations
    free(pInputBuffer);

    // Release Platforms and Devices
    OCL_RELEASE_PLATFORMS( platforms );
    OCL_RELEASE_DEVICES( device_list );

    return 0;
}
