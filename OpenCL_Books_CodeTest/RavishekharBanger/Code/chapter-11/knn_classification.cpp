#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include <ocl_macros.h>

#define DATA_SIZE 1024
#define NUM_CLASSES 3

#define NUM_OF_POINTS 1024
#define WORK_GROUP_SIZE 64
#define K_CLASSIFICATION_POINTS 16
//Change in the below struct should result in the OpenCL code change also 
typedef struct {     
    int x;   
    int y;   
    int classification;
} point; 

//We will keep the calssification as zero for now.
point matchPoint = {12,13,0};

//OpenCL kernel which is run for every work item created.
const char *knn_classification_kernel =
"#define DISTANCE_TYPE float                                                            \n"
"                                                                                       \n"
"typedef struct {                                                                       \n"
"    int x;                                                                             \n"
"    int y;                                                                             \n"
"    int classification;                                                                \n"
"} point;                                                                               \n"
"                                                                                       \n"
"typedef point POINT;                                                                   \n"
"                                                                                       \n"
"DISTANCE_TYPE point_distance(POINT from, POINT to)                                     \n"
"{                                                                                      \n"
"    DISTANCE_TYPE distance = 0.0f;                                                     \n"
"    distance += (from.x - to.x)*(from.x - to.x);                                       \n"
"    distance += (from.y - to.y)*(from.y - to.y);                                       \n"
"    return sqrt ( distance );                                                          \n"
"}                                                                                      \n"
"                                                                                       \n"
"                                                                                       \n"
"__kernel                                                                               \n"        
"void knn_distance_kernel(                                                              \n"        
"                    POINT match,                                                       \n"    
"                  __global POINT *data_set,                                            \n"    
"                  __global DISTANCE_TYPE *distance_data)                               \n"        
"{                                                                                      \n"        
"    //Get the index of the work-item                                                   \n"        
"    int gid   = get_global_id (0);                                                     \n"           
"    int lid   = get_local_id (0);                                                      \n"                                                                                                         
"    POINT read_point = data_set[gid];                                                  \n"                     
"    DISTANCE_TYPE computed_distance = point_distance (read_point, match );             \n"                                                                        
"                                                                                       \n"        
"    distance_data[gid] = computed_distance;                                            \n"
"    //printf(\"x=%d y=%d z=%d \",read_point.x, read_point.y, read_point.classification); \n"
"    //printf(\"compute distance =%f\\n\", computed_distance);                          \n"
"    //You may want to sort here locally first;                                         \n"
"}                                                                                      \n"
"                                                                                       \n"
"                                                                                       \n"
"                                                                                       \n"
"//The bitonic sort kernel does an ascending sort                                       \n"
"kernel                                                                                 \n"
"void knn_bitonic_sort_kernel(__global DISTANCE_TYPE * input_ptr,                           \n"
"                 __global POINT *data_set,                                             \n"
"                 const uint stage,                                                     \n"
"                 const uint passOfStage )                                              \n"
"{                                                                                      \n"
"    uint threadId = get_global_id(0);                                                  \n"
"    uint pairDistance = 1 << (stage - passOfStage);                                    \n"
"    uint blockWidth   = 2 * pairDistance;                                              \n"
"    uint temp;                                                                         \n"
"    uint leftId = (threadId & (pairDistance -1))                                       \n"
"                       + (threadId >> (stage - passOfStage) ) * blockWidth;            \n"
"    bool compareResult;                                                                \n"
"    //printf(\"bs_kernel = %d \\n\",get_global_id(0));                                 \n"
"    uint rightId = leftId + pairDistance;                                              \n"
"                                                                                       \n"
"    DISTANCE_TYPE leftElement, rightElement;                                           \n"
"    DISTANCE_TYPE greater, lesser;                                                     \n"
"    POINT     leftPoint, rightPoint;                                                   \n"
"    POINT     greaterPoint, lesserPoint;                                               \n"
"    leftElement  = input_ptr[leftId];                                                  \n"
"    leftPoint    = data_set[leftId];                                                   \n"
"    rightElement = input_ptr[rightId];                                                 \n"
"    rightPoint   = data_set[rightId];                                                  \n"
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
"    greaterPoint = compareResult?rightPoint:leftPoint;                                 \n"
"    lesser       = compareResult?leftElement:rightElement;                             \n"
"    lesserPoint  = compareResult?leftPoint:rightPoint;                                 \n"
"                                                                                       \n"
"    input_ptr[leftId]  = lesser;                                                       \n"
"    data_set[leftId]   = lesserPoint;                                                  \n"
"    input_ptr[rightId] = greater;                                                      \n"
"    data_set[rightId]  = greaterPoint;                                                 \n"
"}                                                                                      \n";

// This function reads a file of points and its classifications. 
// Each line in the file is of the form. 
// <x-coord> <y-coord> <point-classification>
//
// points - is the data structure consisting of points in 2D coordinate space. 
// numPoints - Number of files to be read from the file

bool readKNNData(point *pPoints, int numPoints)
{
    if( (NULL == pPoints) )
    {
        return false;
    }
    // 
    std::ifstream inputFile("kNNData.txt");

    if(!inputFile.is_open())
    {
        return false;
    }
    int pCheckClassSum[3];
    for(int i=0;i<NUM_CLASSES;++i)
    {
        pCheckClassSum[i] = 0;
    }
    int x1,x2,y;
    for(int i=0; i<numPoints; ++i)
    {
        inputFile >> x1 >> x2 >> y;
        pPoints[i].x = x1;
        pPoints[i].y = x2;
        pPoints[i].classification = y;
        pCheckClassSum[y]++;
        //std::cout <<"File Input:  x = "<<pPoints[i].x<<", y = "<<pPoints[i].y<<"  yclass = "<<pPoints[i].classification <<"\n";
    }
    for(int i=0;i<NUM_CLASSES;++i)
    {
        std::cout<<"pCheckClassSum["<<i<<"]="<<pCheckClassSum[i]<<"\n";
    }
    return true;
}
int main(void) {
    cl_int clStatus;

    // Allocate space for vectors of 
    point *pPoints = (point*)malloc(sizeof(point)*NUM_OF_POINTS);
    readKNNData(pPoints, NUM_OF_POINTS);

    printf("Host sizeof (point) = %ld\n", sizeof(point) );
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
    size_t global_size = NUM_OF_POINTS;             // Process all points. Each work item shall process a point
    size_t local_size  = WORK_GROUP_SIZE;           // This is the size of the work group.
    size_t num_of_work_groups = global_size/local_size; // Calculate the Number of work groups.

    //Create memory buffers on the device for each vector
    cl_mem pPoints_clmem = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR,
            NUM_OF_POINTS * sizeof(point), (void *)pPoints, &clStatus);
    cl_mem pDistance_clmem = clCreateBuffer(context, CL_MEM_READ_WRITE,
            NUM_OF_POINTS * sizeof(float), NULL, &clStatus);

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1,
            (const char **)&knn_classification_kernel, NULL, &clStatus);

    // Build the program
    clStatus = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);
    if(clStatus != CL_SUCCESS)
        LOG_OCL_COMPILER_ERROR(program, device_list[0]);

    // Create the OpenCL kernel
    cl_kernel distance_kernel = clCreateKernel(program, "knn_distance_kernel", &clStatus);
    LOG_OCL_ERROR(clStatus, "CreateKernel Failed on kernel knn_distance_kernel." );
    cl_kernel bitonic_sort_kernel = clCreateKernel(program, "knn_bitonic_sort_kernel", &clStatus);
    LOG_OCL_ERROR(clStatus, "CreateKernel Failed on kernel knn_bitonic_sort_kernel." );

    // Set the arguments of the distance kernel
    clStatus  = clSetKernelArg(distance_kernel, 0, sizeof(point), &matchPoint);
    clStatus |= clSetKernelArg(distance_kernel, 1, sizeof(cl_mem), (void *)&pPoints_clmem);
    clStatus |= clSetKernelArg(distance_kernel, 2, sizeof(cl_mem), (void *)&pDistance_clmem);
    LOG_OCL_ERROR(clStatus, "Kernel Arguments setting failed." );
    cl_event distance_event;
    clStatus = clEnqueueNDRangeKernel(command_queue, distance_kernel, 1, NULL,
            &global_size, &local_size, 0, NULL, &distance_event);
    LOG_OCL_ERROR(clStatus, "NDRange Failed." );
    clStatus = clWaitForEvents(1, &distance_event);
    LOG_OCL_ERROR(clStatus, "Wait for distance_kernel Failed." );

    //Sort the distance buffer using Bitonic Sort.
    clStatus = clSetKernelArg(bitonic_sort_kernel, 0, sizeof(cl_mem), (void *)&pDistance_clmem);
    clStatus |= clSetKernelArg(bitonic_sort_kernel, 1, sizeof(cl_mem), (void *)&pPoints_clmem);
    LOG_OCL_ERROR(clStatus, "SetKernelArg failed for bitonic_sort_kernel." );

    unsigned int stage, passOfStage, numStages, temp;
    stage = passOfStage = numStages = 0;
    for(temp = NUM_OF_POINTS; temp > 1; temp >>= 1)
        ++numStages;
    global_size = NUM_OF_POINTS>>1;
    local_size  = WORK_GROUP_SIZE;
    for(stage = 0; stage < numStages; ++stage)
    {
        // stage of the algorithm
        clStatus = clSetKernelArg(bitonic_sort_kernel, 2, sizeof(int), (void *)&stage);
        // Every stage has stage + 1 passes
        for(passOfStage = 0; passOfStage < stage + 1; ++passOfStage) {
            // pass of the current stage
            clStatus = clSetKernelArg(bitonic_sort_kernel, 3, sizeof(int), (void *)&passOfStage);
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
    
    float *mapped_distance = (float *)clEnqueueMapBuffer(command_queue, pDistance_clmem, true, CL_MAP_READ, 0, sizeof(float) * NUM_OF_POINTS, 0, NULL, NULL, &clStatus);
    point *mapped_points   = (point *)clEnqueueMapBuffer(command_queue, pPoints_clmem, true, CL_MAP_WRITE, 0, sizeof(point) * NUM_OF_POINTS, 0, NULL, NULL, &clStatus);
    // Display the Sorted K points on the screen
    for(int i = 0; i < K_CLASSIFICATION_POINTS; i++)
        printf( "point(%d, %d, %d) = %3.8f \n", mapped_points[i].x,mapped_points[i].y,mapped_points[i].classification, mapped_distance[i] );

    // Finally release all OpenCL allocated objects and host buffers.
    clStatus = clReleaseKernel(distance_kernel);
    clStatus = clReleaseKernel(bitonic_sort_kernel);
    clStatus = clReleaseProgram(program);
    clStatus = clReleaseMemObject(pPoints_clmem);
    clStatus = clReleaseMemObject(pDistance_clmem);
    clStatus = clReleaseCommandQueue(command_queue);
    clStatus = clReleaseContext(context);
    
    // Free all Memory Allocations
    free(pPoints);

    // Release Platforms and Devices
    OCL_RELEASE_PLATFORMS( platforms );
    OCL_RELEASE_DEVICES( device_list );

    return 0;
}
