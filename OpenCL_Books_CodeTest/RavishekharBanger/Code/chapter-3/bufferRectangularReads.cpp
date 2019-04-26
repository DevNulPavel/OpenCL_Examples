#include <stdio.h>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include <iostream>
#include <ocl_macros.h>

#define NUM_OF_ELEMENTS 32 

int main(int argc, char *argv[])
{
    cl_int status = 0;
    cl_device_type dType = CL_DEVICE_TYPE_GPU;
    cl_platform_id platform = NULL;
    cl_device_id   device;
    cl_context     context;
    cl_command_queue commandQueue;
    cl_mem clBuffer;
    cl_int hostBuffer[NUM_OF_ELEMENTS] =
    {
         0,  1,  2,  3, 
         4,  5,  6,  7,
         8,  9, 10, 11, 
        12, 13, 14, 15,
        16, 17, 18, 19,
        20, 21, 22, 23, 
        24, 25, 26, 27,
        28, 29, 30, 31, 
    };

    //Setup the OpenCL Platform, 
    //Get the first available platform. Use it as the default platform
    status = clGetPlatformIDs(1, &platform, NULL);
    LOG_OCL_ERROR(status, "clGetPlatformIDs Failed..." );

    //Get the first available device 
    status = clGetDeviceIDs (platform, dType, 1, &device, NULL);
    LOG_OCL_ERROR(status, "clGetDeviceIDs Failed..." );
    
    //Create an execution context for the selected platform and device. 
    cl_context_properties cps[3] = 
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platform,
        0
    };
    context = clCreateContextFromType(
        cps,
        dType,
        NULL,
        NULL,
        &status);
    LOG_OCL_ERROR(status, "clCreateContextFromType Failed..." );

    // Create command queue
    commandQueue = clCreateCommandQueue(context,
                                        device,
                                        0,
                                        &status);
    LOG_OCL_ERROR(status, "clCreateCommandQueue Failed..." );

    //Create OpenCL device input buffer
    clBuffer = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(cl_uint) * NUM_OF_ELEMENTS,
        hostBuffer,
        &status); 
    LOG_OCL_ERROR(status, "clCreateBuffer Failed..." );

    //Read a 2D rectangular object from the clBuffer of 32 elements
    int hostPtr2D[6] = {0, 0, 0, 0, 0, 0};
    size_t bufferOrigin2D[3] = {1*sizeof(int), 6, 0};
    size_t hostOrigin2D[3] = {0 ,0, 0};
    size_t region2D[3] = {3* sizeof(int), 2,1};
    status = clEnqueueReadBufferRect(
                        commandQueue,
                        clBuffer,
                        CL_TRUE,
                        bufferOrigin2D, /*Start of a 2D buffer to read from*/
                        hostOrigin2D,
                        region2D,
                        (NUM_OF_ELEMENTS / 8) * sizeof(int), /*buffer_row_pitch  */
                        0,                                   /*buffer_slice_pitch*/
                        0,                                   /*host_row_pitch    */
                        0,                                   /*host_slice_pitch  */
                        static_cast<void*>(hostPtr2D),
                        0,
                        NULL,
                        NULL);
    LOG_OCL_ERROR(status, "clEnqueueReadBufferRect Failed..." );
    std::cout << "2D rectangle selected is as follows" << std::endl;
    std::cout << " " << hostPtr2D[0];
    std::cout << " " << hostPtr2D[1];
    std::cout << " " << hostPtr2D[2] << std::endl;
    std::cout << " " << hostPtr2D[3];
    std::cout << " " << hostPtr2D[4];
    std::cout << " " << hostPtr2D[5] << std::endl;

    //Read a 3D rectangular object from the clBuffer of 32 elements
    int hostPtr3D[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    size_t bufferOrigin3D[3] = {1*sizeof(int), 1, 0};
    size_t hostOrigin3D[3] = {0 ,0, 0};
    size_t region3D[3] = {3* sizeof(int), 1,3};
    status = clEnqueueReadBufferRect(
                        commandQueue,
                        clBuffer,
                        CL_TRUE,
                        bufferOrigin3D, /*Start of a 2D buffer to read from*/
                        hostOrigin3D,
                        region3D,
                        (NUM_OF_ELEMENTS / 8) * sizeof(int), /*buffer_row_pitch  */
                        (NUM_OF_ELEMENTS / 4) * sizeof(int), /*buffer_slice_pitch*/
                        0,                                   /*host_row_pitch    */
                        0,                                   /*host_slice_pitch  */
                        static_cast<void*>(hostPtr3D),
                        0,
                        NULL,
                        NULL);
    LOG_OCL_ERROR(status, "clEnqueueReadBufferRect Failed..." );
    std::cout << "3D rectangle selected is as follows" << std::endl;
    std::cout << " " << hostPtr3D[0];
    std::cout << " " << hostPtr3D[1];
    std::cout << " " << hostPtr3D[2] << std::endl;
    std::cout << " " << hostPtr3D[3];
    std::cout << " " << hostPtr3D[4];
    std::cout << " " << hostPtr3D[5] << std::endl;
    std::cout << " " << hostPtr3D[6];
    std::cout << " " << hostPtr3D[7];
    std::cout << " " << hostPtr3D[8] << std::endl;

    return 0;
}
