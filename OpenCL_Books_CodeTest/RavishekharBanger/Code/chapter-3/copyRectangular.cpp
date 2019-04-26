#include <stdio.h>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include <ocl_macros.h>
#include <iostream>

#define NUM_OF_ELEMENTS 32 
#define DEVICE_TYPE CL_DEVICE_TYPE_GPU

int main(int argc, char *argv[])
{
    cl_int           status = 0;
    cl_context       context;
    cl_command_queue commandQueue;
    cl_mem           clBufferSrc, clBufferDst;

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
    // Get platform and device information
    cl_platform_id * platforms = NULL;
    OCL_CREATE_PLATFORMS( platforms );

    // Get the devices list and choose the type of device you want to run on
    cl_device_id *device_list = NULL;
    OCL_CREATE_DEVICE( platforms[0], DEVICE_TYPE, device_list);
    
    //Create an execution context for the selected platform and device. 
    cl_context_properties cps[3] = 
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platforms[0],
        0
    };
    context = clCreateContextFromType(
        cps,
        DEVICE_TYPE,
        NULL,
        NULL,
        &status);
    LOG_OCL_ERROR(status, "clCreateContextFromType Failed..." );

    // Create command queue
    commandQueue = clCreateCommandQueue(context,
                                        device_list[0],
                                        0,
                                        &status);
    LOG_OCL_ERROR(status, "clCreateCommandQueue Failed..." );

    //Create OpenCL device input buffer
    clBufferSrc = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(cl_uint) * NUM_OF_ELEMENTS,
        hostBuffer,
        &status); 
    LOG_OCL_ERROR(status, "clCreateBuffer Failed..." );

    //Copy commands
    cl_uint copyBuffer[NUM_OF_ELEMENTS] = 
    {
        0,  0,  0,  0, 
        0,  0,  0,  0,
        0,  0,  0,  0, 
        0,  0,  0,  0,
        0,  0,  0,  0,
        0,  0,  0,  0, 
        0,  0,  0,  0,
        0,  0,  0,  0
    };
    
    clBufferDst = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(cl_uint) * NUM_OF_ELEMENTS,
        copyBuffer,
        &status); 
    LOG_OCL_ERROR(status, "clCreateBuffer Failed..." );

    //Copy the contents of the rectangular buffer pointed by clBufferSrc to  clBufferSrc
    size_t   src_origin[3] = {0, 4, 0};
    size_t   dst_origin[3] = {0, 1, 0};
    size_t   region[3]     = {3*sizeof(int), 2, 1}; /*width in bytes, height in rows, depth in slices 
                                                      For 2D depth should be 1*/
    cl_event copyEvent;
    status = clEnqueueCopyBufferRect ( commandQueue, 	//copy command will be queued
                    clBufferSrc,		
                    clBufferDst,	
                    src_origin,	  //in bytes - src_origin[2] * src_slice_pitch + src_origin[1] * src_row_pitch + src_origin[0]
                    dst_origin,   //in bytes - dst_origin[2] * dst_slice_pitch + dst_origin[1] * dst_row_pitch + dst_origin[0]
                    region,		  //(width, height, depth) in bytes of the 2D or 3D rectangle being copied
                    (NUM_OF_ELEMENTS / 8) * sizeof(int), /*src_row_pitch  */
                    0,            //src_slice_pitch - Its a 2D buffers. Hence the slice size is 0
                    (NUM_OF_ELEMENTS / 8) * sizeof(int), /*dst_row_pitch  */
                    0,            //dst_slice_pitch - Its a 2D buffers. Hence the slice size is 0
                    0,
                    NULL,
                    &copyEvent);
    LOG_OCL_ERROR(status, "clEnqueueCopyBufferRect Failed..." );
    
    status = clWaitForEvents(1, &copyEvent);
    
    status = clEnqueueReadBuffer(
        commandQueue,
        clBufferDst,
        CL_TRUE,
        0,
        NUM_OF_ELEMENTS * sizeof(cl_uint),
        copyBuffer,
        0,
        NULL,
        NULL);
    LOG_OCL_ERROR(status, "clEnqueueReadBuffer Failed..." );

    std::cout << "The copied destination buffer is as follows" << std::endl;
    for(int i=0; i<8; i++)
    {
        std::cout << std::endl; 
        for(int j=0; j<NUM_OF_ELEMENTS/8; j++)
        {
            std::cout << " " << copyBuffer[i*(NUM_OF_ELEMENTS/8) + j];
        }
    }
    return 0;
}
