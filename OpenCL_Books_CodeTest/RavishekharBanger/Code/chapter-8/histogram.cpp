#include <cstdlib>
#include <iostream>
#include <string.h>
#include <stdio.h>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include <ocl_macros.h>
#include <bmp_image.h>
//#define USE_HOST_MEMORY
const char *histogram_kernel =
"#define BIN_SIZE            256                                    \n"
"#define ELEMENTS_TO_PROCESS 256                                    \n"
"__kernel                                                           \n"
"void histogram_kernel(__global const uint* data,                   \n"
"                  __global uint* binResultR,                       \n"
"                  __global uint* binResultG,                       \n"
"                  __global uint* binResultB)                       \n"
"{                                                                  \n"
"  __local int sharedArrayR[BIN_SIZE];                              \n"
"  __local int sharedArrayG[BIN_SIZE];                              \n"
"  __local int sharedArrayB[BIN_SIZE];                              \n"
"  __global uchar4 * image_data = data;                             \n"
"  size_t localId   = get_local_id(0);                              \n"
"  size_t globalId  = get_global_id(0);                             \n"
"  size_t groupId   = get_group_id(0);                              \n"
"  size_t groupSize = get_local_size(0);                            \n"
"                                                                   \n"
"  /* initialize shared array to zero */                            \n"
"  sharedArrayR[localId] = 0;                                       \n"
"  sharedArrayG[localId] = 0;                                       \n"
"  sharedArrayB[localId] = 0;                                       \n"
"                                                                   \n"
"  barrier(CLK_LOCAL_MEM_FENCE);                                    \n"
"  int groupOffset = groupId * groupSize * ELEMENTS_TO_PROCESS;     \n"
"  /* calculate thread-histograms */                                \n"
"  for(int i = 0; i < ELEMENTS_TO_PROCESS; ++i)                     \n"
"  {                                                                \n"
"    int index = groupOffset + i * groupSize + localId;             \n"
"    //Coalesced read from global memory                            \n"
"    uchar4 value = image_data[index];                              \n"
"    atomic_inc(&sharedArrayR[value.x]);                            \n"
"    atomic_inc(&sharedArrayG[value.y]);                            \n"
"    atomic_inc(&sharedArrayB[value.w]);                            \n"
"  }                                                                \n"
"                                                                   \n"
"  barrier(CLK_LOCAL_MEM_FENCE);                                    \n"
"  //Coalesced write to global memory                               \n"
"  binResultR[groupId * BIN_SIZE + localId] = sharedArrayR[localId];\n"
"  binResultG[groupId * BIN_SIZE + localId] = sharedArrayR[localId];\n"
"  binResultB[groupId * BIN_SIZE + localId] = sharedArrayR[localId];\n"
"                                                                   \n"
"}                                                                  \n";


int main(int argc, char *argv[])
{
    cl_int status = 0;
    cl_int binSize = 256;
    cl_int groupSize = 256;
    cl_int subHistgCnt;
    cl_device_type dType = CL_DEVICE_TYPE_GPU;
    cl_platform_id platform = NULL;
    cl_device_id   device;
    cl_context     context;
    cl_command_queue commandQueue;
    cl_mem         imageBuffer;
    cl_mem     intermediateHistR, intermediateHistG, intermediateHistB; /*Intermediate Image Histogram buffer*/
    cl_uint *  midDeviceBinR, *midDeviceBinG, *midDeviceBinB;
    cl_uint  *deviceBinR,*deviceBinG,*deviceBinB;
    //Read a BMP Image
    Image *image;
    std::string filename = "sample_color.bmp";
    ReadBMPImage(filename, &image);
    if(image == NULL)
    {
        printf("File %s not present...\n", filename.c_str());
        return 0;
    }

    subHistgCnt   = (image->width * image->height)/(binSize*groupSize);
    midDeviceBinR = (cl_uint*)malloc(binSize * subHistgCnt * sizeof(cl_uint));
    midDeviceBinG = (cl_uint*)malloc(binSize * subHistgCnt * sizeof(cl_uint));
    midDeviceBinB = (cl_uint*)malloc(binSize * subHistgCnt * sizeof(cl_uint));
    deviceBinR    = (cl_uint*)malloc(binSize * sizeof(cl_uint));
    deviceBinG    = (cl_uint*)malloc(binSize * sizeof(cl_uint));
    deviceBinB    = (cl_uint*)malloc(binSize * sizeof(cl_uint));
    
    //Setup the OpenCL Platform, 
    //Get the first available platform. Use it as the default platform
    status = clGetPlatformIDs(1, &platform, NULL);
    LOG_OCL_ERROR(status, "clGetPlatformIDs Failed." );

    //Get the first available device 
    status = clGetDeviceIDs (platform, dType, 1, &device, NULL);
    LOG_OCL_ERROR(status, "clGetDeviceIDs Failed." );
    
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
    LOG_OCL_ERROR(status, "clCreateContextFromType Failed." );

    // Create command queue
    commandQueue = clCreateCommandQueue(context,
                                        device,
                                        0,
                                        &status);
    LOG_OCL_ERROR(status, "clCreateCommandQueue Failed." );
#if !defined(USE_HOST_MEMORY)
    //Create OpenCL device input buffer
    imageBuffer = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY,
        sizeof(cl_uint) * image->width * image->height,
        NULL,
        &status); 
    LOG_OCL_ERROR(status, "clCreateBuffer Failed while creating the image buffer." );

    //Set input data 
    cl_event writeEvt;
    status = clEnqueueWriteBuffer(commandQueue,
                                  imageBuffer,
                                  CL_FALSE,
                                  0,
                                  image->width * image->height * sizeof(cl_uint),
                                  image->pixels,
                                  0,
                                  NULL,
                                  &writeEvt);
    LOG_OCL_ERROR(status, "clEnqueueWriteBuffer Failed while writing the image data." );
    status = clFinish(commandQueue);
    LOG_OCL_ERROR(status, "clFinish Failed while writing the image data." );
#else
    //Create OpenCL device input buffer
    imageBuffer = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR,
        sizeof(cl_uint) * image->width * image->height,
        image->pixels,
        &status); 
    LOG_OCL_ERROR(status, "clCreateBuffer Failed while creating the image buffer." );
#endif

    
    //Create OpenCL device output buffer
    intermediateHistR = clCreateBuffer(
        context, 
        CL_MEM_WRITE_ONLY,
        sizeof(cl_uint) * binSize * subHistgCnt, 
        NULL, 
        &status);
    LOG_OCL_ERROR(status, "clCreateBuffer Failed." );

    intermediateHistG = clCreateBuffer(
        context,
        CL_MEM_WRITE_ONLY,
        sizeof(cl_uint) * binSize * subHistgCnt,
        NULL,
        &status);
    LOG_OCL_ERROR(status, "clCreateBuffer Failed." );

    intermediateHistB = clCreateBuffer(
        context,
        CL_MEM_WRITE_ONLY,
        sizeof(cl_uint) * binSize * subHistgCnt,
        NULL,
        &status);
    LOG_OCL_ERROR(status, "clCreateBuffer Failed." );

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1,
            (const char **)&histogram_kernel, NULL, &status);
    LOG_OCL_ERROR(status, "clCreateProgramWithSource Failed." );

    // Build the program
    status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    LOG_OCL_ERROR(status, "clBuildProgram Failed" );
    if(status != CL_SUCCESS)
    {
        if(status == CL_BUILD_PROGRAM_FAILURE)
            LOG_OCL_COMPILER_ERROR(program, device);
        LOG_OCL_ERROR(status, "clBuildProgram Failed" );
    }


    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "histogram_kernel", &status);
    LOG_OCL_ERROR(status, "clCreateKernel Failed." );
    // Set the arguments of the kernel
    status =  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&imageBuffer); 
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&intermediateHistR);
    status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&intermediateHistG);
    status |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&intermediateHistB);
    LOG_OCL_ERROR(status, "clSetKernelArg Failed." );
    // Execute the OpenCL kernel on the list
    cl_event ndRangeEvt;
    size_t globalThreads = (image->width * image->height) / (binSize*groupSize) * groupSize;
    size_t localThreads = groupSize;
    status = clEnqueueNDRangeKernel(
        commandQueue,
        kernel,
        1,
        NULL,
        &globalThreads,
        &localThreads,
        0,
        NULL,
        &ndRangeEvt);
    LOG_OCL_ERROR(status, "clEnqueueNDRangeKernel Failed." );

    status = clFinish(commandQueue);
    LOG_OCL_ERROR(status, "clFinish Failed." );

    //Read the histogram back into the host memory.
    memset(deviceBinR, 0, binSize * sizeof(cl_uint));
    memset(deviceBinG, 0, binSize * sizeof(cl_uint));
    memset(deviceBinB, 0, binSize * sizeof(cl_uint));
    cl_event readEvt[3];
    status = clEnqueueReadBuffer(
        commandQueue,
        intermediateHistR,
        CL_FALSE,
        0,
        subHistgCnt * binSize * sizeof(cl_uint),
        midDeviceBinR,
        0,
        NULL,
        &readEvt[0]);
    LOG_OCL_ERROR(status, "clEnqueueReadBuffer of intermediateHistR Failed." );
    
    status = clEnqueueReadBuffer(
        commandQueue,
        intermediateHistG,
        CL_FALSE,
        0,
        subHistgCnt * binSize * sizeof(cl_uint),
        midDeviceBinG,
        0,
        NULL,
        &readEvt[1]);
    LOG_OCL_ERROR(status, "clEnqueueReadBuffer of intermediateHistG Failed." );
    
    status = clEnqueueReadBuffer(
        commandQueue,
        intermediateHistB,
        CL_FALSE,
        0,
        subHistgCnt * binSize * sizeof(cl_uint),
        midDeviceBinB,
        0,
        NULL,
        &readEvt[2]);
    LOG_OCL_ERROR(status, "clEnqueueReadBuffer of intermediateHistB Failed." );
    
    status = clWaitForEvents(3, readEvt);
    //status = clFinish(commandQueue);
    LOG_OCL_ERROR(status, "clWaitForEvents for readEvt." );

    // Calculate final histogram bin 
    for(int i = 0; i < subHistgCnt; ++i)
    {
        for(int j = 0; j < binSize; ++j)
        {
            deviceBinR[j] += midDeviceBinR[i * binSize + j];
            deviceBinG[j] += midDeviceBinG[i * binSize + j];
            deviceBinB[j] += midDeviceBinB[i * binSize + j];
        }
    }

    // Validate the histogram operation. 
    // The idea behind this is that once a histogram is computed the sum of all the bins should be equal to the number of pixels.
    int totalPixelsR = 0;
    int totalPixelsG = 0;
    int totalPixelsB = 0;
    for(int j = 0; j < binSize; ++j)
    {
        totalPixelsR += deviceBinR[j];
        totalPixelsG += deviceBinG[j];
        totalPixelsB += deviceBinB[j];
    }
    printf ("Total Number of Red Pixels = %d\n",totalPixelsR);
    printf ("Total Number of Green Pixels = %d\n",totalPixelsG);
    printf ("Total Number of Blue Pixels = %d\n",totalPixelsB);
    ReleaseBMPImage(&image);
    //free all allocated memory
    free(midDeviceBinR);
    free(midDeviceBinG);
    free(midDeviceBinB);
    free(deviceBinR);
    free(deviceBinG);
    free(deviceBinB);

    return 0;
}