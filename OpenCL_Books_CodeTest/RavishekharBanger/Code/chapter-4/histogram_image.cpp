#include <cstdlib>
#include <iostream>
#include <string.h>
#include <stdio.h>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include <bmp_image.h>
#include <ocl_macros.h>


const char *histogram_image_kernel =
"#define BIN_SIZE 256                                                                                             \n"
"#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable                                                  \n"
"__kernel                                                                                                         \n"
"void histogram_image_kernel(__read_only image2d_t image,                                                         \n"
"                  __local uchar* sharedArray,                                                                    \n"
"                  __global uint* binResultR,                                                                     \n"
"                  __global uint* binResultG,                                                                     \n"
"                  __global uint* binResultB,                                                                     \n"
"                           uint  blockWidth,                                                                     \n"
"                           uint  blockHeight)                                                                    \n"
"{                                                                                                                \n"
"    size_t localIdX = get_local_id(0);                                                                           \n"
"    size_t localIdY = get_local_id(1);                                                                           \n"
"    size_t localSizeX = get_local_size(0);                                                                       \n"
"    size_t localSizeY = get_local_size(1);                                                                       \n"
"    size_t globalIdX  = get_global_id(0);                                                                        \n"
"    size_t globalIdY  = get_global_id(1);                                                                        \n"
"    size_t groupIdX   = get_group_id(0);                                                                         \n"
"    size_t groupIdY   = get_group_id(1);                                                                         \n"
"    size_t totalGroupSize  = get_local_size(0) * get_local_size(1);                                              \n"
"    size_t groupSizeX = get_global_size(0)/get_local_size(0);                                                    \n"
"     __local uchar* sharedArrayR = sharedArray;                                                                  \n"
"     __local uchar* sharedArrayG = sharedArray + totalGroupSize * BIN_SIZE;                                      \n"
"     __local uchar* sharedArrayB = sharedArray + 2 * totalGroupSize * BIN_SIZE;                                  \n"
"     sampler_t smp = CLK_ADDRESS_REPEAT | CLK_FILTER_NEAREST;                                                    \n"
"    uint sharedArrayOffset = localIdY * localSizeX + localIdX;                                                   \n"
"    /* initialize shared array to zero */                                                                        \n"
"    for(int i = 0; i < BIN_SIZE; ++i)                                                                            \n"
"    {                                                                                                            \n"
"        sharedArrayR[sharedArrayOffset * BIN_SIZE + i] = 0;                                                      \n"
"        sharedArrayG[sharedArrayOffset * BIN_SIZE + i] = 0;                                                      \n"
"        sharedArrayB[sharedArrayOffset * BIN_SIZE + i] = 0;                                                      \n"
"    }                                                                                                            \n"
"                                                                                                                 \n"
"    /* calculate the histograms */                                                                               \n"
"    int xCoord = globalIdX*blockWidth;                                                                           \n"
"    int yCoord = globalIdY*blockHeight;                                                                          \n"
"    for(int i = 0; i < blockHeight; ++i)                                                                         \n"
"    {                                                                                                            \n"
"       for(int j = 0; j < blockWidth; ++j)                                                                       \n"
"       {                                                                                                         \n"
"           int pixelCoordX  = xCoord+j;                                                                          \n"
"           int pixelCoordY  = yCoord+i;                                                                          \n"
"           uint4 pixelValue = read_imageui(image, smp, (int2)(pixelCoordX, pixelCoordY));                        \n"
"           uint valueR = pixelValue.x;                                                                           \n"
"           uint valueG = pixelValue.y;                                                                           \n"
"           uint valueB = pixelValue.z;                                                                           \n"
"           sharedArrayR[sharedArrayOffset * BIN_SIZE + valueR]++;                                                \n"
"           sharedArrayG[sharedArrayOffset * BIN_SIZE + valueG]++;                                                \n"
"           sharedArrayB[sharedArrayOffset * BIN_SIZE + valueB]++;                                                \n"
"       }                                                                                                         \n"
"    }                                                                                                            \n"
"    barrier(CLK_LOCAL_MEM_FENCE);                                                                                \n"
"                                                                                                                 \n"
"    uint numOfElements = BIN_SIZE/totalGroupSize;                                                                \n"
"    uint offsetforWI = (localIdY*localSizeX + localIdX)*numOfElements;                                           \n"
"    for(int i = 0; i < numOfElements; ++i)                                                                       \n"
"    {                                                                                                            \n"
"       int binCountR = 0;                                                                                        \n"
"       int binCountG = 0;                                                                                        \n"
"       int binCountB = 0;                                                                                        \n"
"       for(int k = 0; k < totalGroupSize; ++k)                                                                   \n"
"       {                                                                                                         \n"
"           int localOffset = k*BIN_SIZE + offsetforWI;                                                           \n"
"           binCountR += sharedArrayR[localOffset + i];                                                           \n"
"           binCountG += sharedArrayG[localOffset + i];                                                           \n"
"           binCountB += sharedArrayB[localOffset + i];                                                           \n"
"       }                                                                                                         \n"
"       uint WGBinOffset = groupIdY * groupSizeX + groupIdX;                                                      \n"
"       binResultR[WGBinOffset * BIN_SIZE + offsetforWI + i] = binCountR;                                         \n"
"       binResultG[WGBinOffset * BIN_SIZE + offsetforWI + i] = binCountG;                                         \n"
"       binResultB[WGBinOffset * BIN_SIZE + offsetforWI + i] = binCountB;                                         \n"
"    }                                                                                                            \n"
"                                                                                                                 \n"
"                                                                                                                 \n"
"}                                                                                                                \n"
"                                                                                                                 \n";


int main(int argc, char *argv[])
{
    cl_int status = 0;
    cl_int binSize = 256; //The number of possibles bins for the pixel value. Since we are using a unsigned cahar type to represent the pixel
    cl_int groupSize = 16;//4*4, We will spawn a 2D NDRange
    cl_int blockWidth  = 16; //Each Work Item computes a 16X16 element = blockWidth X blockHeight
    cl_int blockHeight = 16; 
    cl_int subHistgCnt;
    cl_device_type dType = CL_DEVICE_TYPE_GPU;
    cl_platform_id platform = NULL;
    cl_device_id   device;
    cl_context     context;
    cl_command_queue commandQueue;
    //cl_mem  imageBuffer;
    cl_mem  intermediateHistR, intermediateHistG, intermediateHistB; /*Intermediate Image Histogram buffer*/
    cl_uint *midDeviceBinR, *midDeviceBinG, *midDeviceBinB;
    cl_uint *deviceBinR,*deviceBinG,*deviceBinB;
    //Read a BMP Image
    Image *image;
    std::string filename = "sample_color.bmp";
    ReadBMPImage(filename, &image);
    if(image == NULL)
    {
        printf("Copy the file sample_color.bmp from the folder input_images. And then run again.");
        return 0;
    }
    subHistgCnt  = (image->width * image->height)/(groupSize*blockWidth*blockHeight);
    midDeviceBinR = (cl_uint*)malloc(binSize * subHistgCnt * sizeof(cl_uint));
    midDeviceBinG = (cl_uint*)malloc(binSize * subHistgCnt * sizeof(cl_uint));
    midDeviceBinB = (cl_uint*)malloc(binSize * subHistgCnt * sizeof(cl_uint));
    deviceBinR    = (cl_uint*)malloc(binSize * sizeof(cl_uint));
    deviceBinG    = (cl_uint*)malloc(binSize * sizeof(cl_uint));
    deviceBinB    = (cl_uint*)malloc(binSize * sizeof(cl_uint));
    
    //Setup the OpenCL Platform,
    //Get the first available platform. Use it as the default platform
    status = clGetPlatformIDs(1, &platform, NULL);
    LOG_OCL_ERROR(status, "clGetPlatformIDs failed." );

    //Get the first available device
    status = clGetDeviceIDs (platform, dType, 1, &device, NULL);
    LOG_OCL_ERROR(status, "clGetDeviceIDs failed." );
    
    //Create an execution context for the selected platform and device.
    cl_context_properties contextProperty[3] = 
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platform,
        0
    };
    context = clCreateContextFromType(
        contextProperty,
        dType,
        NULL,
        NULL,
        &status);
    LOG_OCL_ERROR(status, "clCreateContextFromType Failed" );

    // Create command queue
    commandQueue = clCreateCommandQueue(context,
                                        device,
                                        0,
                                        &status);
    LOG_OCL_ERROR(status, "clCreateCommandQueue Failed" );

    //Create OpenCL device input image
    
    cl_image_format image_format;
    image_format.image_channel_data_type = CL_UNSIGNED_INT8;
    image_format.image_channel_order = CL_RGBA;
#ifdef OPENCL_1_2
    cl_image_desc image_desc;
    image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    image_desc.image_width = image->width;
    image_desc.image_height = image->height;
    image_desc.image_depth = 1;
    image_desc.image_array_size = 1;
    image_desc.image_row_pitch = image->width * 4; //RGBA
    image_desc.image_slice_pitch = image->width * image->height * 4;
    image_desc.num_mip_levels = 0;
    image_desc.num_samples = 0;
    image_desc.buffer= NULL;
#endif

#ifdef OPENCL_1_2
    cl_mem clImage = clCreateImage(
#else
    cl_mem clImage = clCreateImage2D(
#endif
        context,
        CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR,
        &image_format,
#ifdef OPENCL_1_2
        &image_desc,
#else
        image->width, image->height, image->width *4,
#endif
        image->pixels, //R,G,B,A, R,G,B,A, R,G,B,A
        &status); 
    LOG_OCL_ERROR(status, "clCreateImage Failed" );

    //Create OpenCL device output buffer
    intermediateHistR = clCreateBuffer(
        context, 
        CL_MEM_WRITE_ONLY,
        sizeof(cl_uint) * binSize * subHistgCnt, 
        NULL, 
        &status);
    LOG_OCL_ERROR(status, "clCreateBuffer Failed" );

    intermediateHistG = clCreateBuffer(
        context,
        CL_MEM_WRITE_ONLY,
        sizeof(cl_uint) * binSize * subHistgCnt,
        NULL,
        &status);
    LOG_OCL_ERROR(status, "clCreateBuffer Failed" );

    intermediateHistB = clCreateBuffer(
        context,
        CL_MEM_WRITE_ONLY,
        sizeof(cl_uint) * binSize * subHistgCnt,
        NULL,
        &status);
    LOG_OCL_ERROR(status, "clCreateBuffer Failed" );

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1,
            (const char **)&histogram_image_kernel, NULL, &status);
    LOG_OCL_ERROR(status, "clCreateProgramWithSource Failed" );

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
    cl_kernel kernel = clCreateKernel(program, "histogram_image_kernel", &status);

    // Set the arguments of the kernel
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&clImage); 
    status |= clSetKernelArg(kernel, 1, 3 * groupSize * binSize * sizeof(cl_uchar), NULL); 
    status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&intermediateHistR);
    status |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&intermediateHistG);
    status |= clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&intermediateHistB);
    status |= clSetKernelArg(kernel, 5, sizeof(cl_int), (void*)&blockWidth);
    status |= clSetKernelArg(kernel, 6, sizeof(cl_int), (void*)&blockHeight);
    LOG_OCL_ERROR(status, "clCreateContext failed." );

    // Execute the OpenCL kernel on the list
    cl_event ndrEvt;
    size_t localThreads[2]  = {4,4};
    size_t globalThreads[2] = {image->width/blockWidth, image->height/blockHeight};
    status = clEnqueueNDRangeKernel(
        commandQueue,
        kernel,
        2,
        NULL,
        globalThreads,
        localThreads,
        0,
        NULL,
        &ndrEvt);
    LOG_OCL_ERROR(status, "clEnqueueNDRangeKernel Failed" );

    status = clFlush(commandQueue);
    LOG_OCL_ERROR(status, "clFlush Failed" );
    //Read the histogram back into the host memory.
    memset(deviceBinR, 0, binSize * sizeof(cl_uint) );
    memset(deviceBinG, 0, binSize * sizeof(cl_uint) );
    memset(deviceBinB, 0, binSize * sizeof(cl_uint) );
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
    LOG_OCL_ERROR(status, "clEnqueueReadBuffer Failed" );
    
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
    LOG_OCL_ERROR(status, "clEnqueueReadBuffer Failed" );
    
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
    LOG_OCL_ERROR(status, "clEnqueueReadBuffer Failed" );
    
    status = clWaitForEvents(3, readEvt);
    //status = clFinish(commandQueue);
    LOG_OCL_ERROR(status, "clWaitForEvents Failed" );

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
    printf ("\nTotal Number of Red Pixels = %d\n",totalPixelsR);
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
