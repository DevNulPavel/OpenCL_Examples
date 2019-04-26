#include "JPEG_Decoder.hpp"
#include "Framewave.h"

int 
JPEG_Decoder::initialize()
{
    return OCL_SUCCESS;
}

int 
JPEG_Decoder::setupJPEG_Decoder()
{
    return OCL_SUCCESS;
}

int
JPEG_Decoder::setupCL(void)
{
    cl_int status = 0;
    size_t deviceListSize;

    //Get the first available platform. Use it as the default platform
    status = clGetPlatformIDs(1, &platform, NULL);
    LOG_OCL_ERROR(status, "clGetPlatformIDs Failed" );
    cl_device_type dType;
    
    dType = CL_DEVICE_TYPE_CPU;
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
    /* 
     * if device is not set using command line arguments and opencl fails to open a 
     * context on default device GPU then it falls back to CPU 
     */
    if(status != CL_SUCCESS)
    {
        std::cout << "Unsupported GPU device; falling back to CPU ..." << std::endl;
        context = clCreateContextFromType(
                      0, 
                      CL_DEVICE_TYPE_CPU, 
                      NULL, 
                      NULL, 
                      &status);
        LOG_OCL_ERROR(status, "clCreateContextFromType Failed" );
    }

    LOG_OCL_ERROR(status, "clCreateContextFromType Failed" );

    /* First, get the size of device list data */
    status = clGetContextInfo(
                 context, 
                 CL_CONTEXT_DEVICES, 
                 0, 
                 NULL, 
                 &deviceListSize);
    LOG_OCL_ERROR(status, "clGetContextInfo-CL_CONTEXT_DEVICES Failed" );
    //std::cout << "no of devices = "<< deviceListSize <<std::endl;
    //{
    //    cl_uint num_devices;
    //    status = clGetDeviceIDs (NULL,
    //        CL_DEVICE_TYPE_CPU,
    //        0,
    //        NULL,
    //        &num_devices);
    //    std::cout<< " num of devices = " << num_devices << std::endl;
    //}
    /* Now allocate memory for device list based on the size we got earlier */
    devices = (cl_device_id *)malloc(deviceListSize);
    LOG_OCL_ERROR((devices==NULL), "malloc for devices Failed" );


    /* Now, get the device list data */
    status = clGetContextInfo(
                 context, 
                 CL_CONTEXT_DEVICES, 
                 deviceListSize, 
                 devices, 
                 NULL);
    LOG_OCL_ERROR(status, "clGetContextInfo-CL_CONTEXT_DEVICES Failed" );

    /* Get Device specific Information */
    status = clGetDeviceInfo(
            devices[0],
            CL_DEVICE_MAX_WORK_GROUP_SIZE,
            sizeof(size_t),
            (void *)&maxWorkGroupSize,
            NULL);

    LOG_OCL_ERROR(status, "clGetDeviceInfo-CL_DEVICE_MAX_WORK_GROUP_SIZE Failed" );

    status = clGetDeviceInfo(
            devices[0],
            CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
            sizeof(cl_uint),
            (void *)&maxDimensions,
            NULL);

    LOG_OCL_ERROR(status, "clGetDeviceInfo-CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS Failed" );

    maxWorkItemSizes = (size_t *)malloc(maxDimensions*sizeof(size_t));
    
    status = clGetDeviceInfo(
            devices[0],
            CL_DEVICE_MAX_WORK_ITEM_SIZES,
            sizeof(size_t)*maxDimensions,
            (void *)maxWorkItemSizes,
            NULL);

    LOG_OCL_ERROR(status, "clGetDeviceInfo-CL_DEVICE_MAX_WORK_ITEM_SIZES Failed" );

    status = clGetDeviceInfo(
            devices[0],
            CL_DEVICE_LOCAL_MEM_SIZE,
            sizeof(cl_ulong),
            (void *)&totalLocalMemory,
            NULL);

    LOG_OCL_ERROR(status, "clGetDeviceInfo-CL_DEVICE_LOCAL_MEM_SIZE Failed" );

    {
        /* The block is to move the declaration of prop closer to its use */
        cl_command_queue_properties prop = 0;
        if(timing)
            prop |= CL_QUEUE_PROFILING_ENABLE;

        commandQueue = clCreateCommandQueue(
                           context, 
                           devices[0], 
                           prop, 
                           &status);
        LOG_OCL_ERROR(status, "clCreateCommandQueue Failed" );
    }
    
    /*Create Open CL buffers for JPEG Decoder*/
    int sizeOfMCUBuffer = 1;
    
    for (int i=0;i<Image.noOfComponents;i++)
    {
        sizeOfMCUBuffer = Image.componentData[i].noOfMCUsForComponent *
                          Image.noOfxMCU*Image.noOfyMCU*64;
        pMCUdstBuffer[i] = clCreateBuffer(
                      context, 
                      CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                      sizeof(cl_short) * sizeOfMCUBuffer,
                      (void *)Image.pMCUdst[i], /*Fill the pointer to the buffer*/
                      &status);
        LOG_OCL_ERROR(status, "clCreateBuffer Failed" );
        quantInvTableBuffer[i] = clCreateBuffer(
                      context, 
                      CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                      sizeof(cl_ushort) * 64,
                      (void *)&(Image.quantInvTable[Image.componentData[(Image.component[i].ComponentId-1)].DQTTableSelector][0]), /*Fill the pointer to the buffer*/
                      &status);
        LOG_OCL_ERROR(status, "clCreateBuffer Failed" );        
    }
        
    Image.pOutputFinalOpenclDst = (Fw8u *)malloc(sizeof(cl_uchar) * Image.imageWidth * Image.imageHeight * Image.noOfComponents);
    openclOutputBuffer = clCreateBuffer(
                      context, 
                      CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                      sizeof(cl_uchar) * Image.imageWidth * Image.imageHeight * Image.noOfComponents,
                      (void *)Image.pOutputFinalOpenclDst, /*Fill the pointer to the buffer*/
                      &status);
    LOG_OCL_ERROR(status, "clCreateBuffer for openclOutputBuffer Failed" );

    Image.pOutputFinalReferenceDst = (Fw8u*) malloc(Image.imageWidth * Image.imageWidth  * Image.noOfComponents*1/*sizeof (FW8u)*/);
    referenceOutputBuffer = clCreateBuffer(
                      context,
                      CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                      sizeof(cl_uchar) * Image.imageWidth * Image.imageHeight * Image.noOfComponents,
                      (void *)Image.pOutputFinalReferenceDst, /*Fill the pointer to the buffer*/
                      &status);
    LOG_OCL_ERROR(status, "clCreateBuffer for referenceOutputBuffer Failed" );

    Image.pOutputFinalNoOfDevicesDst = (Fw8u*) malloc(Image.imageWidth * Image.imageWidth  * Image.noOfComponents*1/*sizeof (FW8u)*/);
    noOfDevicesOutputBuffer = clCreateBuffer(
                      context,
                      CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                      sizeof(cl_uchar) * Image.imageWidth * Image.imageHeight * Image.noOfComponents,
                      (void *)Image.pOutputFinalNoOfDevicesDst, /*Fill the pointer to the buffer*/
                      &status);
    LOG_OCL_ERROR(status, "clCreateBuffer for noOfDevicesOutputBuffer Failed" );

    /* create a CL program using the kernel source */
    size_t sourceSize[] = { strlen(JPEG_Decoder_kernel) };

    program = clCreateProgramWithSource( 
        context,
        1,
        &JPEG_Decoder_kernel,
        sourceSize,
        &status);
    LOG_OCL_ERROR(status, "clCreateProgramWithSource Failed" );

    /* create a cl program executable for all the devices specified */
    status = clBuildProgram(program, 1, devices, NULL, NULL, NULL);
    LOG_OCL_ERROR(status, "clBuildProgram Failed" );

    /* get a kernel object handle for a kernel with the given name */
    openclKernel = clCreateKernel(program, "JPEGdecoder_MCU", &status);
    LOG_OCL_ERROR(status, "clCreateKernel-JPEGdecoder_MCU Failed" );

    referenceKernel = clCreateKernel(program, "JPEGdecoder_Reference", &status);
    LOG_OCL_ERROR(status, "clCreateKernel-JPEGdecoder_Reference Failed" );

    devicesKernel = clCreateKernel(program, "JPEGdecoder_Devices", &status);
    LOG_OCL_ERROR(status, "clCreateKernel-JPEGdecoder_Devices Failed" );

    return OCL_SUCCESS;
}

int
JPEG_Decoder::runRefCLKernels(void)
{
    cl_int   status;
    cl_event events[2];

    /* 
     * Kernel runs over complete output matrix with blocks of blockSize x blockSize 
     * running concurrently
     */
    size_t globalThreads[2]= {1, 1};
    size_t localThreads[2] = {1, 1};

    long long kernelsStartTime;
    long long kernelsEndTime;

    status =  clGetKernelWorkGroupInfo(
                    referenceKernel,
                    devices[0],
                    CL_KERNEL_LOCAL_MEM_SIZE,
                    sizeof(cl_ulong),
                    &usedLocalMemory,
                    NULL);
    LOG_OCL_ERROR(status, "clGetKernelWorkGroupInfo-CL_KERNEL_LOCAL_MEM_SIZE Failed" );

    availableLocalMemory = totalLocalMemory - usedLocalMemory;
    neededLocalMemory    = 2*blockSize*blockSize*sizeof(cl_float); 
    if(neededLocalMemory > availableLocalMemory)
    {
        std::cout << "Unsupported: Insufficient local memory on device." << std::endl;
        return OCL_SUCCESS;
    } 

    globalThreads[0] = 1;
    globalThreads[1] = 1;
    localThreads[0] = 1;
    localThreads[1] = 1;

    if(localThreads[0]                 > maxWorkItemSizes[0] ||
       localThreads[1]                 > maxWorkItemSizes[1] ||
       localThreads[0]*localThreads[1] > maxWorkGroupSize    )
    {
        std::cout << "Unsupported: Device does not support requested number of work items."<<std::endl;
        return 0;
    }

    /*** Set appropriate arguments to the kernel ***/
    for (int i=0;i<Image.noOfComponents;i++)
    {
        status = clSetKernelArg(
                    referenceKernel, 
                    2*i, 
                    sizeof(cl_mem), 
                    (void *)&pMCUdstBuffer[i]);
        LOG_OCL_ERROR(status, "clSetKernelArg Failed" );
        
        status = clSetKernelArg(
                    referenceKernel, 
                    2*i+1, 
                    sizeof(cl_mem), 
                    (void *)&quantInvTableBuffer[i]);
        LOG_OCL_ERROR(status, "clSetKernelArg Failed" );
    }

    status = clSetKernelArg(
                    referenceKernel,
                    6,
                    sizeof(cl_mem),
                    (void *)&referenceOutputBuffer);
    LOG_OCL_ERROR(status, "clSetKernelArg Failed" );

    /* width of the input image matrix as 8th argument - width */
    status = clSetKernelArg(
                    referenceKernel, 
                    7, 
                    sizeof(cl_int),
                    (void*)&(Image.imageWidth));
    LOG_OCL_ERROR(status, "clSetKernelArg Failed" );

    /* Height of the input image matrix as 8th argument - height */
    status = clSetKernelArg(
                    referenceKernel, 
                    8, 
                    sizeof(cl_int),
                    (void*)&(Image.imageHeight));
    LOG_OCL_ERROR(status, "clSetKernelArg Failed" );

    /* mcuWidth of the input image matrix as 8th argument - mcuWidth */
    status = clSetKernelArg(
                    referenceKernel, 
                    9, 
                    sizeof(cl_int),
                    (void*)&(Image.mcuWidth));
    LOG_OCL_ERROR(status, "clSetKernelArg Failed" );

    /* mcuWidth of the input image matrix as 8th argument - mcuWidth */
    status = clSetKernelArg(
                    referenceKernel, 
                    10, 
                    sizeof(cl_int),
                    (void*)&(Image.mcuHeight));
    LOG_OCL_ERROR(status, "clSetKernelArg Failed" );

    /*Enqueue a kernel run call */ 
    status = clEnqueueNDRangeKernel(
                 commandQueue,
                 referenceKernel,
                 2,
                 NULL,
                 globalThreads,
                 localThreads,
                 0,
                 NULL,
                 &events[0]);
    LOG_OCL_ERROR(status, "clEnqueueNDRangeKernel Failed" );

    /* wait for the kernel call to finish execution */
    status = clWaitForEvents(1, &events[0]); 
    LOG_OCL_ERROR(status, "clWaitForEvents Failed" );

    if(timing)
    {
        status = clGetEventProfilingInfo(
                     events[0],
                     CL_PROFILING_COMMAND_START,
                     sizeof(long long),
                     &kernelsStartTime,
                     NULL);
        LOG_OCL_ERROR(status, "clGetEventProfilingInfo-CL_PROFILING_COMMAND_START Failed" );

        status = clGetEventProfilingInfo(
                     events[0],
                     CL_PROFILING_COMMAND_END,
                     sizeof(long long),
                     &kernelsEndTime,
                     NULL);
        LOG_OCL_ERROR(status, "clGetEventProfilingInfo-CL_PROFILING_COMMAND_END Failed" );

        /* Compute total time (also convert from nanoseconds to seconds) */
        totalKernelTime = (double)(kernelsEndTime - kernelsStartTime)/1e9;
        std::cout << std::endl<< "kernelsEndTime   = " << kernelsEndTime << std::endl;
        std::cout << "kernelsStartTime = " << kernelsStartTime << std::endl;
        std::cout << "Reference Implementation totalKernelTime = " << totalKernelTime << std::endl<< std::endl;
    }

    clReleaseEvent(events[0]);

    /* Enqueue readBuffer*/
    status = clEnqueueReadBuffer(
                commandQueue,
                referenceOutputBuffer,
                CL_TRUE,
                0,
                sizeof(cl_uchar) * Image.imageWidth * Image.imageHeight * Image.noOfComponents,
                Image.pOutputFinalReferenceDst,
                0,
                NULL,
                &events[1]);
    LOG_OCL_ERROR(status, "clEnqueueReadBuffer Failed" );
    
    /* Wait for the read buffer to finish execution */
    status = clWaitForEvents(1, &events[1]);
    LOG_OCL_ERROR(status, "clWaitForEvents Failed" );
    
    clReleaseEvent(events[1]);

    return OCL_SUCCESS;
}

int 
JPEG_Decoder::runCLKernels(void)
{
    cl_int   status;
    cl_event events[2];

    /* 
     * Kernel runs over complete output matrix with blocks of blockSize x blockSize 
     * running concurrently
     */
    size_t globalThreads[2]= {Image.noOfxMCU, Image.noOfyMCU};
    size_t localThreads[2] = {blockSize, blockSize};

    long long kernelsStartTime;
    long long kernelsEndTime;

    status =  clGetKernelWorkGroupInfo(
                    openclKernel,
                    devices[0],
                    CL_KERNEL_LOCAL_MEM_SIZE,
                    sizeof(cl_ulong),
                    &usedLocalMemory,
                    NULL);
    LOG_OCL_ERROR(status, "clGetKernelWorkGroupInfo Failed" );

    availableLocalMemory = totalLocalMemory - usedLocalMemory;

    neededLocalMemory    = 2*blockSize*blockSize*sizeof(cl_float); 

    if(neededLocalMemory > availableLocalMemory)
    {
        std::cout << "Unsupported: Insufficient local memory on device." << std::endl;
        return 0;
    }


    globalThreads[0] = Image.noOfxMCU;
    globalThreads[1] = Image.noOfyMCU;
    localThreads[0] = 1;
    localThreads[1] = 1;

    if(localThreads[0]                 > maxWorkItemSizes[0] ||
       localThreads[1]                 > maxWorkItemSizes[1] ||
       localThreads[0]*localThreads[1] > maxWorkGroupSize    )
    {
        std::cout << "Unsupported: Device does not support requested number of work items."<<std::endl;
        return 0;
    }

    /*** Set appropriate arguments to the kernel ***/

    for (int i=0;i<Image.noOfComponents;i++)
    {
        status = clSetKernelArg(
                    openclKernel, 
                    2*i, 
                    sizeof(cl_mem), 
                    (void *)&pMCUdstBuffer[i]);
        LOG_OCL_ERROR(status, "clSetKernelArg Failed" );

        status = clSetKernelArg(
                    openclKernel, 
                    2*i+1, 
                    sizeof(cl_mem), 
                    (void *)&quantInvTableBuffer[i]);
        LOG_OCL_ERROR(status, "clSetKernelArg Failed" );
    }

    status = clSetKernelArg(
                    openclKernel,
                    6,
                    sizeof(cl_mem),
                    (void *)&openclOutputBuffer);
    LOG_OCL_ERROR(status, "clSetKernelArg Failed" );

    /* width of the input image matrix as 8th argument - width */
    status = clSetKernelArg(
                    openclKernel, 
                    7, 
                    sizeof(cl_int),
                    (void*)&(Image.imageWidth));
    LOG_OCL_ERROR(status, "clSetKernelArg Failed" );

    /* Height of the input image matrix as 8th argument - heigth */
    status = clSetKernelArg(
                    openclKernel, 
                    8, 
                    sizeof(cl_int),
                    (void*)&(Image.imageHeight));
    LOG_OCL_ERROR(status, "clSetKernelArg Failed" );

    /* mcuWidth of the input image matrix as 8th argument - mcuWidth */
    status = clSetKernelArg(
                    openclKernel, 
                    9, 
                    sizeof(cl_int),
                    (void*)&(Image.mcuWidth));
    LOG_OCL_ERROR(status, "clSetKernelArg Failed" );

    /* mcuWidth of the input image matrix as 8th argument - mcuWidth */
    status = clSetKernelArg(
                    openclKernel, 
                    10, 
                    sizeof(cl_int),
                    (void*)&(Image.mcuHeight));
    LOG_OCL_ERROR(status, "clSetKernelArg Failed" );

    /*Enqueue a kernel run call */ 
    status = clEnqueueNDRangeKernel(
                 commandQueue,
                 openclKernel,
                 2,
                 NULL,
                 globalThreads,
                 localThreads,
                 0,
                 NULL,
                 &events[0]);
    LOG_OCL_ERROR(status, "clEnqueueNDRangeKernel Failed" );


    /* wait for the kernel call to finish execution */
    status = clWaitForEvents(1, &events[0]); 
    LOG_OCL_ERROR(status, "clWaitForEvents Failed" );

    if(timing)
    {
        status = clGetEventProfilingInfo(
                     events[0],
                     CL_PROFILING_COMMAND_START,
                     sizeof(long long),
                     &kernelsStartTime,
                     NULL);
        LOG_OCL_ERROR(status, "clGetEventProfilingInfo-CL_PROFILING_COMMAND_START Failed" );

        status = clGetEventProfilingInfo(
                     events[0],
                     CL_PROFILING_COMMAND_END,
                     sizeof(long long),
                     &kernelsEndTime,
                     NULL);
        LOG_OCL_ERROR(status, "clGetEventProfilingInfo-CL_PROFILING_COMMAND_END Failed" );

        /* Compute total time (also convert from nanoseconds to seconds) */
        totalKernelTime = (double)(kernelsEndTime - kernelsStartTime)/1e9;
        std::cout << std::endl << "kernelsEndTime   = " << kernelsEndTime << std::endl;
        std::cout << "kernelsStartTime = " << kernelsStartTime << std::endl;
        std::cout << "Opencl  Implementation totalKernelTime = " << totalKernelTime << std::endl << std::endl;
    }

    clReleaseEvent(events[0]);

    /* Enqueue readBuffer*/
    status = clEnqueueReadBuffer(
                commandQueue,
                openclOutputBuffer,
                CL_TRUE,
                0,
                sizeof(cl_uchar) * Image.imageWidth * Image.imageHeight * Image.noOfComponents,
                Image.pOutputFinalOpenclDst,
                0,
                NULL,
                &events[1]);
    LOG_OCL_ERROR(status, "clEnqueueReadBuffer Failed" );
    
    /* Wait for the read buffer to finish execution */
    status = clWaitForEvents(1, &events[1]);
    LOG_OCL_ERROR(status, "clWaitForEvents Failed" );

    clReleaseEvent(events[1]);

    return OCL_SUCCESS;
}

int
JPEG_Decoder::runOnDevicesCLKernels(void)
{
    cl_int   status;
    cl_event events[2];

    /* 
     * Kernel runs over complete output matrix with blocks of blockSize x blockSize 
     * running concurrently
     */
    size_t globalThreads[2]= {2, 1};
    size_t localThreads[2] = {1, 1};

    long long kernelsStartTime;
    long long kernelsEndTime;

    status =  clGetKernelWorkGroupInfo(
                    referenceKernel,
                    devices[0],
                    CL_KERNEL_LOCAL_MEM_SIZE,
                    sizeof(cl_ulong),
                    &usedLocalMemory,
                    NULL);
    LOG_OCL_ERROR(status, "clGetKernelWorkGroupInfo Failed" );

    availableLocalMemory = totalLocalMemory - usedLocalMemory;

    neededLocalMemory    = 2*blockSize*blockSize*sizeof(cl_float); 

    if(neededLocalMemory > availableLocalMemory)
    {
        std::cout << "Unsupported: Insufficient local memory on device." << std::endl;
        return OCL_SUCCESS;
    } 


    globalThreads[0] = 1;
    globalThreads[1] = 2;
    localThreads[0] = 1;
    localThreads[1] = 1;

    if(localThreads[0]                 > maxWorkItemSizes[0] ||
       localThreads[1]                 > maxWorkItemSizes[1] ||
       localThreads[0]*localThreads[1] > maxWorkGroupSize    )
    {
        std::cout << "Unsupported: Device does not support requested number of work items."<<std::endl;
        return OCL_SUCCESS;
    }

    /*** Set appropriate arguments to the kernel ***/



    for (int i=0;i<Image.noOfComponents;i++)
    {
        status = clSetKernelArg(
                    devicesKernel, 
                    2*i, 
                    sizeof(cl_mem), 
                    (void *)&pMCUdstBuffer[i]);
        LOG_OCL_ERROR(status, "clSetKernelArg Failed" );
            //return OCL_FAILURE; /*You shud not break from Inside the for loop*/
        status = clSetKernelArg(
                    devicesKernel, 
                    2*i+1, 
                    sizeof(cl_mem), 
                    (void *)&quantInvTableBuffer[i]);
        LOG_OCL_ERROR(status, "clSetKernelArg Failed" );
    }

    status = clSetKernelArg(
                    devicesKernel,
                    6,
                    sizeof(cl_mem),
                    (void *)&noOfDevicesOutputBuffer);
    LOG_OCL_ERROR(status, "clSetKernelArg Failed" );

    /* width of the input image matrix as 8th argument - width */
    status = clSetKernelArg(
                    devicesKernel, 
                    7, 
                    sizeof(cl_int),
                    (void*)&(Image.imageWidth));
    LOG_OCL_ERROR(status, "clSetKernelArg Failed" );

    /* Height of the input image matrix as 8th argument - heigth */
    status = clSetKernelArg(
                    devicesKernel, 
                    8, 
                    sizeof(cl_int),
                    (void*)&(Image.imageHeight));
    LOG_OCL_ERROR(status, "clSetKernelArg Failed" );

    /* mcuWidth of the input image matrix as 8th argument - mcuWidth */
    status = clSetKernelArg(
                    devicesKernel, 
                    9, 
                    sizeof(cl_int),
                    (void*)&(Image.mcuWidth));
    LOG_OCL_ERROR(status, "clSetKernelArg Failed" );

    /* mcuWidth of the input image matrix as 8th argument - mcuWidth */
    status = clSetKernelArg(
                    devicesKernel, 
                    10, 
                    sizeof(cl_int),
                    (void*)&(Image.mcuHeight));
    LOG_OCL_ERROR(status, "clSetKernelArg Failed" );

    /*Enqueue a kernel run call */ 
    status = clEnqueueNDRangeKernel(
                 commandQueue,
                 devicesKernel,
                 2,
                 NULL,
                 globalThreads,
                 localThreads,
                 0,
                 NULL,
                 &events[0]);
    LOG_OCL_ERROR(status, "clEnqueueNDRangeKernel Failed" );


    /* wait for the kernel call to finish execution */
    status = clWaitForEvents(1, &events[0]); 
    LOG_OCL_ERROR(status, "clWaitForEvents Failed" );

    if(timing)
    {
        status = clGetEventProfilingInfo(
                     events[0],
                     CL_PROFILING_COMMAND_START,
                     sizeof(long long),
                     &kernelsStartTime,
                     NULL);
        LOG_OCL_ERROR(status, "clGetEventProfilingInfo Failed" );

        status = clGetEventProfilingInfo(
                     events[0],
                     CL_PROFILING_COMMAND_END,
                     sizeof(long long),
                     &kernelsEndTime,
                     NULL);
        LOG_OCL_ERROR(status, "clGetEventProfilingInfo Failed" );

        /* Compute total time (also convert from nanoseconds to seconds) */
        totalKernelTime = (double)(kernelsEndTime - kernelsStartTime)/1e9;
        std::cout << std::endl << "kernelsEndTime   = " << kernelsEndTime << std::endl;
        std::cout << "kernelsStartTime = " << kernelsStartTime << std::endl;
        std::cout << "No Of Devices Implementation totalKernelTime = " << totalKernelTime << std::endl<< std::endl;
    }

    clReleaseEvent(events[0]);

    /* Enqueue readBuffer*/
    status = clEnqueueReadBuffer( 
                commandQueue,
                noOfDevicesOutputBuffer,
                CL_TRUE,
                0,
                sizeof(cl_uchar) * Image.imageWidth * Image.imageHeight * Image.noOfComponents,
                Image.pOutputFinalNoOfDevicesDst,
                0,
                NULL,
                &events[1]);

    LOG_OCL_ERROR(status, "clEnqueueReadBuffer Failed" );
    
    /* Wait for the read buffer to finish execution */
    status = clWaitForEvents(1, &events[1]);
    LOG_OCL_ERROR(status, "clWaitForEvents Failed" );
    
    clReleaseEvent(events[1]);
    return OCL_SUCCESS;
}

int 
JPEG_Decoder::setup()
{
    if(setupJPEG_Decoder()!=OCL_SUCCESS)
        return OCL_FAILURE;


    if(setupCL()!=OCL_SUCCESS)
        return -1;

    /*int timer = sampleCommon->createTimer();
    sampleCommon->resetTimer(timer);
    sampleCommon->startTimer(timer);
    sampleCommon->stopTimer(timer);

    setupTime = (cl_double)sampleCommon->readTimer(timer);*/

    return 0;
}

int 
JPEG_Decoder::run()
{
    /* Arguments are set and execution call is enqueued on command buffer */
    if(runCLKernels()!=OCL_SUCCESS)
        return OCL_FAILURE;
    return OCL_SUCCESS;
}

int 
JPEG_Decoder::runRef()
{
    /* Arguments are set and execution call is enqueued on command buffer */
    if(runRefCLKernels()!=OCL_SUCCESS)
        return OCL_FAILURE;
    return OCL_SUCCESS;
}

int 
JPEG_Decoder::runOnDevices()
{
    /* Arguments are set and execution call is enqueued on command buffer */
    if(runOnDevicesCLKernels()!=OCL_SUCCESS)
        return OCL_FAILURE;
    return OCL_SUCCESS;
}

int 
JPEG_Decoder::verifyResults()
{
    return OCL_SUCCESS;
}

void 
JPEG_Decoder:: printStats()
{
    std::string strArray[1] = {"JPEG Decoder Data"};
    std::string stats[1];

    totalTime = setupTime + totalKernelTime;
}

int 
JPEG_Decoder::cleanup()
{
    /* Releases OpenCL resources (Context, Memory etc.) */
    cl_int status;

    status = clReleaseKernel(openclKernel);
    LOG_OCL_ERROR(status, "clReleaseKernel Failed" );

    status = clReleaseKernel(referenceKernel);
    LOG_OCL_ERROR(status, "clReleaseKernel Failed" );

    status = clReleaseKernel(devicesKernel);
    LOG_OCL_ERROR(status, "clReleaseKernel Failed" );


    status = clReleaseProgram(program);
    LOG_OCL_ERROR(status, "clReleaseProgram Failed" );
 
    for (int i=0;i<Image.noOfComponents;i++)
    {
        status = clReleaseMemObject(pMCUdstBuffer[i]);
        LOG_OCL_ERROR(status, "clReleaseMemObject Failed" );
    }

    status = clReleaseMemObject(openclOutputBuffer);
    LOG_OCL_ERROR(status, "clReleaseMemObject Failed" );

    status = clReleaseMemObject(referenceOutputBuffer);
    LOG_OCL_ERROR(status, "clWaitForEvents Failed" );

    status = clReleaseCommandQueue(commandQueue);
    LOG_OCL_ERROR(status, "clReleaseCommandQueue Failed" );

    status = clReleaseContext(context);
    LOG_OCL_ERROR(status, "clReleaseContext Failed" );

    /* release program resources (input memory etc.) */
    for(int i=0; i<Image.noOfComponents; i++)
    {
        if(Image.pMCUdst[i]) 
            free(Image.pMCUdst[i]);
    }

    if(Image.pOutputFinalOpenclDst) 
        free(Image.pOutputFinalOpenclDst);


    /* release device list */
    if(devices)
        free(devices);

    if(maxWorkItemSizes)
        free(maxWorkItemSizes);

    return OCL_SUCCESS;
}

int 
JPEG_Decoder::decodeImage(const char *fileName)
{
    if(Image.open(fileName) == OCL_FAILURE)
    {
        std::cout<<"File lena.jpg is not available......\n";
        std::cout<<"Copy lena.jpg file from the input_images folder to the location where the executable is generated\n";
        exit(-1);
    }
    
    Image.decode(); //This parses all the MCUs and stores it in a buffer
    Image.close();
    /*First get the reference implementations data*/
    if(setup()!= OCL_SUCCESS)
        return OCL_FAILURE;


    std::cout << "*****************************************\n";
    std::cout << "Running Reference Implementation\n";
    if(runRef()!= OCL_SUCCESS)
        return OCL_FAILURE;
    Image.write("referenceOutput.bmp", Image.pOutputFinalReferenceDst);
    std::cout << "Decoded output file written to referenceOutput.bmp\n";


    std::cout << "*****************************************\n";
    std::cout << "Running Devices Implementation\n";
    if(runOnDevices()!= OCL_SUCCESS)
        return OCL_FAILURE;	
    Image.write("noOfDevicesOutput.bmp", Image.pOutputFinalNoOfDevicesDst);
    std::cout << "Decoded output file written to noOfDevicesOutput.bmp\n";


    /*Now get the Opencl implementation*/
    std::cout << "*****************************************\n";
    std::cout << "Running OpenCL Implementation\n";    
    if(run()!=OCL_SUCCESS)
        return OCL_FAILURE;
    Image.write("openclOutput.bmp", Image.pOutputFinalOpenclDst);
    std::cout << "Decoded output file written to openclOutput.bmp\n";
    std::cout << "*****************************************\n";


    if(cleanup()!=OCL_SUCCESS)
        return OCL_FAILURE;
    return OCL_SUCCESS;
}

int 
main(int argc, char * argv[])
{
    //char *inputFile = "lena.jpg";
    const char *inputFile = "lena.jpg";

    JPEG_Decoder  clJPEG_Decoder("OpenCL JPEG Decoder");
    clJPEG_Decoder.initialize();

    if(clJPEG_Decoder.decodeImage(inputFile) != OCL_SUCCESS)
        return OCL_FAILURE;
    clJPEG_Decoder.printStats();

    return OCL_SUCCESS;
}
