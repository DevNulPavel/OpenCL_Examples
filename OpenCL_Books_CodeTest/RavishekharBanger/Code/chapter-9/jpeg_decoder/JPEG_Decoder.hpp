
#ifndef JPEG_DECODER_H_
#define JPEG_DECODER_H_


#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <ocl_macros.h>
#include "image.h"
#include "JPEG_Decoder_Kernels.h"
#define OCL_SUCCESS 0
#define OCL_FAILURE -1
/**
 * JPEG_Decoder 
 * Class implements OpenCL JPEG_Decoder sample
 * Derived from SDKSample base class
 */

class JPEG_Decoder 
{
    cl_uint                  seed;      /**< Seed value for random number generation */
    cl_double           setupTime;      /**< Time for setting up OpenCL */
    cl_double     totalKernelTime;      /**< Time for kernel execution */
    cl_double    totalProgramTime;      /**< Time for program execution */
    cl_double referenceKernelTime;      /**< Time for reference implementation */

    cl_int                 width0;      /**< width of input Array */
    cl_int                height0;      /**< height of input Array */
    cl_uint             blockSize;      /**< Size of the block used for shared memory */
 
/**/cl_context            context;      /**< CL context */
/**/cl_device_id         *devices;      /**< CL device list */
/**/cl_command_queue commandQueue;      /**< CL command queue */
    cl_platform_id       platform;      /**< CL platform  */
    cl_program            program;      /**< CL program  */
    cl_kernel        openclKernel;      /**< Open CL kernel */
    cl_kernel     referenceKernel;      /**< Reference CL kernel */
    cl_kernel       devicesKernel;  /*Kernel equivalent to the number of devices*/
    cl_mem		 pMCUdstBuffer[4];
    cl_mem quantInvTableBuffer[4]; //
    cl_mem     openclOutputBuffer;
    cl_mem	referenceOutputBuffer;
    cl_mem	noOfDevicesOutputBuffer;
    cl_int                      n;      /**< for command line args */
    cl_int                      m;
    cl_int                      k;
    cl_int                 timing;
    cl_double           totalTime;
    
/**/size_t       maxWorkGroupSize;      /**< Device Specific Information */
/**/cl_uint         maxDimensions;
/**/size_t *     maxWorkItemSizes;
/**/cl_ulong     totalLocalMemory; 
/**/cl_ulong      usedLocalMemory; 
/**/cl_ulong availableLocalMemory; 
    cl_ulong    neededLocalMemory;
    image       Image;
public:
    /** 
     * Constructor 
     * Initialize member variables
     * @param name name of sample (string)
     */
    JPEG_Decoder(std::string name)
        {
            seed   = 123;
            n = 64;
            m = 64;
            k = 64;
            blockSize = 8;
            setupTime = 0;
            totalKernelTime = 0;
        }

    /** 
     * Constructor 
     * Initialize member variables
     * @param name name of sample (const char*)
     */
    JPEG_Decoder(const char* name)
        {
            seed   = 123;
            n = 64;
            m = 64;
            k = 64;
            blockSize = 8;
            setupTime = 0;
            totalKernelTime = 0;
            //timing = 1;
        }

    /**
     * Allocate and initialize host memory array with random values
     * @return 1 on success and 0 on failure
     */
    int setupJPEG_Decoder();

    /**
     * OpenCL related initialisations. 
     * Set up Context, Device list, Command Queue, Memory buffers
     * Build CL kernel program executable
     * @return 1 on success and 0 on failure
     */
    int setupCL();

    /**
     * Set values for kernels' arguments, enqueue calls to the kernels
     * on to the command queue, wait till end of kernel execution.
     * Get kernel start and end time if timing is enabled
     * @return 1 on success and 0 on failure
     */
    int runCLKernels();

    /*runs the reference code*/
    int runRefCLKernels();

    int genBinaryImage() { return 0;};

    int runOnDevicesCLKernels();

    /**
     * Override from SDKSample. Print sample stats.
     */
    void printStats();

    /**
     * Override from SDKSample. Initialize 
     * command line parser, add custom options
     */
    int initialize();

    /**
     * Override from SDKSample, adjust width and height 
     * of execution domain, perform all sample setup
     */
    int setup();

    /**
     * Override from SDKSample
     * Run OpenCL JPEG Decoder
     */
    int run();

    /*Run the reference Code*/
    int runRef();
    
    /*Run based on the number of cores in the system*/
    int runOnDevices();

    /**
     * Override from SDKSample
     * Cleanup memory allocations
     */
    int cleanup();

    /**
     * Override from SDKSample
     * Verify against reference implementation
     */
    int verifyResults();

    int decodeImage(const char *fName);
};



#endif
