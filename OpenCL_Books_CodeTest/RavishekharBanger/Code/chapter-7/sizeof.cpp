
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include <ocl_macros.h>

#ifdef WIN32
#define ALIGN(X) __declspec( align( X ) )
#else
#define ALIGN(X) __attribute__( (aligned( (X) ) ) )
#endif

#define VECTOR_SIZE 1024
typedef struct
{
    cl_float8 ALIGN(32) y;
    cl_float3 ALIGN(16) x;
} OpenCLStruct;

//OpenCL kernel which is run for every work item created.
// Nvidia OpenCL implementation does not provide printf support hence 
// we create a cl_mem object clSizeMem and write the data to this buffer in 
// inside the kernel. Note that the host side data structure needs explicit 
// alignment of the data buffer

const char *sizeof_kernel =
"typedef struct                                     \n"
"{                                                  \n"
"   float8 y;                                       \n"
"   float3 x;                                       \n"
"} OpenCLStruct;                                    \n"
"__kernel                                           \n"
"void sizeof_kernel(                                \n"
"#ifdef NV_OCL                                      \n"
"   __global int *size                              \n"
"#endif                                             \n"
"                  )                                \n"
"{                                                  \n"
"#ifdef NV_OCL                                      \n"
"    size[get_global_id(0)] = sizeof(OpenCLStruct); \n"
"#else                                              \n"
"    printf(\"The size of the OpenCLStruct provided by the OpenCL compiler is = %d bytes.\\n \",sizeof(OpenCLStruct));   \n"
"#endif                                             \n"
"}                                                  \n";

#define NV_OCL

int main(void) {
    OpenCLStruct* oclStruct = (OpenCLStruct*)malloc(sizeof(OpenCLStruct)*VECTOR_SIZE);
    printf("The size of the OpenCLStruct provided by the host compiler is = %ld bytes\n",sizeof(OpenCLStruct) );

    // Get platform and device information
    cl_platform_id * platforms = NULL;
    cl_uint     num_platforms;
    //Set up the Platform
    cl_int clStatus = clGetPlatformIDs(0, NULL, &num_platforms);
    LOG_OCL_ERROR(clStatus, "clGetPlatformIDs Failed" );
    platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id)*num_platforms);
    clStatus = clGetPlatformIDs(num_platforms, platforms, NULL);
    LOG_OCL_ERROR(clStatus, "clGetPlatformIDs Failed" );

    //Get the devices list and choose the type of device you want to run on
    cl_device_id     *device_list = NULL;
    cl_uint       num_devices;

    clStatus = clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_GPU, 0,
            NULL, &num_devices);
    LOG_OCL_ERROR(clStatus, "clGetDeviceIDs Failed" );
    device_list = (cl_device_id *)malloc(sizeof(cl_device_id)*num_devices);
    clStatus = clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_GPU, num_devices,
            device_list, NULL);
    LOG_OCL_ERROR(clStatus, "clGetDeviceIDs Failed" );

    // Create one OpenCL context for each device in the platform
    cl_context context;
    cl_context_properties props[3] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platforms,
        0
    };

    context = clCreateContext( NULL, num_devices, device_list, NULL, NULL, &clStatus);
    LOG_OCL_ERROR(clStatus, "clCreateContext Failed" );

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_list[0], 0, &clStatus);
    LOG_OCL_ERROR(clStatus, "clCreateCommandQueue Failed" );

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1,
            (const char **)&sizeof_kernel, NULL, &clStatus);
    LOG_OCL_ERROR(clStatus, "clCreateProgramWithSource Failed" );

    // Build the program
#ifdef NV_OCL
    clStatus = clBuildProgram(program, 1, device_list, "-DNV_OCL", NULL, NULL);
#else
    clStatus = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);
#endif
    if(clStatus != CL_SUCCESS)
        LOG_OCL_COMPILER_ERROR(program, device_list[0]);

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "sizeof_kernel", &clStatus);
    LOG_OCL_ERROR(clStatus, "clCreateKernel Failed" );

#ifdef NV_OCL
    cl_mem clSizeMem = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                    sizeof(int), NULL, &clStatus);
    clStatus = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)(&clSizeMem));
#endif

    // Execute the OpenCL kernel. Lauch only one work item to see what is the sizeof OpenCLStruct
    size_t global_size = 1; // Process the entire lists
    size_t local_size  = 1; // Process one item at a time
    clStatus = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
            &global_size, &local_size, 0, NULL, NULL);
    LOG_OCL_ERROR(clStatus, "clEnqueueNDRangeKernel Failed" );

    // Clean up and wait for all the comands to complete.
    clStatus = clFinish(command_queue);
    LOG_OCL_ERROR(clStatus, "clFinish Failed" );

#ifdef NV_OCL
    int size = 0;
    clStatus = clEnqueueReadBuffer(command_queue, clSizeMem, CL_TRUE, 0,
                                   sizeof(int), &size, 0, NULL, NULL);
    LOG_OCL_ERROR(clStatus, "clEnqueueReadBuffer Failed..." );
    printf("The size of the OpenCLStruct provided by the OpenCL compiler is = %d bytes.\n ",size ); 
#endif
    // Finally release all OpenCL allocated objects and host buffers.
    clStatus = clReleaseKernel(kernel);
    clStatus = clReleaseProgram(program);
    clStatus = clReleaseCommandQueue(command_queue);
    clStatus = clReleaseContext(context);
    free(platforms);
    free(device_list);

    return 0;
}
