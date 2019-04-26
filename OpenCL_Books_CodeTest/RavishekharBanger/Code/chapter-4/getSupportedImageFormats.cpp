#include <stdio.h>
#include <stdlib.h>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include <ocl_macros.h>

#define VECTOR_SIZE 1024
#ifdef WIN32
#define ALIGN(X) __declspec( align( (X) ) )
#else
#define ALIGN(X) __attribute__( (aligned( (X) ) ) )
#endif

const char *    getStrChannelOrder(cl_channel_order channel_order)
{
    const char *returnStr;
    switch(channel_order)
    {
        case CL_R          :     returnStr = "CL_R        "; break;
        case CL_A          :     returnStr = "CL_A        "; break;
        case CL_RG         :     returnStr = "CL_RG       "; break;
        case CL_RA         :     returnStr = "CL_RA       "; break;
        case CL_RGB        :     returnStr = "CL_RGB      "; break;
        case CL_RGBA       :     returnStr = "CL_RGBA     "; break;
        case CL_BGRA       :     returnStr = "CL_BGRA     "; break;
        case CL_ARGB       :     returnStr = "CL_ARGB     "; break;
        case CL_INTENSITY  :     returnStr = "CL_INTENSITY"; break;
        case CL_LUMINANCE  :     returnStr = "CL_LUMINANCE"; break;
        case CL_Rx         :     returnStr = "CL_Rx       "; break;
        case CL_RGx        :     returnStr = "CL_RGx      "; break;
        case CL_RGBx       :     returnStr = "CL_RGBx     "; break;
        default            :     returnStr = "NULL     ";    break;
    }                                                    
    return returnStr;
}
const char *    getStrChannelDataType(cl_channel_type channel_type)
{
    const char *returnStr;
    switch(channel_type)
    {
        case CL_SNORM_INT8       :   returnStr = "CL_SNORM_INT8      "; break;
        case CL_SNORM_INT16      :   returnStr = "CL_SNORM_INT16     "; break;
        case CL_UNORM_INT8       :   returnStr = "CL_UNORM_INT8      "; break;
        case CL_UNORM_INT16      :   returnStr = "CL_UNORM_INT16     "; break;
        case CL_UNORM_SHORT_565  :   returnStr = "CL_UNORM_SHORT_565 "; break;
        case CL_UNORM_SHORT_555  :   returnStr = "CL_UNORM_SHORT_555 "; break;
        case CL_UNORM_INT_101010 :   returnStr = "CL_UNORM_INT_101010"; break;
        case CL_SIGNED_INT8      :   returnStr = "CL_SIGNED_INT8     "; break;
        case CL_SIGNED_INT16     :   returnStr = "CL_SIGNED_INT16    "; break;
        case CL_SIGNED_INT32     :   returnStr = "CL_SIGNED_INT32    "; break;
        case CL_UNSIGNED_INT8    :   returnStr = "CL_UNSIGNED_INT8   "; break;
        case CL_UNSIGNED_INT16   :   returnStr = "CL_UNSIGNED_INT16  "; break;
        case CL_UNSIGNED_INT32   :   returnStr = "CL_UNSIGNED_INT32  "; break;
        case CL_HALF_FLOAT       :   returnStr = "CL_HALF_FLOAT      "; break;
        case CL_FLOAT            :   returnStr = "CL_FLOAT           "; break;
        default                  :   returnStr = "NULL     ";           break;
    }
    return returnStr;
}


void print_image_format(cl_image_format image_format)
{
    cl_channel_order image_channel_order;
    cl_channel_type image_channel_data_type;
    image_channel_order = image_format.image_channel_order;
    image_channel_data_type = image_format.image_channel_data_type;
    printf("%s \t\t%s\n",getStrChannelOrder(image_channel_order), getStrChannelDataType(image_channel_data_type));
}

int main(void) {
    // Get platform and device information
    cl_platform_id * platforms = NULL;
    cl_uint     num_platforms;
    //Set up the Platform
    cl_int clStatus = clGetPlatformIDs(0, NULL, &num_platforms);
    platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id)*num_platforms);
    clStatus = clGetPlatformIDs(num_platforms, platforms, NULL);
    LOG_OCL_ERROR(clStatus, "clGetPlatformIDs failed." );

    //Get the devices list and choose the type of device you want to run on
    cl_device_id     *device_list = NULL;
    cl_uint       num_devices;

    clStatus = clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_GPU, 0,
            NULL, &num_devices);
    LOG_OCL_ERROR(clStatus, "clGetDeviceIDs failed while retreiving the number of GPUs available." );

    device_list = (cl_device_id *)malloc(sizeof(cl_device_id)*num_devices);
    clStatus = clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_GPU, num_devices,
            device_list, NULL);
    LOG_OCL_ERROR(clStatus, "clGetDeviceIDs failed." );

    // Create one OpenCL context for each device in the platform
    cl_context context;
    cl_context_properties props[3] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platforms,
        0
    };

    context = clCreateContext( NULL, num_devices, device_list, NULL, NULL, &clStatus);
    LOG_OCL_ERROR(clStatus, "clCreateContext failed." );

    cl_mem_object_type image_type[6] = {
#ifdef OPENCL_1_2
        CL_MEM_OBJECT_IMAGE1D,
        CL_MEM_OBJECT_IMAGE1D_BUFFER, 
        CL_MEM_OBJECT_IMAGE1D_ARRAY,
#endif
        CL_MEM_OBJECT_IMAGE2D, 
#ifdef OPENCL_1_2
        CL_MEM_OBJECT_IMAGE2D_ARRAY,
#endif
        CL_MEM_OBJECT_IMAGE3D
    };
    cl_image_format *image_formats;
    cl_uint num_image_formats;
    clStatus= clGetSupportedImageFormats (context,
                CL_MEM_READ_ONLY|CL_MEM_READ_WRITE,
                CL_MEM_OBJECT_IMAGE2D,
                0,
                NULL,
                &num_image_formats);
    LOG_OCL_ERROR(clStatus, "clGetSupportedImageFormats failed." );

    image_formats = (cl_image_format *)malloc(sizeof(cl_image_format) * num_image_formats);
    clStatus= clGetSupportedImageFormats (context,
                CL_MEM_READ_ONLY,
#ifdef OPENCL_1_2				
                CL_MEM_OBJECT_IMAGE1D,
#else
                CL_MEM_OBJECT_IMAGE2D,
#endif
                num_image_formats,
                image_formats,
                &num_image_formats);
    LOG_OCL_ERROR(clStatus, "clGetSupportedImageFormats failed." );

    for(int i =0;i<num_image_formats;i++)
    {
         print_image_format(image_formats[i]);
    }
    return 0;
}
