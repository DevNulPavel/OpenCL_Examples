#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <iostream>
#include <ocl_macros.h>

// Build and run the program. 
// Try with all the following combinations and see what are the pixel values retreived. 
// You can either create the sampler at the host code and pass it as a kernel argument 
// or the sampler can be created at the kernel device code.
//         sampler_t sampler = CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
//         sampler_t sampler = CLK_ADDRESS_NONE | CLK_FILTER_LINEAR;
//         sampler_t sampler = CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
//         sampler_t sampler = CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;
//         sampler_t sampler = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
//         sampler_t sampler = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
//         sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_REPEAT | CLK_FILTER_NEAREST;
//         sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_REPEAT | CLK_FILTER_LINEAR;
//         sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_MIRRORED_REPEAT | CLK_FILTER_NEAREST;
//         sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_MIRRORED_REPEAT | CLK_FILTER_LINEAR;
//
// Note that CLK_ADDRESS_REPEAT and CLK_ADDRESS_MIRRORED_REPEAT work only with CLK_NORMALIZED_COORDS_TRUE. 

//" 	const sampler_t sampler = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;               \n"

const char *sample_image_kernel =
" 						                                                    \n"
"__kernel void 						                                        \n"
"image_test(__read_only image2d_t image, 						            \n"
"           __global float4 *out, sampler_t sampler)                        \n"
"{ 						                                                    \n"
"    out[0] = read_imagef(image, sampler, (float2)(0.5f,0.5f)); 			\n"
"    out[1] = read_imagef(image, sampler, (float2)(0.8f,0.5f)); 			\n"
"    out[2] = read_imagef(image, sampler, (float2)(0.3f,0.8f)); 			\n"
"    out[3] = read_imagef(image, sampler, (float2)(0.5f,0.8f)); 			\n"
"    out[4] = read_imagef(image, sampler, (float2)(0.5f,1.3f)); 			\n"
"    out[5] = read_imagef(image, sampler, (float2)(2.3f,1.5f)); 			\n"
"    out[6] = read_imagef(image, sampler, (float2)(5.0f,0.5f)); 			\n"
"    out[7] = read_imagef(image, sampler, (float2)(6.5f,6.5f)); 			\n"
"} 																			\n";


int main()
{
    cl_platform_id platform = NULL;
    cl_device_id device = NULL;
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_int status = 0;
    cl_event task_event, map_event;
    cl_device_type dType = CL_DEVICE_TYPE_GPU;
    cl_int image_width, image_height;
    cl_float4 *result;
    int i, j;
    cl_mem clImage, out;
    cl_bool support;
    int pixels_read = 8;

    //Setup the OpenCL Platform,
    //Get the first available platform. Use it as the default platform
    status = clGetPlatformIDs(1, &platform, NULL);
    LOG_OCL_ERROR(status, "clGetPlatformIDs Failed" );

    //Get the first available device
    status = clGetDeviceIDs (platform, dType, 1, &device, NULL);
    LOG_OCL_ERROR(status, "clGetDeviceIDs Failed" );
    
    /*Check if the device support images */
    clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT, sizeof(support), &support, NULL);
     if (support != CL_TRUE) {
         std::cout <<"IMAGES not supported\n";
         return 1;
     }

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

    /*Create command queue*/
    command_queue = clCreateCommandQueue(context,
                                        device,
                                        0,
                                        &status);
    LOG_OCL_ERROR(status, "clCreateCommandQueue Failed" );

    /* Create Image Object */
    //Create OpenCL device input image with the format and descriptor as below

    cl_image_format image_format;
    image_format.image_channel_data_type = CL_FLOAT;
    image_format.image_channel_order = CL_R;

    //We create a 5 X 5 2D image 
    image_width  = 5; 
    image_height = 5;
#ifdef OPENCL_1_2
    cl_image_desc image_desc;
    image_desc.image_type   = CL_MEM_OBJECT_IMAGE2D;
    image_desc.image_width  = image_width;
    image_desc.image_height = image_height;
    image_desc.image_depth  = 1;
    image_desc.image_array_size  = 1;
    image_desc.image_row_pitch   = image_width*sizeof(float);
    image_desc.image_slice_pitch = 25*sizeof(float);
    image_desc.num_mip_levels = 0;
    image_desc.num_samples    = 0;
    image_desc.buffer         = NULL;
#endif    
    /* Create output buffer */
    out = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float4)*pixels_read, NULL, &status);
    LOG_OCL_ERROR(status, "clCreateBuffer Failed" );

    size_t origin[] = {0,0,0};  /* Transfer target coordinate*/
    size_t region[] = {image_width,image_height,1};  /* Size of object to be transferred */
    float *data = (float *)malloc(image_width*image_height*sizeof(float));
    float pixels[] = {            /* Transfer Data */
        10, 20, 10, 40, 50,
        10, 20, 20, 40, 50,
        10, 20, 30, 40, 50,
        10, 20, 40, 40, 50,
        10, 20, 50, 40, 50
    };
    memcpy(data, pixels, image_width*image_height*sizeof(float));
#ifdef OPENCL_1_2
    clImage = clCreateImage(context,
#else
    clImage = clCreateImage2D(context,
#endif
                            CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, 
                            &image_format, 
#ifdef OPENCL_1_2
                            &image_desc,
#else
                            image_width, image_height, image_width * sizeof(float),
#endif
                            pixels, 
                            &status);
    LOG_OCL_ERROR(status, "clCreateImage Failed" );

    /* If the image was not created using CL_MEM_USE_HOST_PTR, 
       then you can write the image data to the device using the 
       clEnqueueWriteImage function. */
    //status = clEnqueueWriteImage(command_queue, clImage, CL_TRUE, origin, region, 5*sizeof(float), 25*sizeof(float), data, 0, NULL, NULL);
    //LOG_OCL_ERROR(status, "clCreateBuffer Failed" );

    /* Build program */
    program = clCreateProgramWithSource(context, 1, (const char **)&sample_image_kernel,
                                        NULL, &status);
    LOG_OCL_ERROR(status, "clCreateProgramWithSource Failed" );

    // Build the program
    status = clBuildProgram(program, 1, &device, "", NULL, NULL);
    LOG_OCL_ERROR(status, "clBuildProgram Failed" );
    if(status != CL_SUCCESS)
    {
        if(status == CL_BUILD_PROGRAM_FAILURE)
            LOG_OCL_COMPILER_ERROR(program, device);
        LOG_OCL_ERROR(status, "clBuildProgram Failed" );
    }
    printf("Printing the image pixels\n");
    for (i=0; i<image_height; i++) {
        for (j=0; j<image_width; j++) {
            printf("%f ",data[i*image_width +j]);
        }
        printf("\n");
    }

    //Create kernel and set the kernel arguments
    kernel = clCreateKernel(program, "image_test", &status);
    LOG_OCL_ERROR(status, "clCreateKernel of image_test failed" );

    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&clImage);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&out);

    /*********Image sampler with image repeated at every 1.0 normalized coordinate***********/
    /*If host side sampler is not required the sampler objects can also be created on the kernel code. 
      Don't pass the thirsd argument to the kernel and create  a sample object as shown below in the kernel code*/
    //const sampler_t sampler = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST; 

    cl_sampler sampler = clCreateSampler (context,
                            CL_TRUE,
                            CL_ADDRESS_REPEAT,
                            CL_FILTER_NEAREST,
                            &status);
    clSetKernelArg(kernel, 2, sizeof(cl_sampler), (void*)&sampler);
    //Enqueue the kernel 
    status = clEnqueueTask(command_queue, kernel, 0, NULL, &task_event);
    LOG_OCL_ERROR(status, "clEnqueueTask Failed" );
    /* Map the result back to host address */
    result = (cl_float4*)clEnqueueMapBuffer(command_queue, out, CL_TRUE, CL_MAP_READ, 0, 
                                            sizeof(cl_float4)*pixels_read, 1, 
                                            &task_event, &map_event, &status);
    LOG_OCL_ERROR(status, "clEnqueueMapBuffer Failed" );

    printf(" SAMPLER mode set to CL_ADDRESS_REPEAT | CL_FILTER_NEAREST\n");
    printf("\nPixel values retreived based on the filter and Addressing mode selected\n");
    printf("(float2)(0.5f,0.5f) = %f,%f,%f,%f\n",result[0].s[0],result[0].s[1],result[0].s[2],result[0].s[3]);
    printf("(float2)(0.8f,0.5f) = %f,%f,%f,%f\n",result[1].s[0],result[1].s[1],result[1].s[2],result[1].s[3]);
    printf("(float2)(1.3f,0.5f) = %f,%f,%f,%f\n",result[2].s[0],result[2].s[1],result[2].s[2],result[2].s[3]);
    printf("(float2)(0.5f,0.5f) = %f,%f,%f,%f\n",result[3].s[0],result[3].s[1],result[3].s[2],result[3].s[3]);
    printf("(float2)(0.5f,0.8f) = %f,%f,%f,%f\n",result[4].s[0],result[4].s[1],result[4].s[2],result[4].s[3]);
    printf("(float2)(0.5f,1.3f) = %f,%f,%f,%f\n",result[5].s[0],result[5].s[1],result[5].s[2],result[5].s[3]);
    printf("(float2)(4.5f,0.5f) = %f,%f,%f,%f\n",result[5].s[0],result[5].s[1],result[5].s[2],result[5].s[3]);
    printf("(float2)(5.0f,0.5f) = %f,%f,%f,%f\n",result[7].s[0],result[7].s[1],result[7].s[2],result[7].s[3]);
    clEnqueueUnmapMemObject(command_queue, out, result, 0, NULL, NULL);
    clReleaseSampler(sampler);

    /*********Image sampler with image mirrored at every 1.0 normalized coordinate***********/
    sampler = clCreateSampler (context,
                            CL_TRUE,
                            CL_ADDRESS_MIRRORED_REPEAT,
                            CL_FILTER_LINEAR,
                            &status);
    LOG_OCL_ERROR(status, "clCreateSampler Failed" );

    clSetKernelArg(kernel, 2, sizeof(cl_sampler), (void*)&sampler);
    //Enqueue the kernel 
    status = clEnqueueTask(command_queue, kernel, 0, NULL, &task_event);
    LOG_OCL_ERROR(status, "clEnqueueTask Failed" );

    /* Map the result back to host address */
    result = (cl_float4*)clEnqueueMapBuffer(command_queue, out, CL_TRUE, 
                                            CL_MAP_READ, 0, sizeof(cl_float4)*pixels_read, 
                                            1, &task_event, &map_event, &status);
    LOG_OCL_ERROR(status, "clEnqueueMapBuffer Failed" );

    printf(" SAMPLER mode set to CL_ADDRESS_MIRRORED_REPEAT | CL_FILTER_LINEAR\n");
    printf("\nPixel values retreived based on the filter and Addressing mode selected\n");
    printf("(float2)(0.5f,0.5f) = %f,%f,%f,%f\n",result[0].s[0],result[0].s[1],result[0].s[2],result[0].s[3]);
    printf("(float2)(0.8f,0.5f) = %f,%f,%f,%f\n",result[1].s[0],result[1].s[1],result[1].s[2],result[1].s[3]);
    printf("(float2)(1.3f,0.5f) = %f,%f,%f,%f\n",result[2].s[0],result[2].s[1],result[2].s[2],result[2].s[3]);
    printf("(float2)(0.5f,0.5f) = %f,%f,%f,%f\n",result[3].s[0],result[3].s[1],result[3].s[2],result[3].s[3]);
    printf("(float2)(0.5f,0.8f) = %f,%f,%f,%f\n",result[4].s[0],result[4].s[1],result[4].s[2],result[4].s[3]);
    printf("(float2)(0.5f,1.3f) = %f,%f,%f,%f\n",result[5].s[0],result[5].s[1],result[5].s[2],result[5].s[3]);
    printf("(float2)(4.5f,0.5f) = %f,%f,%f,%f\n",result[5].s[0],result[5].s[1],result[5].s[2],result[5].s[3]);
    printf("(float2)(5.0f,0.5f) = %f,%f,%f,%f\n",result[7].s[0],result[7].s[1],result[7].s[2],result[7].s[3]);
    clEnqueueUnmapMemObject(command_queue, out, result, 0, NULL, NULL);
    clReleaseSampler(sampler);
    /********************/

    //Free All OpenCL objects.
    clReleaseMemObject(out);
    clReleaseMemObject(clImage);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
    return 0;

}
