
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include <ocl_macros.h>

#define BLOCK_SIZE 8
/* Enable any one of the following macros. 
 * Each of the following macros enables a part of the code to 
 * explain the different mechanism to implement a Matrix multiplication. */

#define ENABLE_BASIC
//#define ENABLE_BASIC_VECTOR4
//#define ENABLE_LOCAL_A
//#define ENABLE_LOCAL_A_COALLESCED
//#define ENABLE_LOCAL_A_COALLESCED_ROW
//#define ENABLE_COALLESCED_ROW


/* Matrix multiplication kernels. 
 * There are 5 different implementations of this kernel. 
 */
const char *MatrixMul_kernel_basic =
"__kernel                                                                                   \n"
"void MatrixMul_kernel_basic(int dim,                                                       \n"
"                  __global float *A,                                                       \n"
"                  __global float *B,                                                       \n"
"                  __global float *C)                                                       \n"
"{                                                                                          \n"
"    //Get the index of the work-item                                                       \n"
"    int iCol = get_global_id(0);                                                           \n"
"    int iRow = get_global_id(1);                                                           \n"
"    float result = 0.0;                                                                    \n"
"    for(int i=0;i< dim;++i)                                                                \n"
"    {                                                                                      \n"
"        result +=                                                                          \n"
"        A[iRow*dim + i]*B[i*dim + iCol];                                                   \n"
"    }                                                                                      \n"
"    C[iRow*dim + iCol] = result;                                                           \n"
"}                                                                                          \n";

const char *MatrixMul_kernel_basic_vector4 =
"#define VECTOR_SIZE 4                                                                      \n"
"__kernel                                                                                   \n"
"void MatrixMul_kernel_basic_vector4(int dim,                                               \n"
"                  __global float4 *A,                                                      \n"
"                  __global float4 *B,                                                      \n"
"                  __global float *C)                                                       \n"
"{                                                                                          \n"
"    //Get the index of the work-item                                                       \n"
"    int localIdx = get_global_id(0);                                                       \n"
"    int localIdy = get_global_id(1);                                                       \n"
"    float result = 0.0;                                                                    \n"
"    float4 Bvector[4];                                                                     \n"
"    float4 Avector, temp;                                                                  \n"
"    float4 resultVector[4] = {0,0,0,0};                                                    \n"
"    int    rowElements = dim/VECTOR_SIZE;                                                  \n"
"    for(int i=0; i<rowElements; ++i)                                                       \n"
"    {                                                                                      \n"
"        Avector = A[localIdy*rowElements + i];                                             \n"
"        Bvector[0] = B[dim*i + localIdx];                                                  \n"
"        Bvector[1] = B[dim*i + rowElements + localIdx];                                    \n"
"        Bvector[2] = B[dim*i + 2*rowElements + localIdx];                                  \n"
"        Bvector[3] = B[dim*i + 3*rowElements + localIdx];                                  \n"
"        temp = (float4)(Bvector[0].x, Bvector[1].x, Bvector[2].x, Bvector[3].x);           \n"
"        resultVector[0] += Avector * temp;                                                 \n"
"        temp = (float4)(Bvector[0].y, Bvector[1].y, Bvector[2].y, Bvector[3].y);           \n"
"        resultVector[1] += Avector * temp;                                                 \n"
"        temp = (float4)(Bvector[0].z, Bvector[1].z, Bvector[2].z, Bvector[3].z);           \n"
"        resultVector[2] += Avector * temp;                                                 \n"
"        temp = (float4)(Bvector[0].w, Bvector[1].w, Bvector[2].w, Bvector[3].w);           \n"
"        resultVector[3] += Avector * temp;                                                 \n"
"    }                                                                                      \n"
"    C[localIdy*dim + localIdx*VECTOR_SIZE] = resultVector[0].x + resultVector[0].y +       \n"
"                                 resultVector[0].z + resultVector[0].w;                    \n"
"    C[localIdy*dim + localIdx*VECTOR_SIZE + 1] = resultVector[1].x + resultVector[1].y +   \n"
"                                 resultVector[1].z + resultVector[1].w;                    \n"
"    C[localIdy*dim + localIdx*VECTOR_SIZE + 2] = resultVector[2].x + resultVector[2].y +   \n"
"                                 resultVector[2].z + resultVector[2].w;                    \n"
"    C[localIdy*dim + localIdx*VECTOR_SIZE + 3] = resultVector[3].x + resultVector[3].y +   \n"
"                                 resultVector[3].z + resultVector[3].w;                    \n"
"}                                                                                          \n";


const char *MatrixMul_kernel_localA =
"__kernel                                                                                   \n"
"void MatrixMul_kernel_localA(int dim,                                                      \n"
"                  __global float *A,                                                       \n"
"                  __global float *B,                                                       \n"
"                  __global float *C,                                                       \n"
"                  __local  float *lA)                                                      \n"
"{                                                                                          \n"
"    //Get the index of the work-item                                                       \n"
"    int iCol = get_global_id(0);                                                           \n"
"    int iRow = get_global_id(1);                                                           \n"
"    int localIdx = get_local_id(0);                                                        \n"
"    int localSizex = get_local_size(0);                                                    \n"
"    float result = 0.0f;                                                                   \n"
"    int numElements = dim/localSizex;                                                      \n"
"    for(int i=0; i<numElements ; i++)                                                      \n"
"    {                                                                                      \n"
"        lA[localIdx*numElements + i] = A[iRow*dim + localIdx*numElements + i];             \n"
"    }                                                                                      \n"
"    barrier(CLK_LOCAL_MEM_FENCE);                                                          \n"
"    for(int i=0;i< dim;++i)                                                                \n"
"    {                                                                                      \n"
"        result +=                                                                          \n"
"        lA[i]*B[i*dim + iCol];                                                             \n"
"        //printf(\"%d, %d = %f - %f\\n\",iCol, iRow, lA[i],B[i*dim + iCol]);               \n"
"    }                                                                                      \n"
"    C[iRow*dim + iCol] = result;                                                           \n"
"}                                                                                          \n";


const char *MatrixMul_kernel_localA_coallesced =
"__kernel                                                                                   \n"
"void MatrixMul_kernel_localA_coallesced(int dim,                                           \n"
"                  __global float *A,                                                       \n"
"                  __global float *B,                                                       \n"
"                  __global float *C,                                                       \n"
"                  __local  float *lA)                                                      \n"
"{                                                                                          \n"
"    //Get the index of the work-item                                                       \n"
"    int iCol = get_global_id(0);                                                           \n"
"    int iRow = get_global_id(1);                                                           \n"
"    int localIdx = get_local_id(0);                                                        \n"
"    int localSizex = get_local_size(0);                                                    \n"
"    float result = 0.0f;                                                                   \n"
"    int numElements = dim/localSizex;                                                      \n"
"    for(int i=0; i<numElements ; i++)                                                      \n"
"    {                                                                                      \n"
"        lA[i*localSizex + localIdx] = A[iRow*dim + i*localSizex + localIdx];               \n"
"    }                                                                                      \n"
"    barrier(CLK_LOCAL_MEM_FENCE);                                                          \n"
"    for(int i=0;i< dim;++i)                                                                \n"
"    {                                                                                      \n"
"        result +=                                                                          \n"
"        lA[i]*B[i*dim + iCol];                                                             \n"
"        //printf(\"%d, %d = %f - %f\\n\",iCol, iRow, lA[i],B[i*dim + iCol]);               \n"
"    }                                                                                      \n"
"    C[iRow*dim + iCol] = result;                                                           \n"
"}                                                                                          \n";

//4 eleemnts per work Item
const char *MatrixMul_kernel_coallesced_row =
"__kernel                                                                                   \n"
"void MatrixMul_kernel_coallesced_row(int dim,                                              \n"
"                  __global float *A,                                                       \n"
"                  __global float *B,                                                       \n"
"                  __global float *C)                                                       \n"
"{                                                                                          \n"
"    //Get the index of the work-item                                                       \n"
"    int iCol = get_global_id(0);                                                           \n"
"    int iRow = get_global_id(1);                                                           \n"
"    int localIdx = get_local_id(0);                                                        \n"
"    int localSizex = get_local_size(0);                                                    \n"
"    float result = 0.0f;                                                                   \n"
"    int numElements = dim/localSizex;                                                      \n"
"    for(int j=0; j<numElements; j++)                                                       \n"
"    {                                                                                      \n"
"       result = 0.0f;                                                                      \n"
"       for(int i=0;i< dim;++i)                                                             \n"
"       {                                                                                   \n"
"           result += A[iRow*dim + i]*B[i*dim + j*localSizex + localIdx];                   \n"
"       }                                                                                   \n"
"       C[iRow*dim + j*localSizex + iCol] = result;                                         \n"
"    }                                                                                      \n"
"}                                                                                          \n";

const char *MatrixMul_kernel_localA_coallesced_row =
"__kernel                                                                                   \n"
"void MatrixMul_kernel_localA_coallesced_row(int dim,                                       \n"
"                  __global float *A,                                                       \n"
"                  __global float *B,                                                       \n"
"                  __global float *C,                                                       \n"
"                  __local  float *lA)                                                      \n"
"{                                                                                          \n"
"    //Get the index of the work-item                                                       \n"
"    int iCol = get_global_id(0);                                                           \n"
"    int iRow = get_global_id(1);                                                           \n"
"    int localIdx = get_local_id(0);                                                        \n"
"    int localSizex = get_local_size(0);                                                    \n"
"    float result = 0.0f;                                                                   \n"
"    int numElements = dim/localSizex;                                                      \n"
"    for(int i=0; i<numElements ; i++)                                                      \n"
"    {                                                                                      \n"
"        lA[i*localSizex + localIdx] = A[iRow*dim + i*localSizex + localIdx];               \n"
"    }                                                                                      \n"
"    barrier(CLK_LOCAL_MEM_FENCE);                                                          \n"
"    for(int j=0; j<numElements; j++)                                                       \n"
"    {                                                                                      \n"
"       result = 0.0;                                                                       \n"
"       for(int i=0;i< dim;++i)                                                             \n"
"       {                                                                                   \n"
"           result += lA[i]*B[i*dim + j*localSizex + localIdx];                             \n"
"       }                                                                                   \n"
"       C[iRow*dim + j*localSizex + iCol] = result;                                         \n"
"    }                                                                                      \n"
"}                                                                                          \n";


//One kernel computes 1 Row of C
//1 fixed row of A is used to compute that row of C
//But, for each element  of C one distinct column is required.
// We buffer that row in private memory : Extra copy time but
// saving repated global memory fetch


bool resultIsCorrect(float* pA,float* pB,float* pCTest, int dim);
void matrixMultWithLocal();
void matrixMultSimpleKernel();
void matrixMultWithLocal2();

int callMatrixMult1(int MATRIX_WIDTH, int MATRIX_HEIGHT, bool verify)
{
    cl_event events;
    int i;
    // Allocate space for vectors A, B and C
    int alpha = MATRIX_WIDTH;
    float *A = (float*)malloc(sizeof(float)*MATRIX_WIDTH*MATRIX_HEIGHT);
    float *B = (float*)malloc(sizeof(float)*MATRIX_WIDTH*MATRIX_HEIGHT);
    float *C = (float*)malloc(sizeof(float)*MATRIX_WIDTH*MATRIX_HEIGHT);
    for(i = 0; i < MATRIX_WIDTH*MATRIX_HEIGHT; i++)
    {
        A[i] = (float) (rand() % 10);
        B[i] = (float) (rand() % 10);
        C[i] = 0;
    }

    // Get platform and device information
    cl_platform_id * platforms = NULL;
    cl_uint     num_platforms;
    //Set up the Platform
    cl_int clStatus = clGetPlatformIDs(0, NULL, &num_platforms);
    platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id)*num_platforms);
    clStatus = clGetPlatformIDs(num_platforms, platforms, NULL);

    //Get the devices list and choose the type of device you want to run on
    cl_device_id     *device_list = NULL;
    cl_uint       num_devices;

    clStatus = clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_GPU, 0,
            NULL, &num_devices);
    device_list = (cl_device_id *)malloc(sizeof(cl_device_id)*num_devices);
    clStatus = clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_GPU, num_devices,
            device_list, NULL);

    // Create one OpenCL context for each device in the platform
    cl_context context;
    cl_context_properties props[3] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platforms,
        0
    };
    context = clCreateContext( NULL, num_devices, device_list, NULL, NULL, &clStatus);

    // Create a command queue
    cl_command_queue_properties prop = 0;
    prop |= CL_QUEUE_PROFILING_ENABLE;
    cl_command_queue command_queue = clCreateCommandQueue(context, device_list[0], prop, &clStatus);
    //cl_uint buf_uint;
    //clGetDeviceInfo(device_list[0], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(buf_uint), &buf_uint, NULL);
    //clGetDeviceInfo(device_list[0], CL_DEVICE_MAX_COMPUTE_UNITS, 4, &buf_uint, &koushik);
    //printf("KOUSHIK FUN:  DEVICE_MAX_COMPUTE_UNITS = %u\n", (unsigned int)buf_uint);

    // Create memory buffers on the device for each vector
    cl_mem A_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY,
            MATRIX_WIDTH*MATRIX_HEIGHT * sizeof(float), NULL, &clStatus);
    cl_mem B_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY,
            MATRIX_WIDTH*MATRIX_HEIGHT * sizeof(float), NULL, &clStatus);
    cl_mem C_clmem = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
            MATRIX_WIDTH*MATRIX_HEIGHT * sizeof(float), NULL, &clStatus);

    // Copy the Buffer A and B to the device
    clStatus = clEnqueueWriteBuffer(command_queue, A_clmem, CL_TRUE, 0,
            MATRIX_WIDTH*MATRIX_HEIGHT * sizeof(float), A, 0, NULL, NULL);
    clStatus = clEnqueueWriteBuffer(command_queue, B_clmem, CL_TRUE, 0,
            MATRIX_WIDTH*MATRIX_HEIGHT * sizeof(float), B, 0, NULL, NULL);

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1,
#ifdef ENABLE_LOCAL_A
            (const char **)&MatrixMul_kernel_localA, NULL, &clStatus);
            printf("\n ENABLE_LOCAL_A \n");
#endif
#ifdef ENABLE_LOCAL_A_COALLESCED
            (const char **)&MatrixMul_kernel_localA_coallesced, NULL, &clStatus);
            printf("\n ENABLE_LOCAL_A_COALLESCED \n");
#endif
#ifdef ENABLE_LOCAL_A_COALLESCED_ROW
            (const char **)&MatrixMul_kernel_localA_coallesced_row, NULL, &clStatus);
            printf("\n ENABLE_LOCAL_A_COALLESCED_ROW \n");
#endif
#ifdef ENABLE_COALLESCED_ROW
            (const char **)&MatrixMul_kernel_coallesced_row, NULL, &clStatus);
            printf("\n ENABLE_COALLESCED_ROW \n");
#endif
#ifdef ENABLE_BASIC
            (const char **)&MatrixMul_kernel_basic, NULL, &clStatus);
            printf("\n ENABLE_BASIC \n");
#endif
#ifdef ENABLE_BASIC_VECTOR4
            (const char **)&MatrixMul_kernel_basic_vector4, NULL, &clStatus);
            printf("\n ENABLE_BASIC_VECTOR4 \n");
#endif

    LOG_OCL_ERROR(clStatus, "clCreateProgramWithSource Failed" );

    // Build the program
    clStatus = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);
    LOG_OCL_ERROR(clStatus, "clBuildProgram Failed" );
    if(clStatus != CL_SUCCESS)
    {
        if(clStatus == CL_BUILD_PROGRAM_FAILURE)
            LOG_OCL_COMPILER_ERROR(program, device_list[0]);
        LOG_OCL_ERROR(clStatus, "clBuildProgram Failed" );
    }


    // Create the OpenCL kernel
#ifdef ENABLE_LOCAL_A
    cl_kernel kernel = clCreateKernel(program, "MatrixMul_kernel_localA", &clStatus);
#endif
#ifdef ENABLE_LOCAL_A_COALLESCED
    cl_kernel kernel = clCreateKernel(program, "MatrixMul_kernel_localA_coallesced", &clStatus);
#endif
#ifdef ENABLE_LOCAL_A_COALLESCED_ROW
    cl_kernel kernel = clCreateKernel(program, "MatrixMul_kernel_localA_coallesced_row", &clStatus);
#endif
#ifdef ENABLE_COALLESCED_ROW
    cl_kernel kernel = clCreateKernel(program, "MatrixMul_kernel_coallesced_row", &clStatus);
#endif
#ifdef ENABLE_BASIC
    cl_kernel kernel = clCreateKernel(program, "MatrixMul_kernel_basic", &clStatus);
#endif
#ifdef ENABLE_BASIC_VECTOR4
    cl_kernel kernel = clCreateKernel(program, "MatrixMul_kernel_basic_vector4", &clStatus);
#endif
//#ifdef ENABLE_ROW_PER_WI
//    cl_kernel kernel = clCreateKernel(program, "MatrixMul_kernel_RowPerWI", &clStatus);
//#endif    // Set the arguments of the kernel
//#ifdef ENABLE_ROW_PER_WI_A_PRIVATE
//    cl_kernel kernel = clCreateKernel(program, "MatrixMul_kernel_RowPerWI_APriv", &clStatus);
//#endif
//#ifdef ENABLE_ROW_PER_WI_A_PRIVATE_B_LOCAL
//    cl_kernel kernel = clCreateKernel(program, "MatrixMul_kernel_RowPerWI_APriv_BLocal", &clStatus);
//#endif
    clStatus = clSetKernelArg(kernel, 0, sizeof(float), (void *)&alpha);
//#ifdef ENABLE_BASIC_VECTOR4
//#else
    clStatus = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&A_clmem);
//#endif
    clStatus = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&B_clmem);
    clStatus = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&C_clmem);
#ifdef ENABLE_LOCAL_A
    clStatus = clSetKernelArg(kernel, 4, MATRIX_WIDTH*sizeof(float), NULL);
#endif
#ifdef ENABLE_LOCAL_A_COALLESCED
    clStatus = clSetKernelArg(kernel, 4, MATRIX_WIDTH*sizeof(float), NULL);
#endif
#ifdef ENABLE_LOCAL_A_COALLESCED_ROW
    clStatus = clSetKernelArg(kernel, 4, MATRIX_WIDTH*sizeof(float), NULL);
#endif
//#ifdef ENABLE_ROW_PER_WI_A_PRIVATE_B_LOCAL
//    clStatus = clSetKernelArg(kernel, 4, MATRIX_HEIGHT*sizeof(float), NULL);
//#endif

    LOG_OCL_ERROR(clStatus, "clSetKernelArg Failed" );

    // Execute the OpenCL kernel on the list
    size_t global_size[2];    size_t local_size[2];

#ifdef ENABLE_LOCAL_A
    global_size[0] = MATRIX_WIDTH;   global_size[1] = MATRIX_HEIGHT;
    local_size[0] =  256;    local_size[1] =  1;
#endif
#ifdef ENABLE_LOCAL_A_COALLESCED
    global_size[0] = MATRIX_WIDTH;   global_size[1] = MATRIX_HEIGHT;
    local_size[0] =  128;   local_size[0] =  256; local_size[1] =  1;
#endif
#ifdef ENABLE_LOCAL_A_COALLESCED_ROW
    global_size[0] = 128; global_size[1] = MATRIX_HEIGHT;
    local_size[0] =  128; local_size[1]  = 1;
#endif
#ifdef ENABLE_COALLESCED_ROW
    global_size[0] = 128;   global_size[1] = MATRIX_HEIGHT;
    local_size[0] =  128; local_size[1] =  1;
#endif
#ifdef ENABLE_BASIC
    global_size[0] = MATRIX_WIDTH;   global_size[1] = MATRIX_HEIGHT;
    local_size[0] =  BLOCK_SIZE;
    local_size[0] =  BLOCK_SIZE*2;
    local_size[1] =  BLOCK_SIZE;
    local_size[1] =  BLOCK_SIZE*2;
#endif
#ifdef ENABLE_BASIC_VECTOR4
    global_size[0] = MATRIX_WIDTH/4;   global_size[1] = MATRIX_HEIGHT;
    local_size[0] =  BLOCK_SIZE;
    local_size[0] =  BLOCK_SIZE*2;
    local_size[1] =  BLOCK_SIZE;
    local_size[1] =  BLOCK_SIZE*2;
#endif

    printf("Running for GLobal = %ld %ld, Local = %ld %ld\n",global_size[0], global_size[1], local_size[0], local_size[1]);
    clStatus = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL,
            global_size, local_size, 0, NULL, &events);
    LOG_OCL_ERROR(clStatus, "clEnqueueNDRangeKernel Failed" );
    clWaitForEvents(1, &events);

    cl_ulong startTime;
    cl_ulong endTime;

    /* Get kernel profiling info */
    clStatus = clGetEventProfilingInfo(events,
                                     CL_PROFILING_COMMAND_START,
                                     sizeof(cl_ulong),
                                     &startTime,
                                     0);
    if(CL_SUCCESS != clStatus) printf("\nclGetEventProfilingInfo Failed -- %d", clStatus);

    clStatus = clGetEventProfilingInfo(events,
                                     CL_PROFILING_COMMAND_END,
                                     sizeof(cl_ulong),
                                     &endTime,
                                     0);
    if(CL_SUCCESS != clStatus) printf("\nclGetEventProfilingInfo Failed -- %d", clStatus);
    double sec = 1e-9 * (endTime - startTime);

    // Read the memory buffer C_clmem on the device to the local variable C
    clStatus = clEnqueueReadBuffer(command_queue, C_clmem, CL_TRUE, 0,
            MATRIX_WIDTH*MATRIX_HEIGHT * sizeof(float), C, 0, NULL, NULL);
    LOG_OCL_ERROR(clStatus, "clEnqueueReadBuffer Failed" );
    // Clean up and wait for all the comands to complete.
    clStatus = clFinish(command_queue);

    printf("\n Kernel1................................................\n");
    printf("\n Time to execute Kernel=%f ms", sec * 1000);

    if(verify)
    {
        if(resultIsCorrect(A,B,C,MATRIX_WIDTH))
        {
            printf("\nKernel 1 - PASSED");
        }
        else
        {
            printf("\nKernel 1 - FAILED");
        }
    }
    // Finally release all OpenCL allocated objects and host buffers.
    clStatus = clReleaseKernel(kernel);
    clStatus = clReleaseProgram(program);
    clStatus = clReleaseMemObject(A_clmem);
    clStatus = clReleaseMemObject(B_clmem);
    clStatus = clReleaseMemObject(C_clmem);
    clStatus = clReleaseCommandQueue(command_queue);
    clStatus = clReleaseContext(context);
    free(A);
    free(B);
    free(C);
    free(platforms);
    free(device_list);
    return CL_SUCCESS;
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        std::cout << "Usage: chapter8.MatrixMultiplication.exe <Matrix Width>\n";
        std::cout << "The matrix width is equal to the height in this implementation.\n";
        std::cout << "The width and height should be a multiple of 256.\n";
        return 0;
    }

    int MATRIX_WIDTH = atoi(argv[1]);

    bool verify = true;
    callMatrixMult1(MATRIX_WIDTH, MATRIX_WIDTH, verify);
    //getchar();
    return 0;
}

bool resultIsCorrect(float* pA,float* pB,float* pCTest, int dim)
{
    bool result = true;
    int arrayLength = dim*dim;
    float *pGoldenValue = (float *)malloc(sizeof(float)*arrayLength);
    //compute the values
    printf ("\nComputing Golden");
    for(int i=0; i<dim; ++i)
    {
        for(int j=0; j<dim; ++j)
        {
            //init the (i,j)-th element to 0
            pGoldenValue[i*dim+j] = 0;
            //compute the (i,j)-th element
            for(int k=0; k<dim; ++k)
            {
                pGoldenValue[i*dim + j] += pA[i*dim + k]*pB[k*dim + j];
            }
        }
        printf (".");
    }
    int errorCount = 10;
    for(int i=0; i<arrayLength && errorCount;++i)
    {
        if(pGoldenValue[i] != pCTest[i] )
        {
            errorCount--;
            result =false;
            printf("\n%d-th elements A=%f B=%f are %f and %f",i, pA[i], pB[i],pGoldenValue[i], pCTest[i] );
        }
    }

    free(pGoldenValue);

    return result;
}

