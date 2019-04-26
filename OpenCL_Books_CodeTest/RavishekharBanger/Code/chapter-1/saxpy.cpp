//If you want to build the file directly at the command prompt then use the following commands. 
//AMD commands
//cl /c saxpy.cpp /I"%AMDAPPSDKROOT%\include"
//link  /OUT:"saxpy.exe" "%AMDAPPSDKROOT%\lib\x86_64\OpenCL.lib" saxpy.obj

//nVIDIA commands
//cl /c saxpy.cpp /I"%NVSDKCOMPUTE_ROOT%\OpenCL\common\inc"
//link  /OUT:"saxpy.exe" "%NVSDKCOMPUTE_ROOT%\OpenCL\common\lib\x64\OpenCL.lib" saxpy.obj

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __APPLE__
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif
#include <ocl_macros.h>

// Common defines
#define VENDOR_NAME "AMD"
#define DEVICE_TYPE CL_DEVICE_TYPE_GPU // Использовать будем GPU

#define VECTOR_SIZE 1024 // 1024 флоата


// Ядро OpenCL которое запускается на GPU для каждого потока
const char *saxpy_kernel =
"__kernel                                   \n"
"void saxpy_kernel(const float alpha,       \n"
"                  __global float *A,       \n"
"                  __global float *B,       \n"
"                  __global float *C)       \n"
"{                                          \n"
"    //Get the index of the work-item       \n"
"    int index = get_global_id(0);          \n"
"    C[index] = alpha* A[index] + B[index]; \n"
"}                                          \n";

int main(void) {

    cl_int clStatus = 0; //Keeps track of the error values returned. 

    // Информация о платформе и устройстве
    cl_platform_id* platforms = NULL;

    // Создаем платформы, макрос из common/ocl_macros.h
    OCL_CREATE_PLATFORMS( platforms );

    // Получаем список устройств и выбираем тип устройства на котором будем запускать
    cl_device_id* device_list = NULL;
    OCL_CREATE_DEVICE( platforms[0], DEVICE_TYPE, device_list);

    // Создаем OpenCL context для устройств из списка
    cl_context context;
    /*cl_context_properties props[3] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platforms[0],
        0
    };*/
    
    // OpenCL контекст может быть ассоциирован с множеством устройств, в том числе CPU + GPU,
    // на основании DEVICE_TYPE переданного выше
    context = clCreateContext( NULL, num_devices, device_list, NULL, NULL, &clStatus);
    LOG_OCL_ERROR(clStatus, "clCreateContext Failed..." );

    // Создаем очередь команд для первого устройства из списка
    cl_command_queue command_queue = clCreateCommandQueue(context, device_list[0], 0, &clStatus);
    LOG_OCL_ERROR(clStatus, "clCreateCommandQueue Failed..." );

    // Выделяем память для массивов A, B and C
    float alpha = 2.0;
    float* A = (float*)malloc(sizeof(float)*VECTOR_SIZE);
    float* B = (float*)malloc(sizeof(float)*VECTOR_SIZE);
    float* C = (float*)malloc(sizeof(float)*VECTOR_SIZE);
    
    // Заполняем массивы
    for(int32_t i = 0; i < VECTOR_SIZE; i++) {
        A[i] = (float)i;
        B[i] = (float)(VECTOR_SIZE - i);
        C[i] = 0;
    }

    // Создаем буфферы на устройстве
    cl_mem A_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY, VECTOR_SIZE * sizeof(float), NULL, &clStatus);
    cl_mem B_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY, VECTOR_SIZE * sizeof(float), NULL, &clStatus);
    cl_mem C_clmem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, VECTOR_SIZE * sizeof(float), NULL, &clStatus);

    // Ставим в очередь копирование данных на девайс, копирование блокирующее
    clStatus = clEnqueueWriteBuffer(command_queue, A_clmem, CL_TRUE, 0, VECTOR_SIZE * sizeof(float), A, 0, NULL, NULL);
    LOG_OCL_ERROR(clStatus, "clEnqueueWriteBuffer Failed..." );
    clStatus = clEnqueueWriteBuffer(command_queue, B_clmem, CL_TRUE, 0, VECTOR_SIZE * sizeof(float), B, 0, NULL, NULL);
    LOG_OCL_ERROR(clStatus, "clEnqueueWriteBuffer Failed..." );

    // Создаем программу из вычислительного ядра
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&saxpy_kernel, NULL, &clStatus);
    LOG_OCL_ERROR(clStatus, "clCreateProgramWithSource Failed..." );

    // Компилим программу для GPU
    clStatus = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);
    if(clStatus != CL_SUCCESS){
        LOG_OCL_COMPILER_ERROR(program, device_list[0]);
    }

    // Создаем вычислительное ядро OpenCL
    cl_kernel kernel = clCreateKernel(program, "saxpy_kernel", &clStatus);

    // Выставляем аргументы для ядра, обратите внимание на описание в коде ядра -
    // первый параметр контанта, остальные - буфферы
    clStatus = clSetKernelArg(kernel, 0, sizeof(float),  (void *)&alpha);
    clStatus |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&A_clmem);
    clStatus |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&B_clmem);
    clStatus |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&C_clmem);
    LOG_OCL_ERROR(clStatus, "clSetKernelArg Failed..." );

    // Ставим в очередь исполнение ядра
    size_t global_size = VECTOR_SIZE; // Общее количество элементов
    size_t local_size = 64;           // TODO: Количество элементов в пределах тредгруппы, какое число тут лучше выбирать??
    cl_event saxpy_event;
    clStatus = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, &saxpy_event);
    LOG_OCL_ERROR(clStatus, "clEnqueueNDRangeKernel Failed..." );

    // Читаем в блокирующем режиме данные из GPU на CPU когда у нас завершится вычисление,
    // отслеживаем окончание вычисления с помощью евента saxpy_event
    clStatus = clEnqueueReadBuffer(command_queue, C_clmem, CL_TRUE, 0, VECTOR_SIZE * sizeof(float), C, 1, &saxpy_event, NULL);
    LOG_OCL_ERROR(clStatus, "clEnqueueReadBuffer Failed..." );

    // Ждем завершения очереди
    clStatus = clFinish(command_queue);

    // Отображаем результат
    for(int32_t i = 0; i < VECTOR_SIZE; i++){
        printf("%f * %f + %f = %f\n", alpha, A[i], B[i], C[i]);
    }

    // Уничтожаем данные
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

    return 0;
}
