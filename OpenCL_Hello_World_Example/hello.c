#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <OpenCL/opencl.h>

////////////////////////////////////////////////////////////////////////////////

// Размер данных
#define DATA_SIZE (4096)

////////////////////////////////////////////////////////////////////////////////

// Вычислительное ядро для возведения в квадрат большого объема данных
const char* KernelSource = "\n" \
"__kernel void square(                                                  \n" \
"   __global float* input,                                              \n" \
"   __global float* output,                                             \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count)                                                       \n" \
"       output[i] = input[i] * input[i];                                \n" \
"}                                                                      \n" \
"\n";

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
    clock_t beginTime = clock();
    
    int err = 0;                            // error code returned from api calls
    
    // Подключаем вычислительный девайс в виде GPU
    cl_device_id device_id = 0;             // compute device id
    err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL); //  : CL_DEVICE_TYPE_CPU
    if (err != CL_SUCCESS) {
        printf("Error: Failed to create a device group!\n");
        return EXIT_FAILURE;
    }
  
    // Создание вычислительного контекста
    cl_context context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context){
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }

    // Создание очереди комманд
    cl_command_queue commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands) {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }

    // Создание вычислительной программы-шейдера для GPU из текста выше
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&KernelSource, NULL, &err);
    if (!program) {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }

    // Компиляция программы для GPU
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len = 0;
        char buffer[2048] = {0};
        
        // Читаем ошибку
        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }

    // Create the compute kernel in the program we wish to run
    // Создаем вычислительное ядро программы, которую мы хотим запускать (получаем вычислительную функцию)
    cl_kernel kernel = clCreateKernel(program, "square", &err);
    if (!kernel || err != CL_SUCCESS){
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }

    // Создаем входной и выходной массивы в памяти GPU для вычислений
    cl_mem input = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * DATA_SIZE, NULL, NULL);
    cl_mem output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * DATA_SIZE, NULL, NULL);
    if (!input || !output) {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }
    
    // Заполняем входные данные случайными значениями
    float data[DATA_SIZE] = {0};
    int i = 0;
    for(i = 0; i < DATA_SIZE; i++){
        data[i] = rand() / (float)RAND_MAX;
    }
    
    // Записываем наши данные в буффер на GPU
    err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(float) * DATA_SIZE, data, 0, NULL, NULL);
    if (err != CL_SUCCESS){
        printf("Error: Failed to write to source array!\n");
        exit(1);
    }
    
    // Выставляем параметры для нашего вычислительного ядра
    unsigned int inputCount = DATA_SIZE;
    err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &inputCount);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }

    // Получаем максимальный размер группы для выполнения ядра (макс. количество потоков по X,Y,Z в группе)
    size_t local = 0;
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }

    // Выполняем вычислительное ядро на указанном диапазоне одномерных входных данных,
    // используя полученное максимальное количество потоков данного устройства
    size_t global = DATA_SIZE;
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err) {
        printf("Error: Failed to execute kernel!\n");
        return EXIT_FAILURE;
    }

    // Ждем завершения выполнения комманд для получения результата
    clFinish(commands);

    // Читаем данные из буффера устройства для проверки верности данных
    float results[DATA_SIZE] = {0};
    err = clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(float) * DATA_SIZE, results, 0, NULL, NULL );
    if (err != CL_SUCCESS){
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }
    
    clock_t endTime = clock();
    double timeSpent = (double)(endTime - beginTime) / CLOCKS_PER_SEC;
    
    // Проверяем результаты
    unsigned int correct = 0;
    for(i = 0; i < DATA_SIZE; i++) {
        if(results[i] == data[i] * data[i]){
            correct++;
        }
    }
    
    // Выводим инфу
    printf("Computed for %f, '%d/%d' correct values!\n", timeSpent, correct, DATA_SIZE);
    
    // Чистим память
    clReleaseMemObject(input);
    clReleaseMemObject(output);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    return 0;
}

