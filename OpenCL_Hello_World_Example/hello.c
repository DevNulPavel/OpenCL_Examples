#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>

#ifdef __APPLE__
    #include <OpenCL/opencl.h>
#else
    #include <CL/cl.h>
#endif


// https://habrahabr.ru/post/261323/
// https://habrahabr.ru/post/124925/
// https://ru.wikipedia.org/wiki/OpenCL
// https://developer.apple.com/library/content/documentation/Performance/Conceptual/OpenCL_MacProgGuide/Introduction/Introduction.html

////////////////////////////////////////////////////////////////////////////////

#define STRINGIFY(_STR_) (#_STR_)

// Размер данных
#define DATA_SIZE ((1024*1024/sizeof(float)) * 256) // in Mb
#define CPU_TEST_COUNT 30
#define GPU_TEST_COUNT 30

////////////////////////////////////////////////////////////////////////////////

// Чем больше объем вычислений - тем больше прирост от GPU
inline float calculateFunction(float a, float b){
    float result = 0;
    result += a*a*0.45 + b*0.78 + 0.23;
    result += a*a*0.53 + b*0.2318 + 0.756;
    result += a*a*0.57 + b*0.345 + 0.35;
    result += a*a*0.456 + b*0.3458 + 0.35;
    result += a*a*0.765 + b*0.34 + 0.345;
    result += a*a*0.564 + b*0.675 + 0.35567;
    result += a*a*0.57 + b*0.57 + 0.35567;
    result += a*a*0.6735 + b*0.457 + 0.345567;
    result += a*a*0.3453 + b*0.45 + 0.355;
    result += a*a*0.45667 + b*0.547 + 0.35567;
    result += a*a*0.57 + b*0.457 + 0.34557;
    return result;
}

const char* KernelSource = STRINGIFY(
     float calculateFunction(float a, float b){
         float result = 0;
         result += a*a*0.45 + b*0.78 + 0.23;
         result += a*a*0.53 + b*0.2318 + 0.756;
         result += a*a*0.57 + b*0.345 + 0.35;
         result += a*a*0.456 + b*0.3458 + 0.35;
         result += a*a*0.765 + b*0.34 + 0.345;
         result += a*a*0.564 + b*0.675 + 0.35567;
         result += a*a*0.57 + b*0.57 + 0.35567;
         result += a*a*0.6735 + b*0.457 + 0.345567;
         result += a*a*0.3453 + b*0.45 + 0.355;
         result += a*a*0.45667 + b*0.547 + 0.35567;
         result += a*a*0.57 + b*0.457 + 0.34557;
         return result;
     }

    // Вычислительное ядро для возведения в квадрат большого объема данных
    __kernel void square(__global float* input,
                         __global float* output,
                         const unsigned int count)
    {
        // Получаем индекс в массиве
        int i = get_global_id(0);
        // Проверяем, что не вышли за границы
        if(i < count){
            // Выполняем вычисление
            output[i] = calculateFunction(input[i], input[count - i - 1]);
        }
    }
);

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
    int err = 0;                            // error code returned from api calls
    
    // Подключаем вычислительный девайс в виде GPU
    cl_device_id device_id = 0;             // compute device id
    err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL); // CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU
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
    float* data = malloc(sizeof(float) * DATA_SIZE);
    if (data == NULL) {
        printf("Error: Failed to allocate cpu src random memory!\n");
        exit(1);
    }
    for(size_t i = 0; i < DATA_SIZE; i++){
        data[i] = (float)((double)rand() / (double)RAND_MAX);
    }
    
    // Буффер с результатами
    float* results = malloc(sizeof(float) * DATA_SIZE);
    if (results == NULL) {
        printf("Error: Failed to allocate cpu result memory!\n");
        exit(1);
    }
    
    // Получаем максимальный размер группы для выполнения ядра (макс. количество потоков по X,Y,Z в группе)
    size_t local = 0;
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }
    
    double totalGPUTime = 0.0;
    for (size_t i = 0; i < GPU_TEST_COUNT; i++) {
        // Начало вычислений
        clock_t beginTime = clock();
        
        // Записываем наши данные в буффер на GPU
        err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(float) * DATA_SIZE, data, 0, NULL, NULL);
        if (err != CL_SUCCESS){
            printf("Error: Failed to write to source array!\n");
            exit(1);
        }
        
        // Выставляем параметры для нашего вычислительного ядра
        unsigned int inputCount = DATA_SIZE;
        err = CL_SUCCESS;
        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
        err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &inputCount);
        if (err != CL_SUCCESS) {
            printf("Error: Failed to set kernel arguments! %d\n", err);
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
        err = clEnqueueReadBuffer(commands, output, CL_TRUE, 0, sizeof(float) * DATA_SIZE, results, 0, NULL, NULL);
        if (err != CL_SUCCESS){
            printf("Error: Failed to read output array! %d\n", err);
            exit(1);
        }
        
        // Время завершения вычислений
        clock_t endTime = clock();
        double timeSpent = (double)(endTime - beginTime) / CLOCKS_PER_SEC;
        
        totalGPUTime += timeSpent;
    }
    
    // Проверяем результаты
    size_t correct = 0;
    for(size_t i = 0; i < DATA_SIZE; i++) {
        float resultVal = results[i];
        float targetVal = calculateFunction(data[i], data[DATA_SIZE-i-1]);
        if(fabsf(resultVal - targetVal) < 0.00001){
            correct++;
        }else{
            printf("Error values %.10f != %.10f\n", resultVal, targetVal);
            free(results);
            free(data);
            exit(1);
        }
    }
    
    // Выводим инфу по GPU
    printf("Computed using OpenCL for %f, '%ld/%ld' correct values, ok = %s!\n", totalGPUTime/(double)GPU_TEST_COUNT, correct, DATA_SIZE, (correct == DATA_SIZE) ? "true" : "false");

    // Чистим память
    clReleaseMemObject(input);
    clReleaseMemObject(output);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
    
    // Снова заполняем случайными значениями
    for(size_t i = 0; i < DATA_SIZE; i++){
        data[i] = (float)((double)rand() / (double)RAND_MAX);
    }
    
    double totalCPUTime = 0.0;
    for (size_t i = 0; i < CPU_TEST_COUNT; i++) {
        // Время начала
        clock_t beginTime = clock();
        
        // Вычисляем
        for(size_t i = 0; i < DATA_SIZE; i++){
            results[i] = calculateFunction(data[i], data[DATA_SIZE-i-1]);
        }
        
        // Время завершения вычислений
        clock_t endTime = clock();
        double timeSpent = (double)(endTime - beginTime) / CLOCKS_PER_SEC;
        
        totalCPUTime += timeSpent;
    }
    
    // Выводим инфу по CPU
    printf("Computed using CPU for %f\n", totalCPUTime/(double)CPU_TEST_COUNT);
    
    printf("GPU faster then CPU in X%f times \n", (float)(totalCPUTime/totalGPUTime));
    
    free(results);
    free(data);

    return 0;
}

