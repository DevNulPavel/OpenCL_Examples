#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <random>
#include <iostream>
#include <chrono>
#include "scene.h"



const bool USE_CPU = true;
const uint32_t samplesCount = 128;
const int32_t width = 600;
const int32_t height = 600;
const size_t pixelCount = (size_t)width * (size_t)height;
const uint32_t sphereCount = sizeof(spheres)/sizeof(Sphere);
const uint32_t triangleCount = sizeof(triag)/sizeof(Triangle);

uint32_t* pixels = nullptr;
cl_float3* colors = nullptr;
uint32_t* seeds = nullptr;


int main(){
    cl_int err = 0;

    // Получаем количество доступных платформ
    cl_uint numPlatforms = 0;
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to get OpenCL platforms\n");
        exit(-1);
    }
    
    // Выбираем нужную нам платформу
    cl_platform_id platform = 0;
    if (numPlatforms > 0) {
        // Выбираем нужную нам платформу
        cl_platform_id* platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * numPlatforms);
        err = clGetPlatformIDs(numPlatforms, platforms, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Failed to get OpenCL platform IDs\n");
            exit(-1);
        }
        
        for (uint32_t i = 0; i < numPlatforms; ++i) {
            // Получаем информацию о конкретной платформе
            uint8_t pbuf[100];
            err = clGetPlatformInfo(platforms[i],
                                    CL_PLATFORM_VENDOR,
                                    sizeof(pbuf),
                                    pbuf,
                                    NULL);
            
            // Получаем ID платформы
            err = clGetPlatformIDs(numPlatforms, platforms, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Failed to get OpenCL platform IDs\n");
                exit(-1);
            }
            
            // Выводим информацию о платформе
            fprintf(stderr, "OpenCL Platform %d: %s\n", i, pbuf);
        }
        
        platform = platforms[0];
        free(platforms);
    }
    
    // Получаем количество девайсов
    cl_uint deviceCount = 0;
    cl_device_id devices[32];
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 32, devices, &deviceCount);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to get OpenCL device IDs\n");
        exit(-1);
    }
    
    // Обходим все устройства и получаем информацию
    for (uint32_t i = 0; i < deviceCount; ++i) {
        cl_device_type type = 0;
        err = clGetDeviceInfo(devices[i],
                              CL_DEVICE_TYPE,
                              sizeof(cl_device_type),
                              &type,
                              NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Failed to get OpenCL device info: %d\n", err);
            exit(-1);
        }
    }
    
    // Выведем информацию о
    {
        cl_uint addr_data = 0;
        char name_data[48], ext_data[4096];
        for (uint32_t i = 0; i < deviceCount; ++i){
            // Получаем имя устройства
            err = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(name_data), name_data, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Failed to get OpenCL device info\n");
                exit(-1);
            }
            // Получаем адресную информацию
            err = clGetDeviceInfo(devices[i], CL_DEVICE_ADDRESS_BITS, sizeof(ext_data), &addr_data, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Failed to get OpenCL device info\n");
                exit(-1);
            }
            // Получаем расширения устройства
            err = clGetDeviceInfo(devices[i], CL_DEVICE_EXTENSIONS, sizeof(ext_data), ext_data, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Failed to get OpenCL device info\n");
                exit(-1);
            }
            
            // Выводим информацию об устройстве
            std::cout << "Device index: " << i << std::endl;
            std::cout << "Name: " << name_data << std::endl;
            std::cout << "Address width: "<< addr_data << std::endl;
            std::cout << "Extentions:"<< ext_data << std::endl << std::endl;
        }
    }

    // Выбираем устройство
    cl_device_id device = 0;
    if (USE_CPU) {
        device = devices[0];
        std::cout << "Device selected at index: " << 0 << std::endl << std::endl;
    } else {
        device = devices[1];        // OR 2,3,4 ... depending on which GPU to use
        std::cout << "Device selected at index: " << 1 << std::endl << std::endl;
    }
    
    // Создаем контекст OpenCL
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to open OpenCL context\n");
        exit(-1);
    }
    
    // Создаем вычислительное ядро
    FILE* programTextFileHandle = fopen("radianceGPU.cl", "r");       // YOUR PATH TO KERNEL
    if (!programTextFileHandle){
        perror("Failed to open kernel");
    }
    
    // Размер текста программы
    fseek(programTextFileHandle, 0, SEEK_END);
    size_t program_size = ftell(programTextFileHandle);
    rewind(programTextFileHandle);
    
    // Выделяем память под эту программу
    char* program_buffer = (char*)malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    
    // Читаем текст и закрываем файл
    fread(program_buffer, sizeof(char), program_size, programTextFileHandle);
    fclose(programTextFileHandle);
    
    // Создаем вычислительную программу
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&program_buffer, &program_size, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to open OpenCL kernel sources: %d\n", err);
        exit(-1);
    }
    free(program_buffer);
    
    // Компилируем программу
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to build OpenCL kernel: %d\n", err);
    }
    
    // Создаем вычислительное ядро с функцией
    cl_kernel kernel = clCreateKernel(program, "radianceGPU", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create OpenCL kernel: %d\n", err);
        exit(-1);
    }

    // Создание коммандной очереди для обработки
    cl_command_queue commandQueue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create OpenCL command queue: %d\n", err);
        exit(-1);
    }

    // Создаем буффер с данными для пикселей
    colors = (cl_float3 *)malloc(sizeof(cl_float3) * pixelCount);
    
    // Буфер с рандомными значениями
    seeds = (uint32_t*)malloc(sizeof(uint32_t) * pixelCount * 2);
    for (uint32_t i = 0; i < pixelCount * 2; i++) {
        seeds[i] = rand();
        if (seeds[i] < 2){
            seeds[i] = 2;
        }
    }
    
    // Создание буфера цвета в который будет запись
    cl_mem colorBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float3) * pixelCount, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create OpenCL color buffer: %d\n", err);
        exit(-1);
    }
    
    // Буфер с рандомными данными одновременно копируя эти рандомные данные на устройство
    cl_mem randomBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned int) * pixelCount * 2, seeds, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create OpenCL random buffer: %d\n", err);
        exit(-1);
    }

    // Ивенты окончания записи
    cl_event writeCompleteEvents[2];
    
    // Буфер с данными о сферах
    cl_mem sphereBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(Sphere) * (size_t)sphereCount, spheres, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create OpenCL scene buffer: %d\n", err);
        exit(-1);
    }
    
    /*// Копируем данные о сферах в буфер в синхронном режиме
    err = clEnqueueWriteBuffer(commandQueue, sphereBuffer, CL_TRUE, 0,
                               sizeof(Sphere) * (size_t)sphereCount, spheres,
                               0, NULL, &(writeCompleteEvents[0]));
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to write the OpenCL scene Sphere buffer: %d\n", err);
        exit(-1);
    }*/
    
    // Создание буфера с треугольниками
    cl_mem triangleBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(Triangle) * (size_t)triangleCount, triag, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create OpenCL scene Triangle buffer: %d\n", err);
        exit(-1);
    }
    
    /*// Записываем даннные о треугольниках синхронно на устройство
    err = clEnqueueWriteBuffer(commandQueue, triangleBuffer, CL_TRUE, 0,
                               sizeof(Triangle) * (size_t)triangleCount, triag,
                               0, NULL, &(writeCompleteEvents[1]));
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to write the OpenCL scene Triangle buffer: %d\n", err);
        exit(-1);
    }*/
    
    // Устанавливаем данные аргументов вычислительного ядра
    {
        // Буфер цвета
        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&colorBuffer);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Failed to set OpenCL kernel arg. colorBuffer: %d\n", err);
            exit(-1);
        }
        // Буфер сфер
        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&sphereBuffer);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Failed to set OpenCL kernel arg. sphereBuffer: %d\n", err);
            exit(-1);
        }
        // Буфер треугольников
        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&triangleBuffer);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Failed to set OpenCL kernel arg. triangleBuffer: %d\n", err);
            exit(-1);
        }
        // Буфер случайных значений
        err = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&randomBuffer);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Failed to set OpenCL kernel arg. randomBuffer: %d\n", err);
            exit(-1);
        }
        // Ширина
        err = clSetKernelArg(kernel, 4, sizeof(int32_t), (void*)&width);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Failed to set OpenCL kernel arg. width: %d\n", err);
            exit(-1);
        }
        // Высота
        err = clSetKernelArg(kernel, 5, sizeof(int32_t), (void*)&height);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Failed to set OpenCL kernel arg. height: %d\n", err);
            exit(-1);
        }
        // Количество сфер
        err = clSetKernelArg(kernel, 6, sizeof(uint32_t), (void*)&sphereCount);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Failed to set OpenCL kernel arg. sphereCount: %d\n", err);
            exit(-1);
        }
        // Количество треугольников
        err = clSetKernelArg(kernel, 7, sizeof(uint32_t), (void *)&triangleCount);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Failed to set OpenCL kernel arg. triangleCount: %d\n", err);
            exit(-1);
        }
        // Количество семплов
        err = clSetKernelArg(kernel, 8, sizeof(uint32_t), (void *)&samplesCount);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Failed to set OpenCL kernel arg. triangleCount: %d\n", err);
            exit(-1);
        }
    }
    
    // Общее количество работы
    const size_t work_units_per_kernel = pixelCount;
    const size_t workGroupSize = 32;
    
    std::cout<< "Kernel is being executed " << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Ставим вычисления в очередь
    cl_event completeEvent = 0;
    clEnqueueNDRangeKernel(commandQueue,
                           kernel,
                           1,                       // Одномерные данные
                           0,                       // Нулевое смещение по данным
                           &work_units_per_kernel,  // Глобальное количество работы
                           &workGroupSize,          // Дефолтный размер вычислительной группы
                           0,                       // Ивенты окончания записи которые ждем
                           NULL,                    // Ивенты окончания записи которые ждем (writeCompleteEvents)
                           &completeEvent);         // Ивент окончания вычислений
 
    // Получаем данные из устройства в синхронном режиме
    clEnqueueReadBuffer(commandQueue, colorBuffer, CL_TRUE, 0, sizeof(cl_float3)*pixelCount, colors, 1, &completeEvent, NULL);
    
    // Данные о времени вычисления
    std::cout << " "<<std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Computation Time: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() <<" ms"<<std::endl;
    std::cout << "Writing to file " << std::endl;

    // Сохраняем данные в файл
    FILE* f = fopen("result_image.ppm", "w");         // Write image to PPM file.   // YOUR PATH TO IMAGE
    fprintf(f, "P3\n%d %d\n%d\n", width, height, 255);
    for (uint32_t i = 0; i < width*height; i++){
        fprintf(f,"%d %d %d ", toInt(colors[i].x), toInt(colors[i].y),toInt(colors[i].z));
    }
    
    // Уничтожаем данные
    clReleaseMemObject(colorBuffer);
    clReleaseMemObject(randomBuffer);
    clReleaseMemObject(sphereBuffer);
    clReleaseMemObject(triangleBuffer);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commandQueue);
    clReleaseProgram(program);
    clReleaseContext(context);
    clReleaseDevice(device);
    for (uint32_t i = 0 ; i < deviceCount; ++i) {
        clReleaseDevice(devices[i]);
    }
    free(seeds);
    free(colors);
    std::cout<< "Resources Deallocated " << std::endl;

    return 0;
    
}
