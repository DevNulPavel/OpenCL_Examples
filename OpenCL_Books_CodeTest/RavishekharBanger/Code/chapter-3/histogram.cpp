#include <cstdlib>
#include <iostream>
#include <string.h>
#include <stdio.h>
#ifdef __APPLE__
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif
#include <ocl_macros.h>
#include <bmp_image.h>

#define USE_HOST_MEMORY
#define STRINGIFY_TO_TEXT(TEXT) (#TEXT)

const char *histogram_kernel = STRINGIFY_TO_TEXT(
    const int BIN_SIZE = 256;
    //#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable // TODO: НЕ хочет компилиться?
     __kernel void histogram_kernel(__global const uint* data,
                           __local uchar* sharedArray,
                           __global uint* binResultR,
                           __global uint* binResultG,
                           __global uint* binResultB)
    {
        // Получаем данные о текущем месте исполнения в сетке
        size_t localId = get_local_id(0); // Индекс в пределах группы
        size_t globalId = get_global_id(0); // Глобальный индекс
        size_t groupId = get_group_id(0); // Номер группы
        size_t groupSize = get_local_size(0); // Размер группы
        
        // Локальная память очень быстрая, аналог кеша процессора,
        // Распределяем как нам нужно этот буффер
        __local uchar* sharedArrayR = sharedArray;
        __local uchar* sharedArrayG = sharedArray + groupSize * BIN_SIZE;
        __local uchar* sharedArrayB = sharedArray + 2 * groupSize * BIN_SIZE;
        
        // Заполняем нулями наш кеш
        for(int i = 0; i < BIN_SIZE; ++i){
            sharedArrayR[localId * BIN_SIZE + i] = 0;
            sharedArrayG[localId * BIN_SIZE + i] = 0;
            sharedArrayB[localId * BIN_SIZE + i] = 0;
        }
        
        // Здесь все потоки дожидаются завершения заполнения нулями локальной памяти
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Вычисляем гистограмму
        for(int i = 0; i < BIN_SIZE; ++i){
            // Получаем пиксель
            uint value = data[globalId * BIN_SIZE + i];
            // Получаем компоненты пикселя
            uint valueR = value & 0xFF;
            uint valueG = (value & 0xFF00) >> 8;
            uint valueB = (value & 0xFF0000) >> 16;
            // Для полученной градации цвета увеличиваем значение у локальной переменной
            sharedArrayR[localId * BIN_SIZE + valueR]++;
            sharedArrayG[localId * BIN_SIZE + valueG]++;
            sharedArrayB[localId * BIN_SIZE + valueB]++;
        }
        
        // Дожидаемся завершения
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Сливаем гистограмы каждого потока в общее значение
        for(int i = 0; i < BIN_SIZE / groupSize; ++i) {
            uint binCountR = 0;
            uint binCountG = 0;
            uint binCountB = 0;
            for(int j = 0; j < groupSize; ++j) {
                binCountR += sharedArrayR[j * BIN_SIZE + i * groupSize + localId];
                binCountG += sharedArrayG[j * BIN_SIZE + i * groupSize + localId];
                binCountB += sharedArrayB[j * BIN_SIZE + i * groupSize + localId];
            }
            
            // Сохраняем значение в большие буфферы
            binResultR[groupId * BIN_SIZE + i * groupSize + localId] = binCountR;
            binResultG[groupId * BIN_SIZE + i * groupSize + localId] = binCountG;
            binResultB[groupId * BIN_SIZE + i * groupSize + localId] = binCountB;
        }
    }
);


int main(int argc, char *argv[])
{
    cl_int status = 0;
    cl_int binSize = 256;
    cl_int groupSize = 16;
    cl_int subHistgCnt;
    cl_device_type dType = CL_DEVICE_TYPE_GPU;
    cl_platform_id platform = NULL;
    cl_device_id   device;
    cl_context     context;
    cl_command_queue commandQueue;
    cl_mem         imageBuffer;
    cl_mem     intermediateHistR, intermediateHistG, intermediateHistB; /*Intermediate Image Histogram buffer*/
    cl_uint *  midDeviceBinR, *midDeviceBinG, *midDeviceBinB;
    cl_uint  *deviceBinR,*deviceBinG,*deviceBinB;
    
    // Читаем BMP картинку и выделяем буфферы
    Image* image = nullptr;
    std::string filename = "sample_color.bmp";
    ReadBMPImage(filename, &image);
    if(image == NULL){
        printf("File %s not present...\n", filename.c_str());
        return 0;
    }
    subHistgCnt  = (image->width * image->height)/(binSize*groupSize);
    midDeviceBinR = (cl_uint*)malloc(binSize * subHistgCnt * sizeof(cl_uint));
    midDeviceBinG = (cl_uint*)malloc(binSize * subHistgCnt * sizeof(cl_uint));
    midDeviceBinB = (cl_uint*)malloc(binSize * subHistgCnt * sizeof(cl_uint));
    deviceBinR    = (cl_uint*)malloc(binSize * sizeof(cl_uint));
    deviceBinG    = (cl_uint*)malloc(binSize * sizeof(cl_uint));
    deviceBinB    = (cl_uint*)malloc(binSize * sizeof(cl_uint));
    
    // Получаем платформу
    status = clGetPlatformIDs(1, &platform, NULL);
    LOG_OCL_ERROR(status, "clGetPlatformIDs Failed." );

    // Получаем девайс нужного типа
    status = clGetDeviceIDs (platform, dType, 1, &device, NULL);
    LOG_OCL_ERROR(status, "clGetDeviceIDs Failed." );
    
    // Создаем вычислительный контекст
    cl_context_properties cps[3] = 
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platform,
        0
    };
    context = clCreateContextFromType(
        cps,
        dType,
        NULL,
        NULL,
        &status);
    LOG_OCL_ERROR(status, "clCreateContextFromType Failed." );

    // Создаем командную очередь
    commandQueue = clCreateCommandQueue(context,
                                        device,
                                        0,
                                        &status);
    LOG_OCL_ERROR(status, "clCreateCommandQueue Failed." );
    
#if !defined(USE_HOST_MEMORY)
    // Создаем входной буффер размером с картинку
    imageBuffer = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY,
        sizeof(cl_uint) * image->width * image->height,
        NULL,
        &status); 
    LOG_OCL_ERROR(status, "clCreateBuffer Failed while creating the image buffer." );

    // Копируем данные картинки в буффер асинхронно
    cl_event writeEvt;
    status = clEnqueueWriteBuffer(commandQueue,
                                  imageBuffer,
                                  CL_FALSE,
                                  0,
                                  image->width * image->height * sizeof(cl_uint),
                                  image->pixels,
                                  0,
                                  NULL,
                                  &writeEvt);
    LOG_OCL_ERROR(status, "clEnqueueWriteBuffer Failed while writing the image data." );
#else
    // Создаем входной буффер размером с картинку сразу копируя данные
    imageBuffer = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR,
        sizeof(cl_uint) * image->width * image->height,
        image->pixels,
        &status); 
    LOG_OCL_ERROR(status, "clCreateBuffer Failed while creating the image buffer." );
#endif
    
    // Ждем завершения копирования
    status = clFinish(commandQueue);
    LOG_OCL_ERROR(status, "clFinish Failed while writing the image data." );
    
    // Создаем выходные буфферы
    intermediateHistR = clCreateBuffer(
        context, 
        CL_MEM_WRITE_ONLY,
        sizeof(cl_uint) * binSize * subHistgCnt, 
        NULL, 
        &status);
    LOG_OCL_ERROR(status, "clCreateBuffer Failed." );

    intermediateHistG = clCreateBuffer(
        context,
        CL_MEM_WRITE_ONLY,
        sizeof(cl_uint) * binSize * subHistgCnt,
        NULL,
        &status);
    LOG_OCL_ERROR(status, "clCreateBuffer Failed." );

    intermediateHistB = clCreateBuffer(
        context,
        CL_MEM_WRITE_ONLY,
        sizeof(cl_uint) * binSize * subHistgCnt,
        NULL,
        &status);
    LOG_OCL_ERROR(status, "clCreateBuffer Failed." );

    // Создаем вычислительную программу
    cl_program program = clCreateProgramWithSource(context, 1,
            (const char **)&histogram_kernel, NULL, &status);
    LOG_OCL_ERROR(status, "clCreateProgramWithSource Failed." );

    // Копилируем
    status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if(status != CL_SUCCESS)
        LOG_OCL_COMPILER_ERROR(program, device);

    // Создаем вычислительное ядро с нужной функцией
    cl_kernel kernel = clCreateKernel(program, "histogram_kernel", &status);
    LOG_OCL_ERROR(status, "clCreateKernel Failed." );
    
    // Аргументы для ядра
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&imageBuffer); // Обычный буффер
    status |= clSetKernelArg(kernel, 1, 3 * groupSize * binSize * sizeof(cl_uchar), NULL);  // Локальный буфер для вычислительной группы с высокой скоростью
    status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&intermediateHistR);
    status |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&intermediateHistG);
    status |= clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&intermediateHistB);
    LOG_OCL_ERROR(status, "clSetKernelArg Failed." );
    
    // Выполняем вычислительное OpenCL ядро
    cl_event ndrEvt;
    size_t globalThreads = (image->width * image->height) / (binSize*groupSize) * groupSize;
    size_t localThreads = groupSize;
    status = clEnqueueNDRangeKernel(
        commandQueue,
        kernel,
        1,
        NULL,
        &globalThreads,
        &localThreads,
        0,
        NULL,
        &ndrEvt);
    LOG_OCL_ERROR(status, "clEnqueueNDRangeKernel Failed." );

    status = clFinish(commandQueue);
    LOG_OCL_ERROR(status, "clFinish Failed." );

    // Вычитываем гистограмму назад
    memset(deviceBinR, 0, binSize * sizeof(cl_uint));
    memset(deviceBinG, 0, binSize * sizeof(cl_uint));
    memset(deviceBinB, 0, binSize * sizeof(cl_uint));
    cl_event readEvt[3];
    status = clEnqueueReadBuffer(
        commandQueue,
        intermediateHistR,
        CL_FALSE,
        0,
        subHistgCnt * binSize * sizeof(cl_uint),
        midDeviceBinR,
        0,
        NULL,
        &readEvt[0]);
    LOG_OCL_ERROR(status, "clEnqueueReadBuffer of intermediateHistR Failed." );
    
    status = clEnqueueReadBuffer(
        commandQueue,
        intermediateHistG,
        CL_FALSE,
        0,
        subHistgCnt * binSize * sizeof(cl_uint),
        midDeviceBinG,
        0,
        NULL,
        &readEvt[1]);
    LOG_OCL_ERROR(status, "clEnqueueReadBuffer of intermediateHistG Failed." );
    
    status = clEnqueueReadBuffer(
        commandQueue,
        intermediateHistB,
        CL_FALSE,
        0,
        subHistgCnt * binSize * sizeof(cl_uint),
        midDeviceBinB,
        0,
        NULL,
        &readEvt[2]);
    LOG_OCL_ERROR(status, "clEnqueueReadBuffer of intermediateHistB Failed." );
    
    status = clWaitForEvents(3, readEvt);
    //status = clFinish(commandQueue);
    LOG_OCL_ERROR(status, "clWaitForEvents for readEvt." );

    // Calculate final histogram bin 
    for(int i = 0; i < subHistgCnt; ++i)
    {
        for(int j = 0; j < binSize; ++j)
        {
            deviceBinR[j] += midDeviceBinR[i * binSize + j];
            deviceBinG[j] += midDeviceBinG[i * binSize + j];
            deviceBinB[j] += midDeviceBinB[i * binSize + j];
        }
    }

    // Validate the histogram operation. 
    // The idea behind this is that once a histogram is computed the sum of all the bins should be equal to the number of pixels.
    int totalPixelsR = 0;
    int totalPixelsG = 0;
    int totalPixelsB = 0;
    for(int j = 0; j < binSize; ++j)
    {
        totalPixelsR += deviceBinR[j];
        totalPixelsG += deviceBinG[j];
        totalPixelsB += deviceBinB[j];
    }
    printf ("Total Number of Red Pixels = %d\n",totalPixelsR);
    printf ("Total Number of Green Pixels = %d\n",totalPixelsG);
    printf ("Total Number of Blue Pixels = %d\n",totalPixelsB);
    ReleaseBMPImage(&image);
    //free all allocated memory
    free(midDeviceBinR);
    free(midDeviceBinG);
    free(midDeviceBinB);
    free(deviceBinR);
    free(deviceBinG);
    free(deviceBinB);

    return 0;
}
