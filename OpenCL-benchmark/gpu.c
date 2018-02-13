#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef __APPLE__
  #include <OpenCL/opencl.h>
#else
  #include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)

int main() {
  cl_uint ret_num_devices;
  cl_uint ret_num_platforms;
  cl_int ret = 0;

  cl_ulong val[1];

  char fileName[] = "./sum.cl";
  size_t source_size = 0;

  FILE* fp = fopen(fileName, "r");
  if (!fp) {
    fprintf(stderr, "Failed to load kernel\n");
    exit(1);
  }

  // Читаем текст вычислительного шейдера
  char* source_str = (char*)malloc(MAX_SOURCE_SIZE);
  source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
  fclose(fp);

  // Получаем количество платформ
  cl_platform_id platformId = NULL;
  ret = clGetPlatformIDs(1, &platformId, &ret_num_platforms);

  cl_device_id device_id = NULL;
  ret = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

  cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
  cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
  cl_mem memobj = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_mem), NULL, &ret);
  cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
  ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
  cl_kernel kernel = clCreateKernel(program, "hello", &ret);

  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&memobj);

  clock_t begin = clock();

  ret = clEnqueueTask(command_queue, kernel, 0, NULL, NULL);

  ret = clEnqueueReadBuffer(command_queue, memobj, CL_TRUE, 0, sizeof(cl_mem), val, 0, NULL, NULL);

  clock_t end = clock();
  double runtime = (double)(end - begin) / CLOCKS_PER_SEC;

  ret = clFlush(command_queue);
  ret = clFinish(command_queue);
  ret = clReleaseKernel(kernel);
  ret = clReleaseProgram(program);
  ret = clReleaseMemObject(memobj);
  ret = clReleaseCommandQueue(command_queue);
  ret = clReleaseContext(context);

  printf("Result: %llu\n", val[0]);
  printf("Runtime: %lfms\n", runtime);
  
  free(source_str);

  return 0;
}
