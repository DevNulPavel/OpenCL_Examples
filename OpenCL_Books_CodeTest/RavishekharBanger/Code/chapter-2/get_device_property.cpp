#include <stdio.h>
#include <stdlib.h>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
//#include <ocl_macros.h>

void PrintDeviceInfo(cl_device_id device)
{
    char queryBuffer[1024];
    int queryInt;
    cl_int clError;
    clError = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(queryBuffer), &queryBuffer, NULL);
    printf("CL_DEVICE_NAME: %s\n", queryBuffer);
    queryBuffer[0] = '\0';
    clError = clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(queryBuffer), &queryBuffer, NULL);
    printf("CL_DEVICE_VENDOR: %s\n", queryBuffer);
    queryBuffer[0] = '\0';
    clError = clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(queryBuffer), &queryBuffer, NULL);
    printf("CL_DRIVER_VERSION: %s\n", queryBuffer);
    queryBuffer[0] = '\0';
    clError = clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(queryBuffer), &queryBuffer, NULL);
    printf("CL_DEVICE_VERSION: %s\n", queryBuffer);
    queryBuffer[0] = '\0';
    clError = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(int), &queryInt, NULL);
    printf("CL_DEVICE_MAX_COMPUTE_UNITS: %d\n", queryInt);
}

int main(void) {

    cl_int           clError;

    // Get the Number of Platforms available
    // Note that the second parameter "platforms" is set to NULL. If this is NULL then this argument is ignored
    // and the API returns the total number of OpenCL platforms available.
    cl_platform_id * platforms = NULL;

    cl_uint     num_platforms = 0;                                                       
    if ((clGetPlatformIDs(0, NULL, &num_platforms)) == CL_SUCCESS)                   
    {                         
        printf("Number of OpenCL platforms available in the system: %d\n", num_platforms);
        platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id)*num_platforms);  
        if(clGetPlatformIDs(num_platforms, platforms, NULL) != CL_SUCCESS)           
        {                                                                            
            free(platforms);                                                         
            printf("Error in call to clGetPlatformIDs....\n Exiting");
            exit(0);                                                  
        }                                                                            
    }                                                                                

    if (num_platforms == 0)
    {
        printf("No OpenCL Platforms Found ....\n Exiting");
        exit(0);      
    }
    else
    {
        // We have obtained one platform here.
        // Lets enumerate the devices available in this Platform.
        for (cl_uint idx=0;idx<num_platforms; idx++)
        {
            cl_device_id    *devices;
            cl_uint          num_devices;
            printf("\nPrinting OpenCL Device Info For Platform ID : %d\n", idx);
            clError = clGetDeviceIDs (platforms[idx], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
            if (clError != CL_SUCCESS)
            {
                printf("Error Getting number of devices... Exiting\n ");
                exit(0);
            }
            // If successfull the num_devices contains the number of devices available in the platform
            // Now lets get all the device list. Before that we need to malloc devices
            devices = (cl_device_id *)malloc(sizeof(cl_device_id) * num_devices);
            clError = clGetDeviceIDs (platforms[idx], CL_DEVICE_TYPE_ALL, num_devices, devices, &num_devices);
            if (clError != CL_SUCCESS)
            {
                printf("Error Getting number of devices... Exiting\n ");
                exit(0);
            }
            
            for (cl_uint dIndex = 0; dIndex < num_devices; dIndex++)
            {
                    printf("==================Device No %d======================\n",dIndex);
                    PrintDeviceInfo(devices[dIndex]);
                    printf("====================================================\n\n");
            }
        }
    }
}