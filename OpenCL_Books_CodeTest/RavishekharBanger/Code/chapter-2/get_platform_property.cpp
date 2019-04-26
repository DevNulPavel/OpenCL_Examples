#include <stdio.h>
#include <stdlib.h>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

void PrintPlatformInfo(cl_platform_id platform)
{
    char queryBuffer[1024];
    cl_int clError;

    clError = clGetPlatformInfo (platform, CL_PLATFORM_NAME, 1024, &queryBuffer, NULL);
    if(clError == CL_SUCCESS)
    {
        printf("CL_PLATFORM_NAME   : %s\n", queryBuffer);
    }
    clError = clGetPlatformInfo (platform, CL_PLATFORM_VENDOR, 1024, &queryBuffer, NULL);
    if(clError == CL_SUCCESS)
    {
        printf("CL_PLATFORM_VENDOR : %s\n", queryBuffer);
    }
    clError = clGetPlatformInfo (platform, CL_PLATFORM_VERSION, 1024, &queryBuffer, NULL);
    if (clError == CL_SUCCESS)
    {
        printf("CL_PLATFORM_VERSION: %s\n", queryBuffer);
    }
    clError = clGetPlatformInfo (platform, CL_PLATFORM_PROFILE, 1024, &queryBuffer, NULL);
    if (clError == CL_SUCCESS)
    {
        printf("CL_PLATFORM_PROFILE: %s\n", queryBuffer);
    }
    clError = clGetPlatformInfo (platform, CL_PLATFORM_EXTENSIONS, 1024, &queryBuffer, NULL);
    if (clError == CL_SUCCESS)
    {
        printf("CL_PLATFORM_EXTENSIONS: %s\n", queryBuffer);
    }
    return;
}

int main(void) {
    cl_int           clError;
    cl_platform_id * platforms = NULL;
    cl_uint          num_platforms;
    // Get the number of Platforms available
    // Note that the second parameter "platforms" is set to NULL. 
    // If this is NULL then this argument is ignored
    // and the API returns the total number of OpenCL platforms available.
    clError = clGetPlatformIDs(0, NULL, &num_platforms);
    if(clError != CL_SUCCESS)
    {
        printf("Error in call to clGetPlatformIDs....\n Exiting");
        exit(0);
    }
    else
    {
        if (num_platforms == 0)
        {
            printf("No OpenCL Platforms Found ....\n Exiting");
        }
        else
        {
            //Allocate memory for OpenCL platforms.
            printf ("Found %d Platforms\n", num_platforms);
            platforms = (cl_platform_id *)malloc(num_platforms*sizeof(cl_platform_id));
            // Get the platform id's
            // In contrast to the above call with "platforms" as NULL. The below call actually fills the buffer with
            // the platform IDs. It will list the platforms upto the value specified by num_platforms. One should make
            // sure that the appropriate buffer size is allocated.
            clError = clGetPlatformIDs (num_platforms, platforms, NULL);
            // for each platform now start printing the information
            for(cl_uint index=0;index<num_platforms; index++)
            {
                    printf("==================Platform No %d======================\n",index);
                    PrintPlatformInfo(platforms[index]);
                    printf("======================================================\n\n");            
            }
        }
    }
    return 0;
}
