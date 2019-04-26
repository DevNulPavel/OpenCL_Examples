#pragma once
#ifndef OCL_GAUSSIAN_H
#define OCL_GAUSSIAN_H
#ifdef __APPLE__
#include <OpenCL/cl.h>
#include <OpenCL/cl_gl.h>
#else
#include <CL/cl.h>
#include <CL/cl_gl.h>
#endif
#include "Timer.h"
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <bmp_image.h>
#include <ocl_macros.h>

using std::string;

#define		WINDOW_SIZE   3

class ImageFilter
{
public:
    //Constructors
    ImageFilter(const string &filename);
    //Destructors
	~ImageFilter(){cleanup();}
	void	cleanup();		// cleanup host read image and free image buffers

    //Initialize OpenCL stuffs
	void	init_GPU_OpenCL();		// init GPU FE mem, intit CL runtime, build program, 
        					        // build kernels, init CL images/buffers
    void	setupOCLPlatform();
    cl_int  setupOCLProgram();
    cl_int  setupOCLkernels();
    cl_int  setupOCLbuffers();
    void    setup_filter( );

	void	run_GPU();
    void    run_gaussian_filter_kernel();
    //Read and write image uses bmp_image.h
	void	read_GPU_filtered_image();
    void    load_GPU_raw_image();         
    
    void	start_GPU_Timer()		{timer_GPU.Reset(); timer_GPU.Start();}
    void	stop_GPU_Timer()		{timer_GPU.Stop();}
    void	print_GPU_Timer();

    void    load_bmp_image( );
    void    write_bmp_image( );
    Image * image;
private:
    //OpenCL Stuffs
    cl_platform_id              platform; 
    cl_device_type              deviceType;
    cl_device_id                device;
    cl_context                  context;
    cl_command_queue            commandQueue;
    cl_program                  program;
    cl_kernel                   gd_kernel;
    size_t						gwsize[2];// OpenCL global work size
    size_t						lwsize[2];// OpenCL local work size
    cl_mem                      ocl_filter, ocl_raw, ocl_filtered_image;
    
    std::string                 filename;
    int windowSize;
    float*      				host_image;					//load image data sequence 
    float						*GPU_raw,	*GPU_filtered_image;
    float                       filter[WINDOW_SIZE*WINDOW_SIZE];
    float*				        GPU_output;
    Timer				        timer_GPU;
};
#endif
