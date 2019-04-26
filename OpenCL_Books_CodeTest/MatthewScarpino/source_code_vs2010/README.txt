This folder contains example code for the book OpenCL in Action by Matthew Scarpino. The code is provided as a zipped Visual Studio 2010 solution (oclia_vs2010.zip) along with folders for Chapter 9 (Ch9) and Appendix C (AppC).

In addition to the OpenCL library (OpenCL.lib/OpenCL.dll), many of the projects require additional libraries. Projects in Chapter 15, Chapter 16, and Appendix B require OpenGL, GLUT, and GLEW. Projects in Chapter 6, Chapter 16, and Appendix B require the PNG library. These libraries can be downloaded freely online.

Locations of libraries and header files are defined with environment variables. This solution assumes that the AMD SDK is being used, so the primary OpenCL variable is AMDAPPSDKROOT. To compile this solution on Nvidia-based systems, the NVSDKCOMPUTE_ROOT variable must be used instead.

All of the code is public domain, and you can do with it as you please.