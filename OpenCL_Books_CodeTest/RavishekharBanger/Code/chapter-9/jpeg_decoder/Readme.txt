-This folder contains the files related to the JPEG decoder.
-We have a reference implementation and two OpenCL implementation.
-The height and width of the input image should be a multiple of 32.
-To choose the device on which you want to run the JPEG Decoder by modifying the
 device type dType in function JPEG_Decoder::setupCL in file JPEG_Decoder.cpp,
 to the device you want to run on.



