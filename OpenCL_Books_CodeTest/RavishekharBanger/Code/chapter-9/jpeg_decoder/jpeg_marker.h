#ifndef APP0_H
#define APP0_H


//#include "debug.h"

typedef struct
    {
    unsigned char componentId;
    unsigned char horizontalSamplingFactor;
    unsigned char verticalSamplingFactor;
    unsigned char DQTTableSelector;
    int noOfMCUsForComponent;
    }SOF0ComponentData_t;
#include "image.h"
class SOF0
    {
    public:
        static unsigned char samplePrecision; //specifies the no. of bits for the samples(pixels in the image)
        static unsigned short int noOfLines; // specifies the maximum number of lines in the source image
        static unsigned short int samplesPerLine; //specifies the no. of samples per line
        static unsigned char noOfComponents; // specifies the number of components in the frame
        static SOF0ComponentData_t componentData[4]; // This is the reason we support only 4 components now(increase this if requied) 

    public:
        static void processSOF0();
        
    };

class APP0
{
    public:
        static unsigned char units;
        static unsigned short int xDensity;
        static unsigned short int yDensity;
        static unsigned char RGBn[1]; // need to be done later..contains RGB values for the thumbnail

    public:
        static void processAPP0();
};

#endif

