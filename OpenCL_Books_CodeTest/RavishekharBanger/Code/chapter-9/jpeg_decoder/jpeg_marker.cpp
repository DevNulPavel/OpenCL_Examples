#ifndef APP0_CPP
#define APP0_CPP
#include "jpeg_marker.h"
#include "Framewave.h"

Fw64u image::curIndex = 0;                                         
Fw8u* image::rawImageBuffer = NULL;

unsigned char APP0::units = 0;
unsigned short int APP0::xDensity = 0;
unsigned short int APP0::yDensity = 0;
unsigned char APP0::RGBn[1];

unsigned char SOF0::samplePrecision; 
unsigned short int SOF0::noOfLines;
unsigned short int SOF0::samplesPerLine;
unsigned char SOF0::noOfComponents; 
SOF0ComponentData_t SOF0::componentData[4]; 

//#include "APP0.h"
//#include "init.h"
void APP0::processAPP0()
{
    unsigned short int length;
    char identifier[5]; 
    unsigned char version[2];
    unsigned char xThumbnail;
    unsigned char yThumbnail; 

    length = image::getNext2Bytes();
    identifier[0] = image::readByte();
    identifier[1] = image::readByte();
    identifier[2] = image::readByte();
    identifier[3] = image::readByte();
    identifier[4] = image::readByte();

    version[0] = image::readByte();
    version[1] = image::readByte();

    units = image::readByte();

    xDensity = image::getNext2Bytes();
    yDensity = image::getNext2Bytes();
    xThumbnail = image::readByte();
    yThumbnail = image::readByte();

    // TODO: should read the RGBn here(for the thubnail).. skipping it for now
    image::moveCurIndex(length-16);

    PrintInfo(true,"\ninfo : found APP0 marker\n");
    PrintInfo(true,"\n********************** APP0 HEADER *****************************\n");
    PrintInfo(true,"\nlength : %u\n", length);
    PrintInfo(true,"\nidentifier : %s\n", identifier);
    PrintInfo(true,"\nversion : %d.%d\n", version[0], version[1]); 
    PrintInfo((units == 0),"\no units for X an Y Densities\n");
    PrintInfo((units == 1),"\nX and Y are dots per Inch\n");
    PrintInfo((units == 2),"\nX and Y are dots per CM\n");
    PrintInfo(true,"\nxDensity : %u\n", xDensity);
    PrintInfo(true,"\nyDensity : %u\n", yDensity);
    PrintInfo(true,"\nxThumbnail : %u\n", xThumbnail);
    PrintInfo(true,"\nyThumbnail : %u\n", yThumbnail);
    PrintInfo(true,"\n****************************************************************\n");
}


void SOF0::processSOF0()
{
        unsigned short int length;
        unsigned char nextByte;

        length = image::getNext2Bytes();              
        samplePrecision = image::readByte();
        noOfLines = image::getNext2Bytes();
        samplesPerLine = image::getNext2Bytes();
        noOfComponents = image::readByte(); 

        //read the component info
        for(int i=0;i<noOfComponents;i++)
        {
                componentData[i].componentId = image::readByte();
                nextByte = image::readByte();
                componentData[i].horizontalSamplingFactor = nextByte>>4;
                componentData[i].verticalSamplingFactor = (nextByte&0x0F);
                componentData[i].DQTTableSelector = image::readByte();
        }//end for loop

        PrintInfo(true,"\ninfo : found SOF0 marker : BaseLine DCT\n");
        PrintInfo(true,"\n********************** SOF0 HEADER *****************************\n");
        PrintInfo(true,"\nlength : %d\n", length);  
        PrintInfo(true,"\nno. of bits for each sample : %u\n",samplePrecision);
        PrintInfo(true,"\nnumber of lines in the image : %u\n",noOfLines);
        PrintInfo(true,"\nsamples per line : %u\n", samplesPerLine);
        PrintInfo(true,"\nnumber of components in the image : %u\n",noOfComponents);
        for(int i=0;i<noOfComponents;i++)
        {
                PrintInfo(true,"\ncomponent ID : %u\n", componentData[i].componentId);      
                PrintInfo(true,"\nhorizontal sampling factor : %u\n", componentData[i].horizontalSamplingFactor); 
                PrintInfo(true,"\nvertical sampling factor : %u\n",componentData[i].verticalSamplingFactor);
                PrintInfo(true,"\nquantisation table selector : %u\n",componentData[i].DQTTableSelector);
        }
        PrintInfo(true,"\n****************************************************************\n");
} //end readSOF0 fn


#endif

