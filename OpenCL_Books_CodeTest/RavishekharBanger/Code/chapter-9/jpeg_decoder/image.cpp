#ifndef IMAGE_CPP
#define IMAGE_CPP
#ifdef WIN32
#include <io.h>
#else
#include <sys/types.h>
#include <unistd.h>
#endif
#include "image.h"
#include <stdio.h>
#include <iostream>
#include "Framewave.h"
#include <ocl_macros.h>


int writeBmpFile(const char *fName, unsigned char *pImageBuffer, unsigned long int imageSize, unsigned int width, unsigned int height);

Fw64u FileLength(int fd)
{
    Fw64u size =0;
#ifdef WIN32
    size = _lseek(fd,0,SEEK_END);
#else
    size = lseek(fd,0,SEEK_END);
#endif
    return size;
}

int image::open(const char *inputFile)
{
    int fd;
    
#ifdef WIN32
    fopen_s(&fp, inputFile, "rb");
#else
    fp = fopen(inputFile, "rb");
#endif

    if (fp == NULL)
        return OCL_FAILURE;
    
    fd = fileno(fp);
    fileLength = FileLength(fd);
    rewind (fp);
    rawImageBuffer = (unsigned char*) malloc((size_t)fileLength);
    PrintInfo(true,"\nInfo : rawImageBuffer empty..filling rawImageBuffer\n");
    if(!feof(fp))
    {
        fread((void *)rawImageBuffer, sizeof(unsigned char),(size_t)fileLength, fp);        
        curIndex = 0;
        if(ferror(fp))
        {
            PrintError((ferror(fp)),"\nerror: could not read from file\n");
            curIndex = 0;
            fileLength =  0;
        }
        if(feof(fp))
        {
            PrintInfo((feof(fp)),"\nInfo : End of file occured during read\n");
            return OCL_FAILURE;
        }
    }
    return OCL_SUCCESS;
}

void image::write(const char *fname, Fw8u *imageYCbCr)
{
    Fw8u *finalImage;
    FwiSize roi; 
    Fw8u *temporary[3];
    roi.width = imageWidth;   
    roi.height = imageHeight; 
    temporary[0] = imageYCbCr;
    temporary[1] = imageYCbCr + (imageWidth*imageHeight);
    temporary[2] = imageYCbCr + 2 * (imageWidth*imageHeight);

    finalImage = (Fw8u *)malloc(sizeof(Fw8u)*imageWidth*imageHeight*noOfComponents);
#if ENABLE_FRAMEWAVE
    fwiYCbCrToBGR_JPEG_8u_P3C3R(temporary,imageWidth, finalImage,imageWidth*noOfComponents,roi );
#endif

    writeBmpFile(fname,finalImage,imageWidth*imageHeight*noOfComponents,imageWidth,imageHeight);
    free (finalImage);

}

void image::close()
{
    free(rawImageBuffer);
    fclose(fp);
}

unsigned char image::readByte()
{
    return image::rawImageBuffer[image::curIndex++];
}

unsigned short int image::getNext2Bytes()
{
    Fw16u next2Bytes;
    next2Bytes = ((rawImageBuffer[curIndex] << 8) | (rawImageBuffer[curIndex+1]));
    curIndex = curIndex + 2; 
    return next2Bytes;
}

void image::moveCurIndex(Fw64u noOfBytes)
{
    Fw64u moveto = curIndex + noOfBytes;
    PrintError((moveto < curIndex),"\nError: curIndex is being decremented\n");
    curIndex = moveto;
}

unsigned char image::findNextMarker()
{
    unsigned char nextMarker;
    while((nextMarker = readByte()) == 0xFF); // to skip any stuffed FF bytes
    return nextMarker;
}

void image::decode()
{
    unsigned char nextMarker;

    nextMarker = findNextMarker();
    if(nextMarker == SOI)
    {
        PrintInfo((nextMarker == SOI),"\nInfo : Found SOI (Start of Image) marker\n");
        //setupDecoder(); //TODO:
        // keep processing markers until you see any SOF marker
        nextMarker = findNextMarker();
        while(((nextMarker!=SOF0)&&(nextMarker!=SOF1)&&(nextMarker!=SOF2)&&(nextMarker!=SOF3)&&(nextMarker!=SOF5)&&(nextMarker!=SOF6)&&(nextMarker!=SOF7)&&(nextMarker!=SOF9)&&(nextMarker!=SOFA)&&(nextMarker!=SOFB)&&(nextMarker!=SOFD)&&(nextMarker!=SOFE)&&(nextMarker!=SOFF)))		
        {
            processTablenMisc(nextMarker);
            //std::cout << "Next Marker = " << nextMarker << "\n";
            nextMarker = findNextMarker();
        }
        decodeFrame(nextMarker);
    }
    else
        PrintError((nextMarker != SOI),"\nerror: could not find SOI marker\n");
}// end decodeImage() fn.

void image::processTablenMisc(unsigned char marker)
{
    switch(marker)
    {
    case DHT:
        {
            processDHT();
            break;
        }
    case DQT:
        {
           processDQT();
           break;
        }
    case DRI:
        {
            processDRI();
            break;
        }
    case DAC:
        {
            PrintError(true,"\nerror: Found DAC marker.. not supported\n");
            break;
        }
    case COM:
        {
            PrintInfo(true,"\nInfo : Found COM marker.. skipping it. Comment Marker\n");
            unsigned short int length = image::getNext2Bytes(); // get length of the segment
            image::moveCurIndex(length - 2); // skip the segment
            break;
        }
    case APP0:
        {
            APP0::processAPP0();
            break;
        }
    case APP1: 
    case APP2: 
    case APP3: 
    case APP4: 
    case APP5: 
    case APP6: 
    case APP7: 
    case APP8: 
    case APP9: 
    case APPA: 
    case APPB: 
    case APPC: 
    case APPD: 
    case APPE:
        {
            PrintInfo(true,"\nInfo : Found unsupported APPn marker, skipping it\n");
            unsigned short int length = image::getNext2Bytes(); // get length of the segment
            image::moveCurIndex(length - 2); // skip the segment
            break;
        }
    default: PrintError(true,"\nError: Invalid marker\n");
    }
}

void image::decodeFrame(unsigned char SOFmarker)
{
    switch(SOFmarker)
    {
    case SOF0:
        {
            SOF0::processSOF0();
            break;
        }
    case SOF1:
    case SOF2:
    case SOF3:
    case SOF5:
    case SOF6:
    case SOF7:
    case SOF9:
    case SOFA:
    case SOFB:
    case SOFD:
    case SOFE:
    case SOFF:
        {
            PrintError(1,"\nError: Found unsupported SOF marker..this mode of encoding not supported right now\n");
            break;
        }
    default:PrintError(1,"\nError: Found invalid frame marker\n");
    }

    unsigned char nextMarker;
    nextMarker = findNextMarker();
    while(nextMarker!=SOS)
        {    // keep processing markers until you see SOS marker
            processTablenMisc(nextMarker);
            nextMarker = findNextMarker();
        }
        decodeScan();
}

void image::processDHT()
{
    unsigned short int length;
    unsigned char tableClass;
    unsigned char tableSelector;
    unsigned char nextByte;
    unsigned long int count;//contains the number of values given
    int errNo;

    typedef struct
    {
    unsigned long count;     // contains the number of values in huffVal i.e it's length
    unsigned char bits[16];  // bits[i] contains the number of Huffman Codes of length i 
    unsigned char *huffVal;  // contains the values for the codes
    }HuffData;
    
    HuffData dcTableData[4];
    HuffData acTableData[4];

    length = image::getNext2Bytes();
    nextByte = image::readByte();
    tableClass = nextByte>>4;
    tableSelector = (nextByte&0x0F); 

    count = 0; 
    switch(tableClass)
    {
    case 0: // DC Table
        {
            for(int i=0;i<16;i++)// read huff bits list
            {	 
                dcTableData[tableSelector].bits[i] = image::readByte();
                count = count + dcTableData[tableSelector].bits[i]; // keep track of no. of vals(needed to allocate memory for huff vals)
            }
            dcTableData[tableSelector].count = count;			
            dcTableData[tableSelector].huffVal = (unsigned char *) malloc(count * sizeof(unsigned char));
            for(unsigned long int i=0;i<count;i++)// read huff values
                dcTableData[tableSelector].huffVal[i] = image::readByte();
            // TODO: if destinations are already installed with tables.. then free that memory
            errNo = 0;
#if ENABLE_FRAMEWAVE
            errNo = fwiDecodeHuffmanSpecInitAlloc_JPEG_8u(dcTableData[tableSelector].bits, dcTableData[tableSelector].huffVal, &pHuffDcTable[tableSelector]);
#endif
            PrintError((errNo!=0),"\nerror: error creating huffman dc table : in fwiDecodeHuffmanSpecInitAlloc_JPEG_8u\n");
            break;
        }

    case 1: // AC Table
        {
            for(int i=0;i<16;i++)
            {	 
                acTableData[tableSelector].bits[i] = image::readByte();
                count = count + acTableData[tableSelector].bits[i]; // keep track of no. of vals(needed to allocate memory for huff vals)
            }
            acTableData[tableSelector].count = count;
            acTableData[tableSelector].huffVal = (unsigned char *) malloc(count * sizeof(unsigned char));		
            for(unsigned long int i=0;i<count;i++)//read huff values
                acTableData[tableSelector].huffVal[i] = image::readByte();
            // TODO: if destinations are already installed with tables.. then free that memory
            errNo =0;
#if ENABLE_FRAMEWAVE
            errNo = fwiDecodeHuffmanSpecInitAlloc_JPEG_8u(acTableData[tableSelector].bits, acTableData[tableSelector].huffVal, &pHuffAcTable[tableSelector]);
#endif
            PrintError((errNo!=0),"\nerror: error creating huffman ac table : in fwiDecodeHuffmanSpecInitAlloc_JPEG_8u\n");
            break;
        }
    }//end switch

    PrintInfo(true,"\nInfo : found DHT marker\n");
    PrintInfo(true,"\n********************** DHT HEADER ******************************\n");
    PrintInfo(true,"\nLength : %d\n", length);
    PrintInfo((tableClass==0),"\nHuffman table type : DC table\n");
    PrintInfo((tableClass==1),"\nHuffman table type : AC table\n");
    PrintInfo(true,"\nTable destination : %u\n", tableSelector);
    unsigned long int indexLow = 0, indexHigh = 0;
    for(unsigned int i=0;i<16;i++)
    {
        PrintInfo(tableClass==0,"\nNumber of Huffman codes of length %u : %u",i+1,dcTableData[tableSelector].bits[i]);
        PrintInfo(tableClass==1,"\nNumber of Huffman codes of length %u : %u",i+1,acTableData[tableSelector].bits[i]);
        indexHigh = (tableClass == 0) ? indexLow + dcTableData[tableSelector].bits[i] : indexLow + acTableData[tableSelector].bits[i];
        PrintInfo(true,"\n");
        for(unsigned long int j=indexLow; j<indexHigh; j++)		
            PrintInfo(true,"%u ",(tableClass==0) ? dcTableData[tableSelector].huffVal[j] : acTableData[tableSelector].huffVal[j]);

        indexLow = indexHigh;
        PrintInfo(true,"\n");
    }
    PrintInfo(true,"\n****************************************************************\n");
}

void image::processDQT()
{
    unsigned short int length;
    unsigned char destination;
    unsigned char nextByte;
    unsigned char rawQuantTable[64];
    int noOfTables;
    int errNo;

    length = image::getNext2Bytes();
    noOfTables = (length - 2)/65;
    while(noOfTables!=0) // fill all the quantisation tables
    {
        nextByte = image::readByte();
        precision = nextByte>>4;
        destination = (nextByte&0x0F);
        switch(precision)
        {
            // TODO: use fwiZigzagFwd8x8_16s_C1 and fwiZigzagInv8x8_16s_C1 here...
        case  0: // 8-bit quantisation values
            {
                for(int i=0;i<64;i++)
                    rawQuantTable[i] = image::readByte(); 
                errNo = 0;
#if ENABLE_FRAMEWAVE
                errNo = fwiQuantInvTableInit_JPEG_8u16u(rawQuantTable, &(quantInvTable[destination][0]));
#endif
                PrintError((errNo!=0),"\nerror: errNo != 0 in fwiQuantInvTableInit_JPEG_8u16u\n");
                break;
            }
        case  1:
            PrintError(true,"\nerror: 16 bit quantisation values not supported\n");	
        default: PrintError(1,"\nError: precision of quantisation values not supported\n");
        }// end switch
        noOfTables = noOfTables - 1; // decrement noOfTables

        PrintInfo(true,"\ninfo : found DQT marker\n");
        PrintInfo(true,"\n********************** DQT HEADER ******************************\n");
        PrintInfo(true,"\nlength : %d\n", length); 

        PrintInfo((precision==0),"\nPrecision of quantisation values : 8 bits\n");
        PrintInfo((precision==1),"\nPrecision of quantisation values : 16 bits\n");
        PrintInfo(true,"\nDestination : %u\n", destination);
        PrintInfo(true,"\nDQT Table:\n\n");
        for(int i=0;i<8;i++)
        {
            for(int j=0;j<8;j++)
                PrintInfo(true,"%2u ", quantInvTable[destination][i*8+j]);
            PrintInfo(true,"\n");
        }
        PrintInfo(true,"\n****************************************************************\n");
    }// end while
}

void image::processDRI()
{
    unsigned short int length;

    length = getNext2Bytes();
    restartInterval = getNext2Bytes();
    if(restartInterval > 0)
        restartEnabled = true;

    PrintInfo(true,"\ninfo : found DRI marker\n");
    PrintInfo(true,"\n********************** DRI HEADER ******************************\n");
    PrintInfo(true,"\nlength : %u\n", length);
    PrintInfo(true,"\nrestart interval : %u \n", restartInterval);
    PrintInfo((restartEnabled == true),"\nrestart is enabled\n");
    PrintInfo((restartEnabled == false),"\nrestart is not enabled\n");
    PrintInfo(true,"\n****************************************************************\n");
}

void image::processScan()
{
    unsigned short int length;	
    unsigned char nextByte;

    length = image::getNext2Bytes();
    noOfComponents = image::readByte();
    for(int i=0;i<noOfComponents;i++)
    {
        component[i].ComponentId = image::readByte();       
        nextByte = image::readByte();
        component[i].DCTableSelector = nextByte>>4;        
        component[i].ACTableSelector = (nextByte&0x0F);        
    }
    startSpectralSelector = image::readByte();   
    endSpectralSelector = image::readByte();    
    nextByte = image::readByte();    
    bitPositionHigh = nextByte>>4;
    bitPositionLow = nextByte&4;

    PrintInfo(true,"\nfound SOS marker 0xFF%0x\n");
    PrintInfo(true,"\n********************** SOS HEADER ******************************\n");
    PrintInfo(true,"\nlength : %u\n", length);
    PrintInfo(true,"\nno. of components in the scan : %u\n",noOfComponents);
    PrintInfo(true,"\nsuccessive approximation bit position high: %u\n",bitPositionHigh); 
    PrintInfo(true,"\nsuccessive approximation bit position low : %u\n",bitPositionLow);
    for(int i=0;i<noOfComponents;i++)
        {
        PrintInfo(true,"\ncomponent ID : %u\n", component[i].ComponentId);
        PrintInfo(true,"\nDC table selector : %u\n", component[i].DCTableSelector);
        PrintInfo(true,"\nAC table selector : %u\n", component[i].ACTableSelector);
        }
    PrintInfo(true,"\nDCT coefficient for start of scan in zigzag order : %u\n", startSpectralSelector);
    PrintInfo(true,"\nlast DCT coefficient encoded in zigzag order : %u\n", endSpectralSelector);
    PrintInfo(true,"\n****************************************************************\n");
}

void image::decodeScan()
{
    int errNo;
    int srcCurPos;
    FwiDecodeHuffmanState* pDecHuffState;

    signed short int lastDc[4] = {0,0,0}; // we support only four components now...
    int marker = 0;
    bool eoiFound = false;
    processScan();
#if ENABLE_FRAMEWAVE
    errNo = fwiDecodeHuffmanStateInitAlloc_JPEG_8u(&pDecHuffState);
#endif
    PrintError((errNo!=0),"\nerror: in fwiDecodeHuffmanStateInitAlloc_JPEG_8u\n");
    srcCurPos = (int)image::curIndex;
    int tableSelector = 0;
    Fw16s pDataUnit[64]; 
    int unSampledDstStep;
    Fw8u *pTemporaryDst2[4];
    // TODO: can find this at the time of reading the header....
    int maxHorizontalSamplingFactor = 0;
    int maxVerticalSamplingFactor = 0;
    int noOfDataUnitsInMCU = 0;
    int noOfMCUforComponent = 0;

    noOfComponents = SOF0::noOfComponents;
    samplesPerLine = SOF0::samplesPerLine;
    noOfLines = SOF0::noOfLines;
    for(int i=0; i<noOfComponents;i++)
    {
        componentData[i].horizontalSamplingFactor = SOF0::componentData[i].horizontalSamplingFactor;
        componentData[i].verticalSamplingFactor = SOF0::componentData[i].verticalSamplingFactor;
        componentData[i].componentId = SOF0::componentData[i].componentId;
        componentData[i].DQTTableSelector = SOF0::componentData[i].DQTTableSelector;
        componentData[i].noOfMCUsForComponent = componentData[i].horizontalSamplingFactor*
                                                componentData[i].verticalSamplingFactor;
    }
    
    for(int i=0; i<noOfComponents;i++)
    {
        if(componentData[i].horizontalSamplingFactor > maxHorizontalSamplingFactor)
            maxHorizontalSamplingFactor = componentData[i].horizontalSamplingFactor;
        if(componentData[i].verticalSamplingFactor > maxVerticalSamplingFactor)
            maxVerticalSamplingFactor = componentData[i].verticalSamplingFactor;
        noOfDataUnitsInMCU = noOfDataUnitsInMCU + componentData[i].horizontalSamplingFactor * componentData[i].verticalSamplingFactor;
    }
    mcuWidth  = 8 * maxHorizontalSamplingFactor; // right now we support only lossy JPEG ..so 8 pixels
    mcuHeight = 8 * maxVerticalSamplingFactor;
    //TODO: calculate xPadding and yPadding here...

    noOfxMCU = (samplesPerLine + mcuWidth - 1)/ mcuWidth;
    noOfyMCU = (noOfLines + mcuHeight - 1) / mcuHeight;
    for(int i=0; i<noOfComponents;i++)
    {
        unSampledDstStep = noOfxMCU*componentData[i].horizontalSamplingFactor*8;
        pMCUdst[i] = (Fw16s*)malloc(unSampledDstStep*componentData[i].verticalSamplingFactor*8 * noOfyMCU * 2/*sizeof(Fw16u)*/);
        pTemporaryDst2[i] = (Fw8u*)malloc(unSampledDstStep*componentData[i].verticalSamplingFactor*8 * noOfyMCU * 1/*sizeof(Fw8u)*/);
    }
    /*Data for kernel processing */
    imageWidth = noOfxMCU * mcuWidth;
    imageHeight = noOfyMCU * mcuHeight;
    // get all MCU's..
    for(int yMCU=0;yMCU<noOfyMCU;yMCU++)
    // get one MCU row..
    {
        for(int xMCU=0;xMCU<noOfxMCU;xMCU++)
        // get one MCU..
        {
            for(int i=0;i<noOfComponents;i++)
            // TODO: the way we are referring componentData with the componentID may be wrong.. change it 
            {           
                for(unsigned int j=0;j<componentData[(component[i].ComponentId-1)].verticalSamplingFactor;j++)
                {
                    for(unsigned int k=0;k<componentData[component[i].ComponentId-1].horizontalSamplingFactor;k++)
                    {
                        // TODO: if restart interval is set then calculate nomber of restarts required and after that many number of MCU's  reset the decoder
                        marker = 0;
#if ENABLE_FRAMEWAVE
                        errNo = fwiDecodeHuffman8x8_JPEG_1u16s_C1(image::rawImageBuffer, (int)image::fileLength - 1, &srcCurPos, pDataUnit, &(lastDc[i]), &marker, pHuffDcTable[component[i].DCTableSelector], pHuffAcTable[component[i].ACTableSelector],pDecHuffState);
#endif
                        if(marker!=0)
                        {
                            PrintError(true,"\nError: marker is not zero\n");
                        }
                        PrintError((errNo!=0),"\nError : errNo != 0 in fwiDecodeHuffman8x8_JPEG_1u16s_C1\n");
                        image::moveCurIndex(srcCurPos - image::curIndex);

                        //copy to pTempDst
                        noOfMCUforComponent = componentData[component[i].ComponentId-1].verticalSamplingFactor * 
                                              componentData[component[i].ComponentId-1].horizontalSamplingFactor ;
                        for (int l=0;l<64;l++)
                        {
                            pMCUdst[i][yMCU * noOfxMCU *(noOfMCUforComponent *64) + (xMCU * noOfMCUforComponent *64 ) + ((j * componentData[component[i].ComponentId-1].horizontalSamplingFactor * 64) + k *64) + l] = pDataUnit[l];
                        }
                    }
                }
            }
        }
    }
    return;
}

void PrintInfo(int condition, const char *fmt_string, ...)  
{
#ifdef _printInfo_
        if(condition)
        {
                va_list arg;
                va_start(arg,fmt_string);
#ifdef WIN32
                vprintf_s(fmt_string,arg); 
#else
                vprintf(fmt_string,arg); 
#endif
                va_end(arg);
        }
#endif 
}

void PrintError(int condition, const char *fmt_string, ...)
{
#ifdef _printError_
        if(condition)
        {
                va_list arg;
                va_start(arg,fmt_string);
#ifdef WIN32
                vprintf_s(fmt_string,arg); 
#else
                vprintf(fmt_string,arg); 
#endif
                va_end(arg);
                getchar();
                exit(-1);
        }
#endif
}





#endif


