
/* This file is a work derived from the Open Source library, Framewave
 * http://framewave.sourceforge.net/
 * All the function definitions and declarartions are derived from the files located at 
 * http://sourceforge.net/p/framewave/code/HEAD/tree/trunk/Framewave/domain/fwJPEG/ 
 * Below is the licence information.*/
/*
Copyright (c) 2006-2013 Advanced Micro Devices, Inc. All Rights Reserved.
This software is subject to the Apache v2.0 License.
*/

#if !defined(EXTERNAL_FRAMEWAVE_H)
#define EXTERNAL_FRAMEWAVE_H

typedef unsigned long Fw64u;
typedef unsigned char Fw8u;
typedef unsigned short Fw16u;
typedef signed short Fw16s;
typedef unsigned int Fw32u;


struct DecodeHuffmanSpec 
{ 
    Fw8u  pListVals[256];				//Copy of pListVals from SpecInit
    Fw16u symcode[256];            // symbol code
    Fw16u symlen[256];             // symbol code length

    Fw16s mincode[18];             // smallest code value of specified length I
    Fw16s maxcode[18];             // largest  code value of specified length I 
    Fw16s valptr[18];              // index to the start of HUFFVAL decoded by code words of Length I 
};

struct DecodeHuffmanState 
{
    Fw8u  *pCurrSrc;               //Current source pointer position
    int srcLenBytes;                //Bytes left in the current source buffer
    Fw32u accbuf;                  //accumulated buffer for extraction
    int accbitnum;                  //bit number for accumulated buffer
    int EOBRUN;                     //EOB run length
    int marker;                     //JPEG marker
};

typedef struct DecodeHuffmanSpec  FwiDecodeHuffmanSpec;
typedef struct DecodeHuffmanState FwiDecodeHuffmanState;



typedef struct
{
    int width;
    int height;
} FwiSize;
#define  FwStatus int
#define  fwStsNoErr               0
#define  fwStsNullPtrErr         -1   
#define  fwStsJPEGHuffTableErr   -2
#define  fwStsSizeErr            -3
#define  fwStsJPEGOutOfBufErr    -4
#define  fwStsStepErr            -5
#define  fwMalloc    malloc

#define DEC_EXTEND(V,T)  (V < (1<<(T-1)) ? (V + ((-1)<<T) + 1) : V)

#define GET_ACCBITS(pDecHuffState, s) \
    (((int) (pDecHuffState->accbuf >> (pDecHuffState->accbitnum -= (s)))) & ((1<<(s))-1))
#define STEPCHECK(X, Y) if (X<=0 || Y<=0) return fwStsStepErr
#define ROISIZECHECK(X) if (X.height <=0 || X.width <=0) return fwStsSizeErr

bool dec_receivebits (FwiDecodeHuffmanState * pDecHuffState, Fw32u accbuf, 
                      int accbitnum, int ssss);

int dec_huff (FwiDecodeHuffmanState * pDecHuffState, Fw32u accbuf, 
              int accbitnum, const FwiDecodeHuffmanSpec *pTable, int nbits);

bool FW_HUFF_DECODE(int *result, FwiDecodeHuffmanState *pDecHuffState, 
                     const FwiDecodeHuffmanSpec *pTable);

FwStatus fwiDecodeHuffmanStateGetBufSize_JPEG_8u(int* size);

FwStatus fwiDecodeHuffmanStateInitAlloc_JPEG_8u(
    FwiDecodeHuffmanState** pDecHuffState);

FwStatus fwiDecodeHuffmanSpecInit_JPEG_8u(
    const Fw8u *pListBits, const Fw8u *pListVals, FwiDecodeHuffmanSpec *pDecHuffSpec);

FwStatus fwiDecodeHuffmanSpecInitAlloc_JPEG_8u(
    const Fw8u *pListBits, const Fw8u *pListVals, FwiDecodeHuffmanSpec** pDecHuffSpec);

FwStatus fwiDecodeHuffman8x8_JPEG_1u16s_C1(
    const Fw8u *pSrc, int srcLenBytes, int *pSrcCurrPos, Fw16s *pDst,
    Fw16s *pLastDC, int *pMarker, const FwiDecodeHuffmanSpec *pDcTable,
    const FwiDecodeHuffmanSpec *pAcTable, FwiDecodeHuffmanState *pDecHuffState);

FwStatus fwiYCbCrToBGR_JPEG_8u_P3C3R(const Fw8u * const pSrcYCbCr[3], int srcStep,
                                     Fw8u *pDstBGR, int dstStep, FwiSize roiSize);

FwStatus fwiQuantInvTableInit_JPEG_8u16u(const Fw8u *pQuantRawTable, Fw16u *pQuantInvTable);

#endif