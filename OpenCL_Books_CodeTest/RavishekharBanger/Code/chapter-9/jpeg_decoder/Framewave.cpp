
/* This file is a work derived from the Open Source library, Framewave
 * http://framewave.sourceforge.net/
 * All the function definitions and declarartions are derived from the files located at 
 * http://sourceforge.net/p/framewave/code/HEAD/tree/trunk/Framewave/domain/fwJPEG/ 
 * Below is the licence information.*/
/*
Copyright (c) 2006-2013 Advanced Micro Devices, Inc. All Rights Reserved.
This software is subject to the Apache v2.0 License.
*/

#include "Framewave.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const Fw8u zigZagFwdOrder[80] =
{
    0,   1,  8, 16,  9,  2,  3, 10, 
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34, 
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 
    53, 60, 61, 54, 47, 55, 62, 63,
    63, 63, 63, 63, 63, 63, 63, 63,
    63, 63, 63, 63, 63, 63, 63, 63
};
const Fw8u zigZagInvOrder[64] =
{
     0,  1,  5,  6, 14, 15, 27, 28,
     2,  4,  7, 13, 16, 26, 29, 42,
     3,  8, 12, 17, 25, 30, 41, 43,
     9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63
};

bool dec_receivebits (FwiDecodeHuffmanState * pDecHuffState, Fw32u accbuf, 
                      int accbitnum, int ssss)
{
    unsigned char  *pCurrSrc = pDecHuffState->pCurrSrc;
    int         srcLenBytes = pDecHuffState->srcLenBytes;
    int c;

    //Figure F.17 procedure for Receive (SSSS)
    if (pDecHuffState->marker == 0) {
        //read to full 32u bytes
        while (accbitnum <= 24) {
            if (srcLenBytes <= 0) break;
            srcLenBytes--;
            c =  *(pCurrSrc++);

            if (c == 0xFF) {
                do {
                    srcLenBytes--;
                    c =  *(pCurrSrc++);
                } while (c == 0xFF);

                if (c == 0) {
                    c = 0xFF; 
                } else {
                    pDecHuffState->marker = c;
                    //prevent data corruption
                        if (ssss > accbitnum) {
                        accbuf <<= 25 - accbitnum;
                        accbitnum = 25;
                    }
                    break;
                }
            }

            accbuf = (accbuf << 8) | c;
            accbitnum += 8;
        }
    } else {
        //prevent data corruption
        if (ssss > accbitnum) {
            accbuf <<= 25 - accbitnum;
            accbitnum = 25;
        }  
    }

    pDecHuffState->pCurrSrc    = pCurrSrc;
    pDecHuffState->srcLenBytes = srcLenBytes;
    pDecHuffState->accbuf      = accbuf;
    pDecHuffState->accbitnum   = accbitnum;

    return true;
}

int dec_huff (FwiDecodeHuffmanState * pDecHuffState, Fw32u accbuf, 
              int accbitnum, const FwiDecodeHuffmanSpec *pTable, int nbits)
{
    Fw16s code;

    if (accbitnum < nbits) { 
        if (! dec_receivebits(pDecHuffState,accbuf,accbitnum,nbits)) 
            return -1; 
        accbitnum = pDecHuffState->accbitnum;
        accbuf	  = pDecHuffState->accbuf;
    }

    accbitnum -= nbits;
    code = (Fw16s)((accbuf >> accbitnum) & ((1<<nbits)-1));

    //Following JPEG standard Figure F.16
    while (code > pTable->maxcode[nbits]) {
        code <<= 1;

        if (accbitnum < 1) { 
            if (! dec_receivebits(pDecHuffState,accbuf,accbitnum,1)) 
                return -1; 
            accbitnum = pDecHuffState->accbitnum;
            accbuf	  = pDecHuffState->accbuf;
        }

        accbitnum--;
        code |= ((accbuf >> accbitnum) & 1);
        nbits++;
    }

    pDecHuffState->accbitnum = accbitnum;

    //To prevent corruption
    if (nbits > 16) return 0;	

    return pTable->pListVals[(code-pTable->mincode[nbits]+pTable->valptr[nbits])];
}


bool FW_HUFF_DECODE(int *result, FwiDecodeHuffmanState *pDecHuffState, 
                     const FwiDecodeHuffmanSpec *pTable)
{	
    int nextbit, look; 

    //Figure F.18 Procedure for fetching the next bit of compressed data
    if (pDecHuffState->accbitnum < 8) { 
        //get more bytes in
        if (! dec_receivebits(pDecHuffState,pDecHuffState->accbuf,pDecHuffState->accbitnum, 0)) {
            return false;
        } 
        if (pDecHuffState->accbitnum < 8) {
            nextbit = 1; 
            *result = dec_huff(pDecHuffState,pDecHuffState->accbuf,pDecHuffState->accbitnum,
                pTable,nextbit);
            if (*result < 0) return false;
            return true;
        } 
    } 

    look = (pDecHuffState->accbuf >> (pDecHuffState->accbitnum - 8)) & 0xff; 
    nextbit = pTable->symcode[look];

    if (nextbit != 0) { 
        pDecHuffState->accbitnum -= nextbit; 
        *result = pTable->symlen[look]; 
    } else { 
        nextbit = 9; 
        *result = dec_huff(pDecHuffState,pDecHuffState->accbuf, pDecHuffState->accbitnum,
            pTable,	nextbit);
        if (*result < 0) return false; 
    }

    return true;
}

//-----------------------------------------------------------------------
//This function returns the buffer size (in bytes) of fwiDecodeHuffmanState 
//structure.
//-----------------------------------------------------------------------
FwStatus fwiDecodeHuffmanStateGetBufSize_JPEG_8u(int* size)
{
    if (size==0) return fwStsNullPtrErr;

    //add alignment requirement
    *size = sizeof(FwiDecodeHuffmanState) + 128;

    return fwStsNoErr;
}

//-----------------------------------------------------------------------
//This function initializes FwiDecodeHuffmanState structure
//-----------------------------------------------------------------------
FwStatus fwiDecodeHuffmanStateInit_JPEG_8u(FwiDecodeHuffmanState *pDecHuffState)
{
    if (pDecHuffState==0) return fwStsNullPtrErr;

    pDecHuffState->accbitnum = 0;
    pDecHuffState->srcLenBytes = 0;
    pDecHuffState->accbuf = 0;
    pDecHuffState->pCurrSrc = (unsigned char *)((unsigned char *)pDecHuffState + 
        sizeof (DecodeHuffmanState)) ;
    pDecHuffState->marker = 0;
    pDecHuffState->EOBRUN=0;

    return fwStsNoErr;
}


//-----------------------------------------------------------------------
//This function allocates memory and initialize FwiDecodeHuffmanState structure
//-----------------------------------------------------------------------
FwStatus fwiDecodeHuffmanStateInitAlloc_JPEG_8u(
    FwiDecodeHuffmanState** pDecHuffState)
{
    if (pDecHuffState==0) return fwStsNullPtrErr;

    int size;

    fwiDecodeHuffmanStateGetBufSize_JPEG_8u(&size);
    *pDecHuffState = (FwiDecodeHuffmanState *) fwMalloc (size);

    return fwiDecodeHuffmanStateInit_JPEG_8u(*pDecHuffState);
}

//-----------------------------------------------------------------------
//This function creates Huffman table for decoder.
//-----------------------------------------------------------------------
FwStatus fwiDecodeHuffmanSpecInit_JPEG_8u(
    const Fw8u *pListBits, const Fw8u *pListVals, FwiDecodeHuffmanSpec *pDecHuffSpec)
{
    if (pListBits==0 || pListVals==0 || pDecHuffSpec==0)
        return fwStsNullPtrErr;

    Fw16u huffsize[257], huffcode[257], code;
    int i, j, k, si;
    Fw16s bits; 

    //Figure C.1 from CCITT Rec. T.81(1992 E) page 51
    //generation of table of Huffman code Sizes
    k=0;
    for (i=1; i<=16; i++) {
        bits = pListBits[i-1];
        //Protection for next for loop
        if (bits+k > 256) return fwStsJPEGHuffTableErr;

        for (j=1; j<=bits; j++) {
            huffsize[k]=(Fw16u)i;
            k++;
        }
    }

    huffsize[k]=0;

    //Figure C.2 from CCITT Rec. T.81(1992 E) page 52
    //generation of table of Huffman codes
    code=0;
    si=huffsize[0];

    //huffsize[k]==0 means the last k to exit the loop
    for (i=0; i<k;) {
        while (huffsize[i]==si) {
            huffcode[i++]=code++;
        }
        code <<=1;
        si++;
    }

    //Figure F.15 from CCITT Rec. T.81(1992 E) page 108
    //ordering procedure for decoding procedure code tables

    //set all codeless symbols to have code length 0
    memset(pDecHuffSpec->symlen, 0, 512);
    memset(pDecHuffSpec->symcode, 0, 512);
    memset(pDecHuffSpec->maxcode, -1, 36);//Fw16s type
    memset(pDecHuffSpec->mincode, 0, 36);//Fw16s type
    memset(pDecHuffSpec->valptr,  0, 36);//Fw16s type

    j=0;
    for (i=1;i<=16;i++) {
        bits = pListBits[i-1];
        if (bits != 0) {
            pDecHuffSpec->valptr[i] =(Fw16s)(j);
            pDecHuffSpec->mincode[i] = huffcode[j];
            j=j+bits-1;
            pDecHuffSpec->maxcode[i] = huffcode[j];
            j++;
        }
        //else maxcode to be -1
        //else pDecHuffSpec->maxcode[i] = -1;
    }

    k=0;
    for (j=1; j<=8; j++) {
        for (i=1;i<= pListBits[j-1];i++) {
            bits = (Fw16s)(huffcode[k] << (8-j));
            for (si=0;si < (1<<(8-j));si++) {
                pDecHuffSpec->symcode[bits]=(Fw16u)j;
                pDecHuffSpec->symlen[bits] = pListVals[k];
                bits++;
            }
            k++;
        }
    }

    memcpy(pDecHuffSpec->pListVals, pListVals, 256);

    //Set 16s max_value to maxcode[17] to prevent data corruption
    pDecHuffSpec->maxcode[17]= 0x7fff;

    return fwStsNoErr;
}

//-----------------------------------------------------------------------
//This function allocates memory and create huffman table for decoder
//-----------------------------------------------------------------------
FwStatus fwiDecodeHuffmanSpecInitAlloc_JPEG_8u(
    const Fw8u *pListBits, const Fw8u *pListVals, FwiDecodeHuffmanSpec** pDecHuffSpec)
{
    //Other parameters will be checked by fwiDecodeHuffmanSpecInit_JPEG_8u
    if (pDecHuffSpec==0) return fwStsNullPtrErr;

    int size;

    size = sizeof(DecodeHuffmanSpec) + 128;
    *pDecHuffSpec = (FwiDecodeHuffmanSpec *) fwMalloc (size);

    return fwiDecodeHuffmanSpecInit_JPEG_8u(pListBits, pListVals, *pDecHuffSpec);
}

//***************************************//
//Saturation function
//***************************************//
unsigned char U8_Sat(float in)
{
    if (in < 0.0)
        return 0;
    if(in > 255.0)
        return 255;
    return (unsigned char)in;
}

//-----------------------------------------------------------------------
//The function fwiYCbCrToBGR_JPEG is declared in the fwImage.h file. It
//operates with ROI (see Regions of Interest in Manual).
//This function converts an YCbCr image to the BGR image according to
//the following formulas:
//  R = Y + 1.402*Cr - 179.456
//  G = Y - 0.34414*Cb - 0.71414*Cr + 135.45984
//  B = Y + 1.772*Cb - 226.816
//-----------------------------------------------------------------------
FwStatus fwiYCbCrToBGR_JPEG_8u_P3C3R(const Fw8u * const pSrcYCbCr[3], int srcStep,
                                     Fw8u *pDstBGR, int dstStep, FwiSize roiSize)
{
    if (pSrcYCbCr == 0 || pDstBGR == 0) return fwStsNullPtrErr;
    if (pSrcYCbCr[0] == 0 || pSrcYCbCr[1] == 0 || pSrcYCbCr[2] == 0 )
        return fwStsNullPtrErr;

    STEPCHECK(srcStep, dstStep);
    ROISIZECHECK(roiSize);

    //Reference code only.
    //SSE2 code need to shift 16 bit
    int x, y;
    int srcPos, dstPos;

    for (y=0;y<roiSize.height; y++) {
        srcPos = y*srcStep;
        dstPos = y*dstStep;
        for (x=0;x<roiSize.width;x++) {
            //add 0.5 for nearest neighbor rounding
            pDstBGR[dstPos++] = U8_Sat(pSrcYCbCr[0][srcPos] + 1.772f*pSrcYCbCr[1][srcPos] - 226.316f);
            pDstBGR[dstPos++] = U8_Sat(pSrcYCbCr[0][srcPos] - 0.34414f*pSrcYCbCr[1][srcPos] - 0.71414f*pSrcYCbCr[2][srcPos]+ 135.95984f);
            pDstBGR[dstPos++] = U8_Sat(pSrcYCbCr[0][srcPos] + 1.402f*pSrcYCbCr[2][srcPos] - 178.956f);
            srcPos++;
        }
    }

    return fwStsNoErr;
}

//-----------------------------------------------------------------------
//This function reorder the zigzag order table to conventional order, and
//is used for fast decoding.
//-----------------------------------------------------------------------
FwStatus fwiQuantInvTableInit_JPEG_8u16u(const Fw8u *pQuantRawTable, Fw16u *pQuantInvTable)
{
    if (pQuantRawTable==0 || pQuantInvTable==0) return fwStsNullPtrErr;

    for(int i=0; i<64; i++)
        pQuantInvTable[i] = pQuantRawTable[zigZagInvOrder[i]];

    return fwStsNoErr;
}

//-----------------------------------------------------------------------
//This function handles the Huffman Baseline decoding for a 8*8 block of the
//quantized DCT coefficients. The decoding procedure follows CCITT Rec. T.81
//section F.2.2
//-----------------------------------------------------------------------
FwStatus fwiDecodeHuffman8x8_JPEG_1u16s_C1(
    const Fw8u *pSrc, int srcLenBytes, int *pSrcCurrPos, Fw16s *pDst,
    Fw16s *pLastDC, int *pMarker, const FwiDecodeHuffmanSpec *pDcTable,
    const FwiDecodeHuffmanSpec *pAcTable, FwiDecodeHuffmanState *pDecHuffState)
{
    if (pSrc==0 || pSrcCurrPos==0 || pDst==0 || pLastDC ==0 ||
        pMarker==0 || pDcTable ==0 || pAcTable==0 ||
        pDecHuffState==0)
        return fwStsNullPtrErr;

    if (srcLenBytes == 0) {
        *pMarker = pDecHuffState->marker;
        return fwStsNoErr;
    }

    if (srcLenBytes <0 || *pSrcCurrPos >= srcLenBytes)
        return fwStsSizeErr;

    int s, k, runlen;

    pDecHuffState->pCurrSrc = (unsigned char *)(pSrc+*pSrcCurrPos);
    pDecHuffState->srcLenBytes = srcLenBytes;
    pDecHuffState->marker   = *pMarker;

    //Follow JPEG standard F.2.2.1 to decode DC coefficient
    if (!FW_HUFF_DECODE(&s, pDecHuffState, pDcTable)) return fwStsJPEGOutOfBufErr;

    if (s) {
        if (pDecHuffState->accbitnum < s) {
            if (! dec_receivebits(pDecHuffState,pDecHuffState->accbuf,pDecHuffState->accbitnum,s))
                return fwStsJPEGOutOfBufErr;
        }
        runlen = GET_ACCBITS(pDecHuffState, s);
        s = DEC_EXTEND(runlen, s);
    }

    //pLastDC
    s += *pLastDC;
    *pLastDC = (Fw16s) s;

    // clean pDst buffer since zero will be skipped
    memset(pDst, 0, 128);//pDst is 16s type

    pDst[0] = (Fw16s) s;

    // Follow JPEG standard F.2.2.2 to decode AC coefficient
    // Figure F.13
    for (k = 1; k < 64; k++) {

        //RS = DECODE
        if (!FW_HUFF_DECODE(&s, pDecHuffState, pAcTable)) return fwStsJPEGOutOfBufErr;

        runlen = s >> 4;
        s &= 0xf;

        if (s) {
            k += runlen;

            //Figure F.14
            //Decode_ZZ(K)
            if (pDecHuffState->accbitnum < s) {
                if (! dec_receivebits(pDecHuffState, pDecHuffState->accbuf,
                    pDecHuffState->accbitnum, s))
                    return fwStsJPEGOutOfBufErr;
            }
            runlen = GET_ACCBITS(pDecHuffState, s);
            s = DEC_EXTEND(runlen, s);

            pDst[zigZagFwdOrder[k]] = (Fw16s) s;
        } else {
            if (runlen != 15)   break;
            k += 15;
        }
    }

    *pSrcCurrPos = (int)(pDecHuffState->pCurrSrc - pSrc);

    *pMarker = pDecHuffState->marker;

    return fwStsNoErr;
}