#ifndef SCAN_CPP
#define SCAN_CPP

//#include "..\include\scan.h"
//#include "fwImage.h"
#include "Framewave.h"
#include "stdio.h"
#include "stdlib.h"
typedef struct {
	/* BMP File Header */
	unsigned	char	bfType[2] ;		/* The characters "BM" */
	unsigned	int		bfSize ;		/* The size of the file in bytes */
	unsigned	short	bfReserved1 ;	/* Unused - must be zero */
	unsigned	short	bfReserved2 ;	/* Unused - must be zero */
	unsigned	int		bfOffBits ;		/* Offset to start of Pixel Data */

	/* BMP Image Header */

	unsigned	int		biSize	;		/* Header Size - Must be at least 40 */
	unsigned	int		biWidth	;		/* Image width in pixels */
	unsigned	int		biHeight ;		/* Image height in pixels */
	unsigned	short	biPlanes ;		/* Must be 1 */
	unsigned	short	biBitCount ;	/* Bits per pixel - 1, 2, 4, 8, 16, 24, or 32 */
	unsigned	int		biCompression;	/* Compression type (0 = uncompressed) */
	unsigned	int		biSizeImage;	/* Image Size - may be zero for uncompressed images */
	unsigned	int		biXPelsPerMeter;/* Preferred resolution in pixels per meter */
	unsigned	int		biYPelsPerMeter;/* Preferred resolution in pixels per meter */
	unsigned	int		biClrUsed;		/* Number Color Map entries that are actually used */
	unsigned	int		biClrImportant;	/* 	Number of significant colors */
}bmpInfo_t;

#include <memory.h>
int writeBmpFile(const char *fName, unsigned char *pImageBuffer, unsigned long int imageSize, unsigned int width, unsigned int height)
{
	bmpInfo_t bmpInfo;
	
	bmpInfo.bfType[0] = 'B' ;
	bmpInfo.bfType[1] = 'M' ;

	bmpInfo.bfSize = imageSize+54 ;

	bmpInfo.bfReserved1 = 0 ;
	bmpInfo.bfReserved2 = 0 ;
	bmpInfo.bfOffBits = 54 ;

	/* Image Header */
	bmpInfo.biSize = 40 ;
	bmpInfo.biWidth = width;
	bmpInfo.biHeight= height;
	bmpInfo.biPlanes = 1 ;
	bmpInfo.biBitCount = 24 ;
	bmpInfo.biCompression = 0 ;
	bmpInfo.biSizeImage = imageSize ;

	bmpInfo.biXPelsPerMeter = 0 ;
	bmpInfo.biYPelsPerMeter = 0 ;
	bmpInfo.biClrUsed = 0 ;		
	bmpInfo.biClrImportant = 0 ;	


	FILE* fp;
#ifdef WIN32
    fopen_s(&fp, fName, "wb");
#else
    fp = fopen(fName, "wb");
#endif

	// write the bitmap file
	
	/* BMP File Header */

	fwrite(&bmpInfo.bfType[0],		1, 2, fp ) ; /* The characters "BM" */
	fwrite(&bmpInfo.bfSize,			1, 4, fp ) ; /* The size of the file in bytes */
	fwrite(&bmpInfo.bfReserved1,		1, 2, fp ) ; /* Unused - must be zero */
	fwrite(&bmpInfo.bfReserved2,		1, 2, fp ) ; /* Unused - must be zero */
	fwrite(&bmpInfo.bfOffBits,		1, 4, fp ) ; /* Offset to start of Pixel Data */
	
	/* BMP Image Header */
	
	fwrite(&bmpInfo.biSize,			1, 4, fp ) ; /* Header Size - Must be at least 40 */
	fwrite(&bmpInfo.biWidth,			1, 4, fp ) ; /* Image width in pixels */
	fwrite(&bmpInfo.biHeight,			1, 4, fp ) ; /* Image height in pixels */
	fwrite(&bmpInfo.biPlanes,			1, 2, fp ) ; /* Must be 1 */
	fwrite(&bmpInfo.biBitCount,		1, 2, fp ) ; /* Bits per pixel - 1, 2, 4, 8, 16, 24, or 32 */
	fwrite(&bmpInfo.biCompression,	1, 4, fp ) ; /* Compression type (0 = uncompressed) */
	fwrite(&bmpInfo.biSizeImage,		1, 4, fp ) ; /* Image Size - may be zero for uncompressed images */
	fwrite(&bmpInfo.biXPelsPerMeter,	1, 4, fp ) ; /* Preferred resolution in pixels per meter */
	fwrite(&bmpInfo.biYPelsPerMeter,	1, 4, fp ) ; /* Preferred resolution in pixels per meter */
	fwrite(&bmpInfo.biClrUsed,		1, 4, fp ) ; /* Number Color Map entries that are actually used */
	fwrite(&bmpInfo.biClrImportant,	1, 4, fp ) ; /*	Number of significant colors */

	//flip the image before writing to file
	Fw8u *temp = (Fw8u*)malloc(imageSize);
	memcpy(temp,pImageBuffer,imageSize);
	FwiSize roiSize;
	roiSize.width = width;
	roiSize.height = height;
#if 0
	fwiMirror_8u_C3IR(temp, width*3, roiSize, fwAxsHorizontal);//flip along horizontal axis
#endif
	fwrite(temp,1,imageSize,fp);
	fclose(fp);
	free(temp);

	return 0;
}

#endif
