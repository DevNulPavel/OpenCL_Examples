#include <bmp_image.h>
#include <stdio.h>
/* For more information on the BMP header, go to 
   http://en.wikipedia.org/wiki/BMP_file_format 
   */
void
ReadBMPImage(std::string filename,  Image **image)
{
    BMPHeader header;
    BMPInfoHeader infoHeader;
    ColorPalette *colors_;
    PixelColor   *pixelColor;
    unsigned int numColors_;
    // Open BMP file
    FILE *fd;
    *image = (Image *)calloc(1,sizeof(Image));
    (*image)->filename = filename.c_str();

    fopen_s(&fd, (*image)->filename, "rb");
    (*image)->width = 0;
    (*image)->height = 0;
    (*image)->pixels = NULL;


    // Opened OK
    if (fd != NULL) {
        // Read the BMP header
        fread(&header, sizeof(BMPHeader), 1, fd);
        if (ferror(fd)) {
            fclose(fd);
            goto fileReadFail;
        }

        // Confirm that we have a bitmap file
        if (header.id != bitMapID) {
            fclose(fd);
            goto fileReadFail;
        }
        
        // Read map info header
        fread(&infoHeader, sizeof(BMPInfoHeader), 1, fd);

        // Failed to read map info header
        if (ferror(fd)) {
            fclose(fd);
            return;
        }

        // Store number of colors
        numColors_ = 1 << infoHeader.bitsPerPixel;

        //load the palate for 8 bits per pixel
        if(infoHeader.bitsPerPixel == 8) {
            colors_ = (ColorPalette*)malloc(numColors_ * sizeof(ColorPalette));
            fread( (char *)colors_, numColors_ * sizeof(ColorPalette), 1, fd);
        }

        // Allocate buffer to hold all pixels
        unsigned int sizeBuffer = header.size - header.offset;
        unsigned char *tmpPixels = (unsigned char*)malloc(sizeBuffer*sizeof(unsigned char));

        // Read pixels from file, including any padding
        fread(tmpPixels, sizeBuffer * sizeof(unsigned char), 1, fd);

        // Allocate image
        pixelColor = (PixelColor*)malloc(infoHeader.width * infoHeader.height*sizeof(PixelColor));

        // Set image, including w component (white)
        memset(pixelColor, 0xff, infoHeader.width * infoHeader.height * sizeof(PixelColor));

        unsigned int index = 0;
        for(unsigned int y = 0; y < infoHeader.height; y++) {
            for(unsigned int x = 0; x < infoHeader.width; x++) {
                // Read RGB values
                if (infoHeader.bitsPerPixel == 8) {
                    pixelColor[(y * infoHeader.width + x)] = colors_[tmpPixels[index++]];
                }
                else { // 24 bit
                    //pixelColor[(y * infoHeader.width + x)].w = 0;
                    pixelColor[(y * infoHeader.width + x)].z = tmpPixels[index++];
                    pixelColor[(y * infoHeader.width + x)].y = tmpPixels[index++];
                    pixelColor[(y * infoHeader.width + x)].x = tmpPixels[index++];
                }
            }

            // Handle padding
            for(unsigned int x = 0; x < (4 - (3 * infoHeader.width) % 4) % 4; x++) {
                index++;
            }
        }

        // Loaded file so we can close the file.
        fclose(fd);
        free(tmpPixels);
        if(infoHeader.bitsPerPixel == 8) {
            free(colors_);
        }
        (*image)->width = infoHeader.width;
        (*image)->height = infoHeader.height;
        (*image)->pixels = (void *)pixelColor;
        return;
    }
fileReadFail:
    free (*image);
    *image = NULL;
    return;
}

//The pixel array is a block of 32-bit DWORDs, that describes the image pixel by pixel. 
//Normally pixels are stored "upside-down" with respect to normal image raster scan order, 
//starting in the lower left corner, going from left to right, and then row by row from 
//the bottom to the top of the image. [wikipedia]

void ReadBMPGrayscaleImageUchar(std::string filename,  Image **image)
{
    BMPHeader header;
    BMPInfoHeader infoHeader;
    unsigned char *pixelColor;
    unsigned int numColors_;
    // Open BMP file
    FILE *fd;
    *image = (Image *)calloc(1,sizeof(Image));
    (*image)->filename = filename.c_str();

    fopen_s(&fd, (*image)->filename, "rb");
    (*image)->width = 0;
    (*image)->height = 0;
    (*image)->pixels = NULL;

    // Opened OK
    if (fd != NULL) {
        // Read the BMP header
        fread(&header, sizeof(BMPHeader), 1, fd);
        if (ferror(fd)) {
            fclose(fd);
            goto ReadFailed;
        }

        // Confirm that we have a bitmap file
        if (header.id != bitMapID) {
            fclose(fd);
            goto ReadFailed;
        }

        // Read map info header
        fread(&infoHeader, sizeof(BMPInfoHeader), 1, fd);

        // Failed to read map info header
        if (ferror(fd)) {
            fclose(fd);
            return;
        }

        // Store number of colors
        numColors_ = 1 << infoHeader.bitsPerPixel;
        //Till header.offset its the pallette information.
        fseek(fd, header.offset, SEEK_SET);
        // Allocate buffer to hold all pixels
        unsigned int sizeBuffer = header.size - header.offset;
        // Allocate image
        pixelColor = (unsigned char*)malloc(sizeBuffer*sizeof(unsigned char) );
        // Set image, including w component (white)
        memset(pixelColor, 0xff, infoHeader.width * infoHeader.height * sizeof(unsigned char));
        // Read pixels from file, including any padding
        fread(pixelColor, sizeBuffer * sizeof(unsigned char), 1, fd);
        // Loaded file so we can close the file.
        fclose(fd);
        (*image)->width = infoHeader.width;
        (*image)->height = infoHeader.height;
        (*image)->pixels = (void *)pixelColor;
        return;
    }
ReadFailed:
    free (*image);
    *image = NULL;
    return;
}

//Reads the BMP image and converts it to floating point array
void ReadBMPGrayscaleImageFloat(std::string filename,  Image **image)
{
    BMPHeader header;
    BMPInfoHeader infoHeader;
    unsigned char *pixelColor;
    float         *pixelColorFloat;
    unsigned int numColors_;
    // Open BMP file
    FILE *fd;
    *image = (Image *)calloc(1, sizeof(Image) );
    (*image)->filename = filename.c_str();

    fopen_s(&fd, (*image)->filename, "rb");
    (*image)->width = 0;
    (*image)->height = 0;
    (*image)->pixels = NULL;


    // Opened OK
    if (fd != NULL) {
        // Read the BMP header
        fread(&header, sizeof(BMPHeader), 1, fd);
        if (ferror(fd)) {
            fclose(fd);
            goto ReadFailed;
        }

        // Confirm that we have a bitmap file
        if (header.id != bitMapID) {
            fclose(fd);
            goto ReadFailed;
        }
        (*image)->offsetSize = header.offset;
        (*image)->storeOffset = (void *)malloc((*image)->offsetSize);
        
        // Read map info header
        fread(&infoHeader, sizeof(BMPInfoHeader), 1, fd);
        if (ferror(fd)) {
            fclose(fd);
            free( (*image)->storeOffset );
            goto ReadFailed;
        }

        // Failed to read map info header
        if (ferror(fd)) {
            fclose(fd);
            return;
        }
        fseek(fd, 0, SEEK_SET);
        fread((*image)->storeOffset, (*image)->offsetSize, 1, fd);
        (*image)->header = header;
        (*image)->infoHeader = infoHeader;

        // Store number of colors
        numColors_ = 1 << infoHeader.bitsPerPixel;

        //Till header.offset its the pallette information.
        fseek(fd, header.offset, SEEK_SET);
        // Allocate buffer to hold all pixels
        unsigned int sizeBuffer = header.size - header.offset;
        // Allocate image
        pixelColor = (unsigned char*)malloc(sizeBuffer*sizeof(unsigned char) );
        pixelColorFloat = (float*)malloc(sizeBuffer*sizeof(float) );
        // Set image, including w component (white)
        memset(pixelColor, 0xff, infoHeader.width * infoHeader.height * sizeof(unsigned char));
        // Read pixels from file, including any padding
        fread(pixelColor, sizeBuffer * sizeof(unsigned char), 1, fd);
        for (unsigned int i =0;i<sizeBuffer; i++ )
            pixelColorFloat[i] = (float)pixelColor[i];
        // Loaded file so we can close the file.
        fclose(fd);
        free(pixelColor);
        (*image)->width = infoHeader.width;
        (*image)->height = infoHeader.height;
        (*image)->pixels = (void *)pixelColorFloat;
        return;
    }
ReadFailed:
    free (*image);
    *image = NULL;
    return;
}

void WriteBMPGrayscaleImageFloat(std::string filename,  Image **image, float*imgBuffer)
{
    //BMPHeader header;
    //BMPInfoHeader infoHeader;
    unsigned char *pixelColor;
    //unsigned int numColors_;
    int width, height;
    // Open BMP file
    FILE *fd;
    //unsigned char *pixelBuffer;
    width = (*image)->width;
    height = (*image)->height;
    fopen_s(&fd, filename.c_str(), "wb");
    fwrite((*image)->storeOffset, (*image)->header.offset, 1, fd);
    pixelColor = (unsigned char*)malloc( width * height * sizeof(unsigned char) );

    for(int i=0; i < (width * height); i++)
    {
        pixelColor[i] = (unsigned char)imgBuffer[i];
    }
    fwrite(pixelColor, (*image)->width * (*image)->height, 1, fd);
    free (pixelColor);
    fclose(fd);
   return;
}

void ReleaseBMPImage(Image **image)
{
    if(*image != NULL)
        if((*image)->pixels !=NULL)
        {
            free((*image)->pixels);
            free(*image);
        }
    return;
}