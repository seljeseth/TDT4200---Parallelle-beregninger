/******************************************************************************************
The "morph" program is to do the morph
*******************************************************************************************/

#ifndef MORPH_H
#define MORPH_H

#include <math.h>

//The default size of the window when there is no picture
const int WIDTH = 600;
const int HEIGHT = 600;

//The width and height of the picture
int imgWidthOrig = 0;
int imgHeightOrig = 0;
int imgWidthDest = 0;
int imgHeightDest = 0;

//the pixel
typedef struct pix{
  unsigned char r,g,b,a;
} pixel;

typedef struct SimplePoint_struct {
        double x, y;
} SimplePoint;


typedef struct SimpleFeatureLine_struct {
      SimplePoint startPoint;
      SimplePoint endPoint;
} SimpleFeatureLine;

//The pixmap is the pointer array which contains the pointers that point at the pixels
pixel * hSrcImgMap;
pixel * hDstImgMap;
pixel * hMorphMap;

//The name of input and output files
const char *inputFileOrig;
const char *inputFileDest;
const char *outputFile;

const char *tempFile;
const char *linePath;
//the parameter of the weight
const char* pStr, *aStr, *bStr, *tStr;
float p = 0;
float a = 1;
float b = 2;
float t = 0.5;

// const SimpleFeatureLine* hSrcLines;
// const SimpleFeatureLine* hDstLines;

/********************************************************************************
This function would allocate the memory space for the pixmap
********************************************************************************/
void allocPixmap(int w, int h, pixel ** map);

/********************************************************************************
This function would read from a image file of various types and store the RGBA
info into the "pixmap".
********************************************************************************/
//notice that if you donot use the "unsigned char * &map" instead of "unsigned char * map"
//what you did in the function is only to initialize the local pointer to point at the pixmap

pixel ** setup2DPixelMap(unsigned char *pixelMap, int w, int h, int channels);

/********************************************************************************
This function would write the image from "pixmap" into a image file.The type
also can be various.
********************************************************************************/

/********************************************************************************
The core funtions to do the morph
********************************************************************************/
void doMorph();

#endif
