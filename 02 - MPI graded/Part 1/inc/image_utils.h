#ifndef _IMAGE_UTILS_H_
#define _IMAGE_UTILS_H_

#include <stb/stb_image.h>
#include <stb/stb_image_write.h>


typedef struct pixel_struct {
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a;
} pixel;


typedef struct image_struct {
    unsigned int width;
    unsigned int height;
    pixel *rawdata;
    pixel **data;
} image_t;


image_t *newImage(unsigned int const width, unsigned int const height);

void freeImage(image_t *image);

image_t *loadImage(char const *filename);
int saveImage(image_t *image, char const *filename);

// Helper function to swap bmpImageChannel pointers
void swapImage(image_t **one, image_t **two);

#endif
