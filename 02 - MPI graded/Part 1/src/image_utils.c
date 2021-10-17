#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <image_utils.h>
#include <stdlib.h>
#include <memory.h>

void image_update_2d_indices(image_t *image)
{
    unsigned int width = image->width;
    unsigned int height = image->height;


    if(image->data != NULL)
        free( image->data );
    image->data = malloc(sizeof(pixel *) * height);

#ifdef DEBUG
    if( 0 == image->data )
    {
        fprintf(stderr, "Failed to allocate space for 2D indices\n");
    }
#endif


    for ( int i = 0; i < height; i++ )
    {
        image->data[i] = &image->rawdata[ i * width ];
    }

}

image_t *newImage(unsigned int const width,
        unsigned int const height)
{
    image_t *result = malloc(sizeof(image_t));

    result->width = width;
    result->height = height;

    result->data = NULL;
    result->rawdata = malloc(sizeof(pixel) * width * height);
    memset( result->rawdata, 0, sizeof(pixel)*width*height);

    image_update_2d_indices(result);

    return result;
}

//! Requires image to be a heap allocated variable.
//! @param image The heap allocated image_t to free.
void freeImage(image_t *image)
{
    if ( NULL == image )
        return;

    if( NULL != image->data )
        free(image->data);

    if ( NULL != image->rawdata )
        free(image->rawdata);

    free(image);
}

image_t *loadImage(char const *filename)
{
    stbi_set_flip_vertically_on_load(1);
    stbi_flip_vertically_on_write(1);

    int width;
    int height;
    int components;
    int num_components_to_request = STBI_rgb_alpha; // 4 components (RGBA)

    unsigned char *imageData =
        stbi_load(filename, &width, &height, &components, num_components_to_request);

    if ( imageData == NULL )
    {
        return NULL;
    }

    image_t *result = malloc(sizeof(image_t));

    result->width = width;
    result->height = height;

    result->rawdata = (pixel *) imageData;
    result->data = NULL;
    image_update_2d_indices(result);

    return result;
}


// TODO: Actually implement
int saveImage(image_t *image, char const *filename)
{
    return stbi_write_bmp(filename,
            image->width,
            image->height,
            STBI_rgb_alpha,
            image->rawdata);

}


//! Helper function the swap two images
void swapImage(image_t **one, image_t **two)
{
  image_t *helper = *two;
  *two = *one;
  *one = helper;
}

