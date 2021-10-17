#include <Image.hpp>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>


Image::Image(const char *imagePath)
{
    imageData = loadImage(imagePath, 3);

    this->numChannels = 3;
}

Image::Image(const char *imagePath, int numChannels)
{
    imageData = loadImage(imagePath, numChannels);

    this->numChannels = numChannels;
}

unsigned char *Image::loadImage(const char *imagePath, int numChannels)
{

    int num_channels_in_file;
    unsigned char *data = stbi_load(imagePath, &width, &height, &num_channels_in_file, numChannels);

    if ( ! data )
    {
        std::cerr << "Could not load image: " << imagePath << std::endl;
        abort();
    }

    return data;

}

void Image::flipOnLoad()
{
    stbi_set_flip_vertically_on_load(true);
}

unsigned char *Image::getData()
{
    return imageData;
}

int Image::getWidth() const
{
    return width;
}

int Image::getHeight() const
{
    return height;
}

int Image::getNumChannels() const
{
    return numChannels;
}
