#ifndef _IMAGE_HPP_
#define _IMAGE_HPP_


class Image {

public:
    Image(const char *imagePath);
    Image(const char *imagePath, int numChannels);

    static void flipOnLoad();
    static void flipOnWrite();

    unsigned char *getData();
    int getWidth() const;
    int getHeight() const;
    int getNumChannels() const;


private:
    unsigned char *loadImage(const char *imagePath, int num_channels);

    unsigned char *imageData;
    int width;
    int height;
    int numChannels;

};


#endif
