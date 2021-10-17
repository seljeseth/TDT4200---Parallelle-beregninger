#include <ImageRenderer.hpp>
#include <iostream>


ImageRenderer::ImageRenderer()
{
    dedicated_shader = new Shader{"res/shaders/Image.vert", "res/shaders/Image.frag"};

}

ImageRenderer::~ImageRenderer()
{
    delete renderables[0];
    delete renderables[1];
}

void ImageRenderer::setImage(unsigned int position, Image *image)
{
    if(position > 1)
    {
        std::cerr << "Cannot set image in positions > 1!" << std::endl;
        abort();
    }

    ImageRenderable *newRenderable = new ImageRenderable{};
    newRenderable->setImage(image);
    newRenderable->setShader(dedicated_shader);

    renderables[position] = newRenderable;
}


ImageRenderer *ImageRenderer::instance = nullptr;

ImageRenderer *ImageRenderer::getInstance()
{
    if(nullptr == instance)
    {
        instance = new ImageRenderer();
    }

    return instance;
}

void ImageRenderer::setActive(unsigned int position)
{

    if(position > 1) return;

    current_image = position;

}


void ImageRenderer::render()
{
    renderables[current_image]->render();
}


Shader *ImageRenderer::getShader()
{
    return dedicated_shader;
}


Image *ImageRenderer::getCurrentImage()
{
    return renderables[current_image]->getImage();
}
