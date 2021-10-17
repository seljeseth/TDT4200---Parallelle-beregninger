#ifndef _IMAGERENDERER_HPP_
#define _IMAGERENDERER_HPP_

#include <IRenderer.hpp>

#include <ImageRenderable.hpp>


class ImageRenderer : public IRenderer
{
public:

    virtual void render() final;

    void setImage(unsigned int position, Image * image);

    Image *getCurrentImage();

    void setActive(unsigned int position);

    Shader *getShader();

    static ImageRenderer *getInstance();


private:
    ImageRenderer();
    virtual ~ImageRenderer() override;

    ImageRenderable *renderables[2]{nullptr};

    static ImageRenderer *instance;

    int current_image = 0;

    Shader *dedicated_shader;



};


#endif
