#ifndef _IMAGERENDERABLE_HPP_
#define _IMAGERENDERABLE_HPP_

#include <IRenderable.hpp>
#include <IShaderRenderable.hpp>
#include <Image.hpp>
#include <Shader.hpp>

class ImageRenderable : public IRenderable, IShaderRenderable
{

public:
    ImageRenderable();

    virtual ~ImageRenderable();

    void render() override;

    void setImage(Image *image);

    Image *getImage();

    virtual void setShader(Shader *shader) final;


private:
    void _generateBuffers();
    unsigned int UV_Buffer;

    Image *image;

    unsigned int textureID;

    Shader *imageShader;


};



#endif
