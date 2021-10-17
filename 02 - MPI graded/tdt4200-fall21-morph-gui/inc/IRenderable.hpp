#ifndef _IRENDERABLE_HPP_
#define _IRENDERABLE_HPP_

class IRenderable
{
public:
    IRenderable() {}

    virtual ~IRenderable() {}

    virtual void render() = 0;

protected:
    unsigned int VAO;
    unsigned int VBO;
    unsigned int EBO;

    unsigned int *indices;
    float *vertices;

    unsigned int numIndices;

};

#endif
