#ifndef _ISHADERRENDERABLE_HPP_
#define _ISHADERRENDERABLE_HPP_

#include <Shader.hpp>

class IShaderRenderable
{
public:

    virtual ~IShaderRenderable() {};

    virtual void setShader(Shader *shader) = 0;

};

#endif
