#ifndef _SIMPLEPOINTSHADER_HPP_
#define _SIMPLEPOINTSHADER_HPP_

#include <Shader.hpp>

class SimplePointShader
{
public:
    static SimplePointShader *getInstance();

    int get();

private:
    SimplePointShader();

    static SimplePointShader *SHADER_INSTANCE;

    Shader *FL_SHADER;

};


#endif
