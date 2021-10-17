#include <SimplePointShader.hpp>

SimplePointShader::SimplePointShader()
{

    FL_SHADER = new Shader{"res/shaders/SimplePoint.vert", "res/shaders/SimplePoint.frag"};

    FL_SHADER->link();
}

SimplePointShader *SimplePointShader::SHADER_INSTANCE = nullptr;

SimplePointShader *SimplePointShader::getInstance()
{
    if(nullptr == SHADER_INSTANCE)
        SHADER_INSTANCE = new SimplePointShader{};

    return SHADER_INSTANCE;
}


int SimplePointShader::get()
{
    return FL_SHADER->get();
}
