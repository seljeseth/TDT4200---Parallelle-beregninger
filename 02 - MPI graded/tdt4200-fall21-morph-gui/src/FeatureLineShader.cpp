#include <FeatureLineShader.hpp>

FeatureLineShader::FeatureLineShader()
{

    FL_SHADER = new Shader{"res/shaders/FeatureLine.vert", "res/shaders/FeatureLine.frag"};

    FL_SHADER->link();
}

FeatureLineShader *FeatureLineShader::SHADER_INSTANCE = nullptr;

FeatureLineShader *FeatureLineShader::getInstance()
{
    if(nullptr == SHADER_INSTANCE)
        SHADER_INSTANCE = new FeatureLineShader{};

    return SHADER_INSTANCE;
}


int FeatureLineShader::get()
{
    return FL_SHADER->get();
}
