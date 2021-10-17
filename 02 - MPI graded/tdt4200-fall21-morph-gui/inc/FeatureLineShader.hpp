#ifndef _FEATURELINESHADER_HPP_
#define _FEATURELINESHADER_HPP_

#include <Shader.hpp>

class FeatureLineShader
{
public:
    static FeatureLineShader *getInstance();

    int get();

private:
    FeatureLineShader();

    static FeatureLineShader *SHADER_INSTANCE;

    Shader *FL_SHADER;

};

#endif
