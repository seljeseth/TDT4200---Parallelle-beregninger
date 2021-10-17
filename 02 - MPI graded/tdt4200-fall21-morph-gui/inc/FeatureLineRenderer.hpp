#ifndef _FEATURELINERENDERER_HPP_
#define _FEATURELINERENDERER_HPP_

#include <IRenderer.hpp>
#include <FeatureLine.hpp>

class FeatureLineRenderer : public IRenderer
{
public:
    FeatureLineRenderer() = default;
    virtual ~FeatureLineRenderer();

    virtual void render() final;

};


#endif
