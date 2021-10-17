#ifndef _IRENDERER_HPP_
#define _IRENDERER_HPP_

class IRenderer
{
public:
    IRenderer() = default;
    virtual ~IRenderer() {}

    virtual void render() = 0;
};


#endif
