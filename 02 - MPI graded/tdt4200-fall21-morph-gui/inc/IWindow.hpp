#ifndef _IWINDOW_HPP_
#define _IWINDOW_HPP_

struct GLFWwindow;

class IWindow
{
public:

    IWindow() {};

    virtual ~IWindow() = default;

    virtual bool shouldClose() = 0;

    virtual void show() = 0;

    virtual void select() = 0;

    virtual void update() = 0;

    virtual GLFWwindow *get() = 0;

};

#endif
