#ifndef _WINDOW_HPP_
#define _WINDOW_HPP_

#include <GLFW/glfw3.h>
#include <IWindow.hpp>
#include <stdint.h>

class Window : public IWindow
{
public:

    Window(int width, int height, const char* title);
    Window(int width, int height);

    virtual ~Window();

    virtual bool shouldClose() override;

    /// Show the window.
    virtual void show() override;

    /// Selects this window as the current context.
    virtual void select() override;

    virtual void update() override;

    virtual void setTitle(const char* title);

    virtual GLFWwindow *get() override;

    /**
     * Indirect call to `glfwWindowHint`.
     */
     static void hint(int hint, int value);

     bool bufferReady = false;



protected:
    GLFWwindow *window;

    int width;
    int height;
    char *title;

};


#endif
