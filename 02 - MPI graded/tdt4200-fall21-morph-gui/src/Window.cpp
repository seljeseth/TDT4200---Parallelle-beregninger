#include <Window.hpp>
#include <cstdio>
#include <cstring>
#include <iostream>

Window::Window(int width, int height, const char* title)
{
    window = glfwCreateWindow(width, height, title, nullptr, nullptr);

    setTitle(title);

}


Window::Window(int width, int height)
{
    window = glfwCreateWindow(width, height, "Application", nullptr, nullptr);

    setTitle("Application");
}

Window::~Window()
{
    glfwDestroyWindow(window);

    delete title;
}

void Window::setTitle(const char* title)
{
    size_t length = strlen(title);

    this->title = new char[length + 1]{'\0'};

    strcpy(this->title, title);
}


bool Window::shouldClose()
{
    return glfwWindowShouldClose(window);
}


void Window::show()
{
    glfwShowWindow(window);
    select();
}

void Window::select()
{
    glfwMakeContextCurrent(window);
}

void Window::hint(int hint, int value)
{
    glfwWindowHint(hint, value);
}

void Window::update()
{
    glfwSwapBuffers(window);
}

GLFWwindow *Window::get()
{
    return window;
}
