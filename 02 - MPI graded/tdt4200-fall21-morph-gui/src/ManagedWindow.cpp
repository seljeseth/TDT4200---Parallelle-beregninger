#include <ManagedWindow.hpp>
#include <WindowManager.hpp>
#include <iostream>

/// Removes the window from the window manager.
/// The base class destructor will ensure destruction of the GLFWwindow instance.
ManagedWindow::~ManagedWindow()
{
    WindowManager::removeWindow(std::string(this->title));

}

ManagedWindow::ManagedWindow(int width, int height, const char* title) : Window{width, height, title}
{
    _registerWindow();
}
ManagedWindow::ManagedWindow(int width, int height) : Window{width, height}
{
    _registerWindow();
}


void ManagedWindow::_registerWindow()
{
    WindowManager::addWindow(this);
}

