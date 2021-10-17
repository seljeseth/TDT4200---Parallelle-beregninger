#ifndef _MANAGEDWINDOW_HPP_
#define _MANAGEDWINDOW_HPP_

#include <Window.hpp>

class ManagedWindow : public Window
{

public:

    ManagedWindow(int width, int height, const char* title);
    ManagedWindow(int width, int height);

    virtual ~ManagedWindow();

    friend class WindowManager;

private:
    void _registerWindow();


};


#endif
