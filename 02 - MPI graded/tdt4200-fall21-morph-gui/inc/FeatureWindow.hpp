#ifndef _FEATUREWINDOW_HPP_
#define _FEATUREWINDOW_HPP_

#include <ManagedWindow.hpp>

class FeatureWindow : public ManagedWindow
{

    FeatureWindow() = delete;

    FeatureWindow(int width, int height, const char* title);
    FeatureWindow(int width, int height);

    ~FeatureWindow();


    static void handleResize(FeatureWindow *window);

};

#endif
