#include <FeatureWindow.hpp>


FeatureWindow:: FeatureWindow(int width, int height, const char* title)
    : ManagedWindow(width, height, title)
{

}

FeatureWindow:: FeatureWindow(int width, int height)
    : ManagedWindow(width, height)
{

}

FeatureWindow::~FeatureWindow()
{
}
