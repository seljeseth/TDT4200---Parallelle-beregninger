#include <WindowManager.hpp>
#include <GLFW/glfw3.h>
#include <iostream>

/// The Key-Value pair type used by the hashmap.
using WM_KV_PAIR = std::pair<std::string, ManagedWindow *>;

void WindowManager::init()
{
}

void WindowManager::addWindow(ManagedWindow *window)
{

    std::string key{window->title};

    windows.emplace(key, window);
    openWindows++;
}

void WindowManager::removeWindow(const std::string& key)
{
    windows.erase(key);

    openWindows--;
}


int WindowManager::openWindows = 0;
ManagedWindow *WindowManager::primary = nullptr;
std::unordered_map<std::string, ManagedWindow *> WindowManager::windows{};


void WindowManager::setPrimary(ManagedWindow *window)
{
    primary = window;

    glfwMakeContextCurrent(window->window);

}

ManagedWindow *WindowManager::getPrimary()
{
    return primary;
}

bool WindowManager::shouldClose()
{
    for (WM_KV_PAIR cur_window : windows )
    {

        if(cur_window.second->window != nullptr && glfwWindowShouldClose(cur_window.second->window)) return true;
    }

    return false;
}

void WindowManager::doCleanup()
{

    ManagedWindow * currentWindow = nullptr;

    std::unordered_map<std::string, ManagedWindow *>::iterator it = windows.begin();


    while( it != windows.end() )
    {
        currentWindow = (*it).second;

        it = windows.erase(it);
        delete currentWindow;
    }
}

void WindowManager::swapBuffers()
{
    for(WM_KV_PAIR cur_window : windows)
    {
        cur_window.second->select();
        glfwSwapBuffers(cur_window.second->window);
    }
}

void WindowManager::listWindows()
{
    for (WM_KV_PAIR window : windows)
    {
        std::cout << "Key: " << window.first << std::endl;
        std::cout << "Active Window: " << window.second->title << std::endl;
    }
}
