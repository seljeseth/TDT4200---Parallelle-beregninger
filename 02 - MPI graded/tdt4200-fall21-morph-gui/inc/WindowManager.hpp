#ifndef _WINDOWMANAGER_HPP_
#define _WINDOWMANAGER_HPP_

#include <GLFW/glfw3.h>
#include <unordered_map>
#include <string>

#include <ManagedWindow.hpp>


class WindowManager
{
public:
    static std::unordered_map<std::string, ManagedWindow *> windows;


    static int openWindows;

    static void init();

    /**
     * Adds a window to the ManagedWindowManager.
     *
     * @param window The ManagedWindow instance to keep track of.
     */
    static void addWindow(ManagedWindow *window);

    /**
     * Removes a ManagedWindow from the ManagedWindowManager.
     *
     * @param window The ManagedWindow instance to remove.
     */
    static void removeWindow(const std::string& key);


    /**
     * Set the primary window.
     */
    static void setPrimary(ManagedWindow *window);

    /**
     * Get the primary window.
     */
    static ManagedWindow *getPrimary();

    /**
     * Checks whether the windows should close.
     */
    static bool shouldClose();

    /**
     * Checks whether there are no more windows to manage.
     */
    static bool empty() { return openWindows > 0; }

    /**
     * Iterates through the windows and checks whether they should be destroyed.
     */
    static void doCleanup();

    /**
     * Iterates through all windows and swaps buffers.
     */
    static void swapBuffers();


    /// DEV METHOD
    // TODO: REMOVE AFTER DEBUGGING
    static void listWindows();
private:
    WindowManager() = delete;

    /**
     * The current primary window.
     */
    static ManagedWindow *primary;


};


#endif
