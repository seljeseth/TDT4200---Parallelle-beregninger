#ifndef _INPUTHANDLER_HPP_
#define _INPUTHANDLER_HPP_
#include <inttypes.h>
#include <GLFW/glfw3.h>
#include <ctime>


/// Nanoseconds
#define IH_DBL_CLICK_TIME 5e5


enum IH_INPUT_STATE {
    IH_RELEASE = GLFW_RELEASE,
    IH_PRESS = GLFW_PRESS,
    IH_REPEAT = GLFW_REPEAT,
    IH_DBL_PRESS = GLFW_REPEAT + 1
};

/**
 * TODO: Add double click handler.
 */

struct IH_MousePosition
{
    double x;
    double y;
};

class InputHandler
{

public:

    InputHandler() = delete;

    /// See e.g. GLFW_MOUSE_BUTTON_1 in glfw3.h
    static bool isClicked(int mousebutton);

    static bool isDoubleClicked(int mousebutton);

    /// See e.g. GLFW_MOUSE_BUTTON_1 in glfw3.h
    static bool isHeld(int mousebutton);

    static bool isPressed(int key);

    static bool isRepeated(int key);

    static bool keyReleased(int key);

    static bool mouseButtonReleased(int key);

    static IH_MousePosition getMousePos();

    static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos);

    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);

    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

    static void setupCallbacks();

    static void update();



private:
    static uint8_t kb_debounce_masks[512];
    static uint8_t mouse_debounce_masks[8];

    static int kb_key_states[512];
    static int prev_kb_key_states[512];
    static int mouse_states[8];
    static int prev_mouse_states[8];

    // Nanoseconds
    static uint64_t mouse_timestamps[8];
    static uint64_t prev_mouse_timestamps[8];

    static double xpos;
    static double ypos;

    static uint64_t timeBetweenClicks(int mousebutton);
    static uint64_t timeSinceClick(int mousebutton);

};


#endif
