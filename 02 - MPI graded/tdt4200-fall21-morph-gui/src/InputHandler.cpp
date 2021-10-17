#include <InputHandler.hpp>
#include <WindowManager.hpp>



uint8_t InputHandler::kb_debounce_masks[512];
uint8_t InputHandler::mouse_debounce_masks[8];

int InputHandler::kb_key_states[512]{0};
int InputHandler::prev_kb_key_states[512]{0};
int InputHandler::mouse_states[8]{0};
int InputHandler::prev_mouse_states[8]{0};


uint64_t InputHandler::mouse_timestamps[8]{0};
uint64_t InputHandler::prev_mouse_timestamps[8]{0};

double InputHandler::xpos;
double InputHandler::ypos;

void InputHandler::cursorPosCallback(GLFWwindow* window, double x, double y)
{
    xpos = x;
    ypos = y;
}

void InputHandler::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{

    prev_mouse_states[button] = mouse_states[button];
    mouse_states[button] = action;

    timespec now;

    clock_gettime(CLOCK_MONOTONIC, &now);

    if(action == GLFW_PRESS)
    {
        prev_mouse_timestamps[button] = mouse_timestamps[button];
        mouse_timestamps[button] = now.tv_nsec / 1000 + now.tv_sec * 1e6;
    }
}

void InputHandler::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{

    kb_key_states[key] = action;

}

void InputHandler::setupCallbacks()
{

    glfwSetCursorPosCallback(WindowManager::getPrimary()->get(), InputHandler::cursorPosCallback);
    glfwSetKeyCallback(WindowManager::getPrimary()->get(), InputHandler::keyCallback);
    glfwSetMouseButtonCallback(WindowManager::getPrimary()->get(), InputHandler::mouseButtonCallback);
}

bool InputHandler::isClicked(int mousebutton)
{

    if(mouse_states[mousebutton] == IH_PRESS && prev_mouse_states[mousebutton] == IH_RELEASE) {
        return true;
    }

    return false;
}

bool InputHandler::isDoubleClicked(int mousebutton)
{
    if( isClicked(mousebutton) && timeBetweenClicks(mousebutton) < IH_DBL_CLICK_TIME )
        return true;

    return false;
}


/**
 * Returns the number of microseconds since last time the button was clicked
 */
uint64_t InputHandler::timeBetweenClicks(int mousebutton)
{
    return mouse_timestamps[mousebutton] - prev_mouse_timestamps[mousebutton];
}

uint64_t InputHandler::timeSinceClick(int mousebutton)
{
    timespec now;

    clock_gettime(CLOCK_MONOTONIC, &now);

    uint64_t time_now = now.tv_nsec / 1000 + now.tv_sec * 1e6;

    return time_now - mouse_timestamps[mousebutton];
}

bool InputHandler::isHeld(int mousebutton)
{
    return mouse_states[mousebutton] == IH_PRESS
        && prev_mouse_states[mousebutton] == IH_PRESS
        && timeSinceClick(mousebutton) > 2e5;
}

bool InputHandler::isPressed(int key)
{
    return prev_kb_key_states[key] == IH_RELEASE && kb_key_states[key] == IH_PRESS;
}

bool InputHandler::isRepeated(int key)
{
    return (prev_kb_key_states[key] == IH_PRESS || prev_kb_key_states[key] == IH_REPEAT)
        &&
        (kb_key_states[key] == IH_REPEAT || kb_key_states[key] == IH_PRESS);
}

bool InputHandler::keyReleased(int key)
{
    return prev_kb_key_states[key] >= IH_PRESS && kb_key_states[key] == IH_RELEASE;
}

bool InputHandler::mouseButtonReleased(int key)
{
    if(mouse_states[key] == IH_RELEASE && prev_mouse_states[key] == IH_PRESS) {
        prev_mouse_states[key] = IH_RELEASE;
        return true;
    }

    return false;
}

void InputHandler::update()
{

    // Update mouse button states
    for(int i = 0; i < 8; i++)
    {
        if(isClicked(i))
        { prev_mouse_states[i] = IH_PRESS; }

        if(isDoubleClicked(i))
        { prev_mouse_timestamps[i] = 0; }
    }

    // Update Keyboard key states
    for(int i = 0; i < 512; i++)
    {
        if(isPressed(i)) { prev_kb_key_states[i] = IH_PRESS; }

        if(keyReleased(i)) { prev_kb_key_states[i] = IH_RELEASE; }

    }
}

IH_MousePosition InputHandler::getMousePos()
{
    return (IH_MousePosition) {
        .x=xpos, .y=ypos
    };
}



