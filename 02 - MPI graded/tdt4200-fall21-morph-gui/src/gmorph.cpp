#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <WindowManager.hpp>
#include <CleanupManager.hpp>
#include <Shader.hpp>
#include <memory>
#include <Image.hpp>
#include <ImageRenderer.hpp>
#include <FeatureLine.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <InputHandler.hpp>
#include <FileReader.hpp>
#include <MorphFile.hpp>
#include <ArgumentParser.hpp>

void GLAPIENTRY glErrorHandler(
                GLenum source,
                 GLenum type,
                 GLuint id,
                 GLenum severity,
                 GLsizei length,
                 const GLchar* message,
                 const void* userParam
                )
{

#if DEBUG
    fprintf( stderr, "GL CALLBACK: %s type = 0x%x, severity = 0x%x, message = %s\n",
           ( type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : "" ),
            type, severity, message );
#endif
}



int main(int argc, char **argv)
{

    //////////////////////////////////
    // INITIALIZATION   /  SETUP    //
    //////////////////////////////////
    CleanupManager::init();

    if( ! glfwInit() )
    {
        std::cerr << "GLFW could not be initialized!" << std::endl;
    }

    print_usage_info();

    Arguments *args = parse_args(argc, argv);


    // Load Images
    Image::flipOnLoad();
    Image secondImage{args->dest_img->c_str(), 3};
    Image firstImage{args->source_img->c_str(), 3};

    // Set up Window Hints for GLFW
    Window::hint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    Window::hint(GLFW_CONTEXT_VERSION_MINOR, 1);
    Window::hint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create a window and make a render context
    ManagedWindow *window = new ManagedWindow{secondImage.getWidth(), secondImage.getHeight(), "PrimaryWindow"};
    WindowManager::setPrimary(window);
    window->show();

    if (glewInit() != GLEW_OK)
    {
        std::cerr << "GLEW could not be initialized!" << std::endl;
    }

    // Read configuration
    std::string morphFilePath = (args->featureline_file) ? *args->featureline_file : "config";

    MorphFile morphFile{morphFilePath.c_str()};
    morphFile.read(FeatureLineManager::getInstance());


#if DEBUG
    glEnable(GL_DEBUG_OUTPUT);
    glDebugMessageCallback(glErrorHandler, 0);
#endif


    InputHandler::setupCallbacks();


    /////////////////////////////////////
    // Shader initialization and setup //
    /////////////////////////////////////

    ImageRenderer *imageRenderer = ImageRenderer::getInstance();

    imageRenderer->setImage(0, &firstImage);
    imageRenderer->setImage(1, &secondImage);
    imageRenderer->setActive(0);




    /**
     * Set the orthogonal projection matrices to simplify correlation between screen coordinates
     * and geometric coordinates
     */
    Shader *imageShader = imageRenderer->getShader();
    unsigned int imageShader_OrthoLocation = glGetUniformLocation(imageShader->get(), "ortho");
    unsigned int lineShader_OrthoLocation  = glGetUniformLocation(SimplePointShader::getInstance()->get(), "ortho");
    unsigned int pointShader_OrthoLocation = glGetUniformLocation(FeatureLineShader::getInstance()->get(), "ortho");

    glm::mat4 MVP = glm::ortho(0.0f, (float) firstImage.getWidth(), 0.0f, (float) firstImage.getHeight(), -1.0f, 1.0f);
    glUseProgram(imageShader->get());
    glUniformMatrix4fv(imageShader_OrthoLocation, 1, GL_FALSE, glm::value_ptr(MVP));

    glUseProgram(SimplePointShader::getInstance()->get());
    glUniformMatrix4fv(pointShader_OrthoLocation, 1, GL_FALSE, glm::value_ptr(MVP));

    glUseProgram(FeatureLineShader::getInstance()->get());
    glUniformMatrix4fv(lineShader_OrthoLocation, 1, GL_FALSE, glm::value_ptr(MVP));

    FL_Point *currently_dragging = nullptr;

    FeatureLineManager::getInstance()->setActiveSet(true);

    while( ! window->shouldClose() )
    {
        GLFWwindow *current_window = WindowManager::getPrimary()->get();
        glfwPollEvents();

        // Handle dragging of nodes
        FeatureLineManager::getInstance()->handleDragging();


        /**
         * Swapping between first and second image
         */
        if( InputHandler::isPressed(GLFW_KEY_RIGHT) )
        {
            imageRenderer->setActive(1);
            FeatureLineManager::getInstance()->setActiveSet(false);
        }

        if( InputHandler::isPressed(GLFW_KEY_LEFT) )
        {
            imageRenderer->setActive(0);
            FeatureLineManager::getInstance()->setActiveSet(true);
        }


        /**
         * Adding new nodes / lines
         */
        if( InputHandler::isDoubleClicked(GLFW_MOUSE_BUTTON_LEFT) )
        {
            IH_MousePosition pos = InputHandler::getMousePos();


            FL_Point *hitPoint = FeatureLineManager::getInstance()
                ->hit(pos.x, firstImage.getHeight() - pos.y );

            if( nullptr == hitPoint ) {
                FeatureLineManager::getInstance()
                    ->addPoint(pos.x, firstImage.getHeight() - pos.y);
            }

            if(InputHandler::isRepeated(GLFW_KEY_LEFT_SHIFT) && nullptr != hitPoint )
            {
                FeatureLineManager::getInstance()->remove(hitPoint);
            }
        }

        /**
         * Writing the line set
         */
        if( InputHandler::isPressed(GLFW_KEY_W) )
        {
            morphFile.write(FeatureLineManager::getInstance());
        }



        glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

        imageRenderer->render();


        FeatureLineManager::getInstance()->render();

        InputHandler::update();

        window->update();
    }

}
