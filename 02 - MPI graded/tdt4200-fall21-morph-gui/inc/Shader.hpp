#ifndef _SHADER_HPP_
#define _SHADER_HPP_

#include <GL/glew.h>

class Shader
{
public:

    /**
     * Shader constructor.
     *
     * @param vertexShader The path to the vertex shader source.
     * @param fragmentShader The path to the fragment shader source.
     */
    Shader(const char* vertexShader, const char* fragmentShader);

    Shader();

    /**
     * Set this program to be the active program.
     */
    void use();

    unsigned int get();

    void add(const char *shader);

    void link();


private:
    unsigned int shaderID;

    bool linked = false;

    /// Vertex and Fragment Shader IDs
    /// Only relevant in the pre-link stage.
    unsigned int vertexID, fragmentID;

};

#endif
