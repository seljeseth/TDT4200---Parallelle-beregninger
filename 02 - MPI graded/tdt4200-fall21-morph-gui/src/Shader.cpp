#include <Shader.hpp>
#include <string>
#include <cstring>
#include <iostream>
#include <FileReader.hpp>

Shader::Shader()
{
    shaderID = glCreateProgram();
}

Shader::Shader(const char* vertexShader, const char* fragmentShader)
{
    shaderID = glCreateProgram();

    add(vertexShader);
    add(fragmentShader);

    link();
}


void Shader::use()
{
    glUseProgram(shaderID);
}

void Shader::add(const char* shader)
{

    const char* extension = strrchr(shader, '.');

    if ( NULL == extension ) {
        std::cerr <<
            "The shader path is incorrectly formatted: " << shader << std::endl;
        abort();
    }

    std::string ext{extension + 1};

    GLenum shaderType;


    unsigned int* field_ID;

    if( ext == "frag" ) {
        shaderType = GL_FRAGMENT_SHADER;
        field_ID = &fragmentID;
    }
    else if ( ext == "vert" ) {
        shaderType = GL_VERTEX_SHADER;
        field_ID = &vertexID;
    }
    else {
        std::cerr << "Unknown shader extension: " << ext << std::endl;
        abort();
    }

    *field_ID = glCreateShader(shaderType);

    auto source = FileReader::readFile(shader);
    const char* str = source->c_str();

    glShaderSource(*field_ID, 1, &str, 0);

    glCompileShader(*field_ID);


    int compileStatus;
    glGetShaderiv(*field_ID, GL_COMPILE_STATUS, &compileStatus);

    if ( ! compileStatus )
    {
        int length;
        glGetShaderiv(*field_ID, GL_INFO_LOG_LENGTH, &length);

        char infoLog[length+1];
        glGetShaderInfoLog(*field_ID, length, nullptr, infoLog);

        std::cerr << "Error compiling: " << ext << std::endl;
        std::cerr << infoLog << std::endl;
        abort();
    }
}

unsigned int Shader::get()
{
    return shaderID;
}


void Shader::link()
{
    glAttachShader(shaderID, vertexID);
    glAttachShader(shaderID, fragmentID);

    glLinkProgram(shaderID);

    int linkStatus;
    glGetProgramiv(shaderID, GL_LINK_STATUS, &linkStatus);

    if( ! linkStatus )
    {
        int errorLogLength;
        glGetProgramiv(shaderID, GL_INFO_LOG_LENGTH, &errorLogLength);

        char errorLog[errorLogLength];

        glGetProgramInfoLog(shaderID, errorLogLength, nullptr, errorLog);

        std::cerr << "Could not link shader program!" << std::endl;
        std::cerr << "------------------------------" << std::endl;
        std::cerr << errorLog << std::endl;

        abort();
    }
}
