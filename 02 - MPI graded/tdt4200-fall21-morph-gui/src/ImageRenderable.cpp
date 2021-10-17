#include <ImageRenderable.hpp>
#include <GL/glew.h>
#include <glm/vec3.hpp>
#include <cstdio>

ImageRenderable::ImageRenderable()
{
    glGenVertexArrays(1, &VAO);

    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    glGenBuffers(1, &UV_Buffer);

    // _generateBuffers();
}

ImageRenderable::~ImageRenderable()
{
    delete[] indices;
    delete[] vertices;

    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteBuffers(1, &UV_Buffer);

    glDeleteVertexArrays(1, &VAO);
}


void ImageRenderable::render()
{
    glUseProgram(imageShader->get());

    glBindVertexArray(VAO);

    glBindTexture(GL_TEXTURE_2D, textureID);

    glDrawElements(GL_TRIANGLES, numIndices, GL_UNSIGNED_INT, nullptr);
}

void ImageRenderable::setImage(Image *image)
{

    glBindVertexArray(VAO);

    glActiveTexture(GL_TEXTURE0);

    glGenTextures(1, &textureID);

    glBindTexture(GL_TEXTURE_2D, textureID);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image->getWidth(), image->getHeight(), 0, GL_RGB, GL_UNSIGNED_BYTE, image->getData());

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glGenerateMipmap(GL_TEXTURE_2D);

    glBindVertexArray(0);

    glBindTexture(GL_TEXTURE_2D, 0);

    this->image = image;

    _generateBuffers();
}

void ImageRenderable::setShader(Shader *shader)
{
    imageShader = shader;

    glBindVertexArray(VAO);
    shader->use();
    glBindVertexArray(0);
}

/**
 * TODO: Write documentation for this one.
 */
void ImageRenderable::_generateBuffers()
{

    glBindVertexArray(VAO);


    ///////// VERTICES
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    vertices = new float[4*3];

    using vectype = glm::vec<3, float>;

    vectype *vertices = (vectype *) this->vertices;

    vertices[0] = {0.0f, 0.0f, 0.0f};
    vertices[1] = {0.0f, static_cast<float>(image->getHeight()), 0.0f};
    vertices[2] = { static_cast<float>(image->getWidth()), static_cast<float>(image->getHeight()), 0.0f};
    vertices[3] = { static_cast<float>(image->getWidth()), 0.0f, 0.0f};


    glEnableVertexArrayAttrib(VAO, 0);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)*4*3, (void *) this->vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void *) 0);

    using uint_vectype = glm::vec<3, unsigned int>;

    ///////// UV COORDINATES
    glBindBuffer(GL_ARRAY_BUFFER, UV_Buffer);
    float uv_coords[2*4] = {
        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f,
        1.0f, 0.0f
    };

    glBufferData(GL_ARRAY_BUFFER, sizeof(float)*2*4, uv_coords, GL_STATIC_DRAW);
    glEnableVertexArrayAttrib(VAO, 1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, (void *) 0);


    ///////// ELEMENTS (INDICES)

    indices = new unsigned int[6];

    uint_vectype *indices = (uint_vectype *) this->indices;

    indices[0] = {0, 1, 2};
    indices[1] = {2, 3, 0};


    numIndices = 6;

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int)*6, indices, GL_STATIC_DRAW);

    glBindVertexArray(0);

}


Image *ImageRenderable::getImage()
{
    return image;
}
