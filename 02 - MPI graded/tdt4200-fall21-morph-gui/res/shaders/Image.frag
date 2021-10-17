#version 410 core

in vec4 vertexPosition;
in vec2 vertexUV;

uniform sampler2D image;

out vec4 FragColor;

void main()
{
    // color = vec4(1.0, 1.0, 1.0, 1.0);
    FragColor = texture(image, vertexUV);

}
