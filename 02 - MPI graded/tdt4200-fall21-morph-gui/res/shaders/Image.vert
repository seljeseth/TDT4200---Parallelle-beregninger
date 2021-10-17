#version 410 core

layout (location = 0) in vec3 pos;
layout (location = 1) in vec2 uv;

uniform mat4 ortho;

out vec4 vertexPosition;
out vec2 vertexUV;


void main()
{
    vertexPosition = vec4(pos.xyz, 1.0);
    vertexUV = uv;

    gl_Position = ortho * vec4(pos.xyz, 1.0);
}
