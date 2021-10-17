#version 410 core

layout (location = 0) in vec2 in_vPos;

uniform mat4 ortho;

out vec4 vPos;

void main ()
{
    gl_Position = ortho * vec4(in_vPos.xy,  0, 1);
}
