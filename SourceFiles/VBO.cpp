#include <glad/glad.h>
#include <vector>
#include <GLFW/glfw3.h>
#include "../HeaderFiles/VBO.h"

VBO::VBO(const std::vector<GLfloat>& vertices) {
    glGenBuffers(1,&ID);
    glBindBuffer(GL_ARRAY_BUFFER,ID);
    glBufferData(GL_ARRAY_BUFFER,vertices.size()*sizeof(GLfloat),&vertices.front(),GL_STATIC_DRAW);
}
void VBO::Bind(){
    glBindBuffer(GL_ARRAY_BUFFER,ID);
}
void VBO::Unbind() {
    glBindBuffer(GL_ARRAY_BUFFER,0);
}
void VBO::Delete(){
    glDeleteBuffers(1,&ID);
}