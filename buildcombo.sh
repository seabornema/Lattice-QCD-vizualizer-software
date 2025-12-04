#!/bin/bash
set -e

# Compiler
CXX=g++
CXXFLAGS="-g -std=c++17 -Wall -Wextra -O2"

# Include directories
INCLUDES="-I HeaderFiles -I glad/include -I src/glfw/3.4/include -I src/glm -pthread -fopenmp -I /usr/include/eigen3 -I /usr/include/freetype2"

# Source files
SRC_FILES="comborender.cpp \
           SourceFiles/shaderClass.cpp \
           SourceFiles/VAO.cpp \
           SourceFiles/VBO.cpp \
           SourceFiles/EBO.cpp \
           SourceFiles/text_helper.cpp\
           glad/src/glad.c \
	       SourceFiles/Camera.cpp \
            SourceFiles/MC_helper.cpp"

# GLFW static library
GLFW_LIB="src/glfw/3.4/build/src/libglfw3.a"

# Linker flags (Linux OpenGL)
LIBS="-lGL -ldl -lpthread -lX11 -lXrandr -lXi -lfreetype"

# Build
echo "building project..."
$CXX $CXXFLAGS $SRC_FILES $INCLUDES $GLFW_LIB $LIBS -o combo

echo "project build complete"
