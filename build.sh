#!/bin/bash
set -e

# Compiler
CXX=g++
CXXFLAGS="-g -std=c++17 -Wall -Wextra -O2"

# Include directories
INCLUDES="-I HeaderFiles -I glad/include -I src/glfw/3.4/include -I src/glm -pthread -fopenmp"

# Source files
SRC_FILES="openrender.cpp \
           SourceFiles/shaderClass.cpp \
           SourceFiles/VAO.cpp \
           SourceFiles/VBO.cpp \
           SourceFiles/EBO.cpp \
           glad/src/glad.c \
	       SourceFiles/Camera.cpp \
            SourceFiles/MC_helper.cpp"

# GLFW static library
GLFW_LIB="src/glfw/3.4/build/src/libglfw3.a"

# Linker flags (Linux OpenGL)
LIBS="-lGL -ldl -lpthread -lX11 -lXrandr -lXi"

# Build
echo "building project..."
$CXX $CXXFLAGS $SRC_FILES $INCLUDES $GLFW_LIB $LIBS -o exp

echo "project build complete"
