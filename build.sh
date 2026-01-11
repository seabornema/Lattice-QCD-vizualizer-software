#!/bin/bash
set -e

CXX=g++
CXXFLAGS="-g -std=c++17 -Wall -Wextra -O2"

INCLUDES="-I HeaderFiles -I glad/include -I src/glfw/3.4/include -I src/glm -pthread -fopenmp"

SRC_FILES="openrender.cpp \
           SourceFiles/shaderClass.cpp \
           SourceFiles/VAO.cpp \
           SourceFiles/VBO.cpp \
           SourceFiles/EBO.cpp \
           glad/src/glad.c \
	       SourceFiles/Camera.cpp \
            SourceFiles/MC_helper.cpp"
GLFW_LIB="src/glfw/3.4/build/src/libglfw3.a"

LIBS="-lGL -ldl -lpthread -lX11 -lXrandr -lXi"

echo "building project..."
$CXX $CXXFLAGS $SRC_FILES $INCLUDES $GLFW_LIB $LIBS -o exp

echo "project build complete"
