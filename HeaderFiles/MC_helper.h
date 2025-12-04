#ifndef MC_CLASS_H
#define MC_CLASS_H

#include<array>
#include<GLFW/glfw3.h>
#include<vector>
#include<cmath>
#include<omp.h>

extern const std::array<std::array<int,2>,12> EdgeVertexIndices;

extern const std::array<int,256> EdgeMasks;

extern const std::array<std::array<int,16>,256> TriangleTable;
uint8_t find_cube_index(const std::array<double,8>& corners, double isovalue);

std::array<int,16> find_edge_set(int index);
GLfloat lerp_pos(GLfloat a,GLfloat b,GLfloat isovalue);
std::vector<GLfloat> coord_from_edge(int index,GLfloat x0,GLfloat y0,GLfloat z0, GLfloat size);
std::vector<GLfloat> calculate_normal(std::vector<GLfloat> points);
std::vector<GLfloat> triangles_from_cube(const std::array<double,8>& corners, double isovalue, GLfloat x0, GLfloat y0,GLfloat z0, GLfloat size,std::vector<GLfloat> color,std::vector<GLfloat> fade_color);

std::vector<GLfloat> marching_builder(std::vector<std::vector<std::vector<double>>> lattice, double isolevel,GLfloat lattice_size,std::vector<GLfloat> color,std::vector<GLfloat> fade_color);
std::vector<GLuint> index_generator(int sizeo);

#endif