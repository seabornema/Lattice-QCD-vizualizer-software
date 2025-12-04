#include <iostream>
#include <vector>
#include <array>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cmath>
#include <random>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <algorithm>
#include <map>


#include "shaderClass.h"
#include "VAO.h"
#include "VBO.h"
#include "EBO.h"
#include "Camera.h"
#include "MC_helper.h"
#include "text_helper.h"
#include <ft2build.h>
#include FT_FREETYPE_H

#include <thread>
#include <ctime>
#include <Eigen/Dense>
#include <complex>


const unsigned int Window_width = 1200;
const unsigned int Window_height = 800;
int L = 16;
double iso = 0.0;
double iso_speed = 0.001;
GLfloat lattice_size = 10.0f;

const int Nt = 6;
const std::complex<double> I(0.0, 1.0);

Eigen::Matrix2cd sigma_x, sigma_y, sigma_z;
inline int tmod(int a,int b){
    int temp = a % b;
    return (temp<0) ? temp+b : temp;
}

fixed_variable iso_fixed("isolevel",&iso,0);
fixed_variable iso_speed_fixed("iso speed",&iso_speed,1);
variable_set GUI({iso_fixed,iso_speed_fixed},iso_fixed);

void init_paulis() {
    
    sigma_x << 0, 1,
               1, 0;
    sigma_y << 0, -I,
               I,  0;
    sigma_z << 1,  0,
               0, -1;
}


struct Links
{
    std::vector<Eigen::Matrix2cd> U;
};

double levi_c(const std::vector<int>& set) {
    Eigen::Matrix4d temp = Eigen::Matrix4d::Zero();
    for(int i = 0; i < 4; i++){
        temp(i, set[i]) = 1.0;
    }
    return temp.determinant(); 
}
double beta = 2.0;

std::vector<std::vector<std::vector<std::vector<Links>>>> lattice(L, std::vector<std::vector<std::vector<Links>>>(L, std::vector<std::vector<Links>>(L, std::vector<Links>(Nt))));
static std::mt19937 rng(std::random_device{}());

double random_range(double a, double b) {
    static std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(a, b);
    return dist(rng);
}

double gauss_number() {
    static std::mt19937 rng(std::random_device{}());
    static std::normal_distribution<double> nd(0.0, 1.0);
    return nd(rng);
}

double x0_generator(double c,double beta) {
	//looks good
    double lambda2 = 1;
    double rsq = 10;
    while(rsq > 1.0 - lambda2) {
        double r1 = random_range(0.0, 1.0);
        double r2 = random_range(0.0, 1.0);
        double r3 = random_range(0.0, 1.0);

        lambda2 = -(1.0 / (2.0 * beta * c)) * (std::log(r1) + std::pow(std::cos(2.0 * M_PI * r2), 2) * std::log(r3));
        rsq = std::pow(random_range(0.0, 1.0),2);
    }
    return 1-2*lambda2;
}

std::array<double,4> xvec_generator(double x0){
	//looks good
    while(true) {
        double a = random_range(-1,1);
        double b = random_range(-1,1);
        double c = random_range(-1,1);
        double norm = ((a*a) + (b*b) + (c*c));

        if(norm <= 1.0) {
            double cool_const = (std::sqrt((1-(x0*x0))/norm));
            return {x0,cool_const*a,cool_const*b,cool_const*c};
        }

    }
}

std::complex<double> det(Eigen::Matrix2cd inpu) {
    return inpu.determinant();
}
Eigen::Matrix2cd SU2_constructor(const std::array<double,4>& xvec){
    return xvec[0]*Eigen::Matrix2cd::Identity() + xvec[1]*I*sigma_x + xvec[2]*I*sigma_y + xvec[3]*I*sigma_z;
}

Eigen::Matrix2cd gauge_link(const std::array<int,4>& position,int direction, std::vector<std::vector<std::vector<std::vector<Links>>>>& lattice){
    return lattice[position[0]][position[1]][position[2]][position[3]].U[direction];
}

Eigen::Matrix2cd staples(const std::array<int,4>& position,const std::vector<std::vector<std::vector<std::vector<Links>>>>& lattice,int mu){
    int x = position[0];
    int y = position[1];
    int z = position[2];
    int t = position[3];
    Eigen::Matrix2cd sum = Eigen::Matrix2cd::Zero();
    std::array<int,4> U_direction = {0,0,0,0};
    U_direction[mu] = 1;
    for(int j = 0; j < 4; j ++){
            if(mu==j) {
                continue;
            }else {
                //i is mu direction
                //j is nu direction

                std::array<int,4> V_direction = {0,0,0,0};
                V_direction[j] = 1;
                
                Eigen::Matrix2cd U1 = lattice[tmod(x+U_direction[0],L)][tmod(y+U_direction[1],L)][tmod(z+U_direction[2],L)][tmod(t+U_direction[3],Nt)].U[j];
                Eigen::Matrix2cd U2 = lattice[tmod(x+V_direction[0],L)][tmod(y+V_direction[1],L)][tmod(z+V_direction[2],L)][tmod(t+V_direction[3],Nt)].U[mu].adjoint();
                Eigen::Matrix2cd U3 = lattice[x][y][z][t].U[j].adjoint();
               
                Eigen::Matrix2cd U4 = lattice[tmod(x+U_direction[0]-V_direction[0],L)][tmod(y+U_direction[1]-V_direction[1],L)][tmod(z+U_direction[2]-V_direction[2],L)][tmod(t+U_direction[3]-V_direction[3],Nt)].U[j].adjoint();
                Eigen::Matrix2cd U5 = lattice[tmod(x-V_direction[0],L)][tmod(y-V_direction[1],L)][tmod(z-V_direction[2],L)][tmod(t-V_direction[3],Nt)].U[mu].adjoint();
                Eigen::Matrix2cd U6 = lattice[tmod(x-V_direction[0],L)][tmod(y-V_direction[1],L)][tmod(z-V_direction[2],L)][tmod(t-V_direction[3],Nt)].U[j];


                sum += ((U1*U2*U3) + (U4*U5*U6));

        }
    }
    return sum;
}




Eigen::Matrix2cd plaquette(const std::vector<std::vector<std::vector<std::vector<Links>>>>& lattice,int x,int y,int z,int t,int mu,int nu) {
    std::array<int,4> mu_direction = {0,0,0,0};
    mu_direction[mu] = 1;
    std::array<int,4> nu_direction = {0,0,0,0};
    nu_direction[nu] = 1;
                                

    return  lattice[x][y][z][t].U[mu]*
            lattice[tmod(x+mu_direction[0],L)][tmod(y+mu_direction[1],L)][tmod(z+mu_direction[2],L)][tmod(t+mu_direction[3],Nt)].U[nu]*
            lattice[tmod(x+nu_direction[0],L)][tmod(y+nu_direction[1],L)][tmod(z+nu_direction[2],L)][tmod(t+nu_direction[3],Nt)].U[mu].adjoint()*
            lattice[x][y][z][t].U[nu].adjoint();

}
Eigen::Matrix2cd clover(const std::vector<std::vector<std::vector<std::vector<Links>>>>& lattice,
                        int x,int y,int z,int t,int mu,int nu) {
    std::array<int,4> mud = {0,0,0,0}; mud[mu] = 1;
    std::array<int,4> nud = {0,0,0,0}; nud[nu] = 1;

    // Sum the four plaquettes that form the clover
    Eigen::Matrix2cd P1 = plaquette(lattice, x, y, z, t, mu, nu);
    Eigen::Matrix2cd P2 = plaquette(lattice, tmod(x-mud[0],L), tmod(y-mud[1],L),
                                    tmod(z-mud[2],L), tmod(t-mud[3],Nt), mu, nu);
    Eigen::Matrix2cd P3 = plaquette(lattice, tmod(x-nud[0],L), tmod(y-nud[1],L),
                                    tmod(z-nud[2],L), tmod(t-nud[3],Nt), mu, nu);
    Eigen::Matrix2cd P4 = plaquette(lattice, tmod(x-nud[0]-mud[0],L),
                                    tmod(y-nud[1]-mud[1],L),
                                    tmod(z-nud[2]-mud[2],L),
                                    tmod(t-nud[3]-mud[3],Nt), mu, nu);

    Eigen::Matrix2cd Q = P1 + P2 + P3 + P4;

    // Field-strength (anti-Hermitian) approximation:
    const std::complex<double> I(0.0,1.0);
    Eigen::Matrix2cd F = (Q - Q.adjoint()) * (1.0 / (8.0 * I));

    // Optional: project to traceless part (for SU(2) it's usually traceless already up to rounding)
    std::complex<double> tr = F.trace() / 2.0;
    F = F - Eigen::Matrix2cd::Identity() * tr;

    return F;
}

double action(const std::vector<std::vector<std::vector<std::vector<Links>>>>& lattice,const std::array<int,4>& pos, int mu){
    double sum =0;
    for(int nu=0;nu<4; nu++){
        if(mu==nu) {
            continue;
        }
        sum+=  plaquette(lattice,pos[0],pos[1],pos[2],pos[3],mu,nu).trace().real();
    }
    return sum;
}


void cold_start(std::vector<std::vector<std::vector<std::vector<Links>>>>& lattice)
{
    for (int x = 0; x < L; ++x){
        for (int y = 0; y < L; ++y){
            for (int z = 0; z < L; ++z){
                for (int t = 0; t < Nt; ++t)
                {
                    lattice[x][y][z][t].U.resize(4); 
                    for (int mu = 0; mu < 4; ++mu){
                        lattice[x][y][z][t].U[mu] = Eigen::Matrix2cd::Identity();
                    }
                }
            }      
        }
    }

}

void heatbath_evolve_single(const std::array<int,4>& position,std::vector<std::vector<std::vector<std::vector<Links>>>>& lattice,int direction,double beta){
    Eigen::Matrix2cd staps = staples(position,lattice,direction);
    double cval = std::sqrt(std::real(det(staps)));
   
    std::array<double,4> temp_guy = xvec_generator(x0_generator(cval,beta));
    Eigen::Matrix2cd X = SU2_constructor(temp_guy);
   
    Eigen::Matrix2cd Ut = (X* staps.adjoint())/cval;

    lattice[position[0]][position[1]][position[2]][position[3]].U[direction] = Ut/(std::sqrt(det(Ut)));    
}

Eigen::Matrix2cd sigma_matrix(const Eigen::Matrix2cd& staple)
{
    double det_h = (staple.determinant().real());
    double norm = std::sqrt(det_h);

    Eigen::Matrix2cd Sigma = staple / norm;

    return Sigma;
}
void action_minimizing(const std::array<int,4>& position,std::vector<std::vector<std::vector<std::vector<Links>>>>& lattice,int direction){
       lattice[position[0]][position[1]][position[2]][position[3]].U[direction] = sigma_matrix(staples(position,lattice,direction));
}

void full_sweep(std::vector<std::vector<std::vector<std::vector<Links>>>>& lattice,int n,double beta){
    for(int step=0; step<n;step++){
       for(int i =0; i< L*L*L*Nt*4; i++){
        int dir = i % 4;
        int x = (i / 4) % L;
        int y = (i / (4*L)) % L;
        int z = (i / (4*L*L)) % L;
        int t = (i / (4*L*L*L)) % Nt;

        heatbath_evolve_single({x,y,z,t},lattice,dir,beta);
    }
    }

}

double topological_single(const std::vector<std::vector<std::vector<std::vector<Links>>>>& lattice,const std::array<int,4> pos) {
    double prefactor = -1.0/((32.0)*(M_PI*M_PI));
    double sum = 0; 
    std::vector<int> starter = {0,1,2,3}; 
    do{
        sum += levi_c(starter)*(clover(lattice,pos[0],pos[1],pos[2],pos[3],starter[0],starter[1])*clover(lattice,pos[0],pos[1],pos[2],pos[3],starter[2],starter[3])).trace().real();
    }while(next_permutation(starter.begin(),starter.end()));
    do{
        sum += levi_c(starter)*(clover(lattice,pos[0],pos[1],pos[2],pos[3],starter[0],starter[1])*clover(lattice,pos[0],pos[1],pos[2],pos[3],starter[2],starter[3])).trace().real();
    }while(next_permutation(starter.begin(),starter.end()));
    return prefactor*sum;
}

std::vector<std::vector<std::vector<double>>> topological_charge(const std::vector<std::vector<std::vector<std::vector<Links>>>>& lattice,int t) { 
    std::vector<std::vector<std::vector<double>>> temp(L, std::vector<std::vector<double>>(L, std::vector<double>(L, 0.0))); 
    for (int i = 0; i < L * L * L; i++) {
         int x = i % L; 
         int y = (i / L) % L; 
         int z = (i / (L * L)) % L; 
         temp[x][y][z] = topological_single(lattice,{x,y,z,t}); 
        } 
        return temp;
}

std::vector<std::vector<std::vector<std::complex<double>>>> Polyakov_loop(const std::vector<std::vector<std::vector<std::vector<Links>>>>& lattice) {
    
    std::vector<std::vector<std::vector<std::complex<double>>>> temp(L, std::vector<std::vector<std::complex<double>>>(L, std::vector<std::complex<double>>(L, std::complex<double>(0.0, 0.0))));
    for (int i = 0; i < L * L * L; i++) {
        int x = i % L;
        int y = (i / L) % L;
        int z = (i / (L * L)) % L;
        Eigen::Matrix2cd starter = lattice[x][y][z][0].U[3];
        for (int t = 1; t < Nt; t++) {
            starter = starter * lattice[x][y][z][t].U[3];
            
        }
    
        temp[x][y][z] = (double)0.5*starter.trace().real();
    }

    return temp;
}



std::vector<std::vector<std::vector<double>>> Polyakov_loop_real(const std::vector<std::vector<std::vector<std::vector<Links>>>>& lattice) {
    
    std::vector<std::vector<std::vector<double>>> temp(L, std::vector<std::vector<double>>(L, std::vector<double>(L, 0.0)));
    for (int i = 0; i < L * L * L; i++) {
        int x = i % L;
        int y = (i / L) % L;
        int z = (i / (L * L)) % L;
        Eigen::Matrix2cd starter = lattice[x][y][z][0].U[3];
        for (int t = 1; t < Nt; t++) {
            starter = starter * lattice[x][y][z][t].U[3];
            
        }
    
        temp[x][y][z] = (double)0.5*starter.trace().real();
    }

    return temp;
}
void action_minimizing_sweep(std::vector<std::vector<std::vector<std::vector<Links>>>>& lattice, int n) {
    for(int step = 0; step < n; step++) {


        for(int i = 0; i < L*L*L*Nt*4; i++){
            int dir = i % 4;
            int x = (i / 4) % L;
            int y = (i / (4*L)) % L;
            int z = (i / (4*L*L)) % L;
            int t = (i / (4*L*L*L)) % Nt;
            action_minimizing({x,y,z,t}, lattice, dir);
        }
    }
}



std::vector<double> potential_generator(std::vector<std::vector<std::vector<std::vector<Links>>>>& lattice) {
    int halfL = L / 2;
    std::vector<double> PP(halfL, 0.0);
    std::vector<double> num(halfL, 0.0);
    std::vector<std::vector<std::vector<std::complex<double>>>> ploops = Polyakov_loop(lattice);
    
    for (int i = 0; i < halfL; i++) {
        for (int j = 0; j < halfL; j++) {
            for (int k = 0; k < halfL; k++) {
                for (int r = 0; r < halfL; r++) {
                    int ip = (i + r) % L;
                    int jp = (j + r) % L;
                    int kp = (k + r) % L;
                    
                    std::complex<double> P0 = ploops[i][j][k];
               
                    PP[r] += (P0*(std::conj(ploops[ip][j][k])) +P0*(std::conj(ploops[i][jp][k])) +P0*(std::conj(ploops[i][j][kp]))).real();
                    num[r] += 3.0;  
                }
            }
        }
        
    }

    return PP;
}

std::vector<GLfloat> box_vertices = {
    0.0f,0.0f,0.0f, // 0 
    0.0f,0.0f,lattice_size, //1 z
    0.0f,lattice_size,0.0f, //2 y
    lattice_size,0.0f,0.0f, //3 x
    lattice_size,lattice_size,0.0f, //4 xy
    0.0f,lattice_size,lattice_size, //5  yz
    lattice_size,0.0f,lattice_size, //6 xz
    lattice_size,lattice_size,lattice_size //7 xyz
};

std::vector<GLuint> box_indices = {
    0,1, 
    0,2, 
    0,3,
    3,4,
    1,5,
    2,4,
    2,5,
    6,3,
    6,1,
    7,4,
    7,5,
    6,7
};


std::vector<GLfloat> lattice_color = {0.93333f,0.0647f,0.023523f};
std::vector<GLfloat> lattice_fade_color = {0.9643f,0.4647f,0.3843f};


std::vector<GLfloat> vertices;
std::vector<GLuint> indices;


std::vector<std::vector<std::vector<double>>> temp;

void regenerate_triangles(double isolevel,const std::vector<std::vector<std::vector<double>>>& lattice,std::vector<GLfloat>& vertices,std::vector<GLuint>& indices,const VBO& VBO1,const EBO& EBO1){
    vertices =  marching_builder(lattice,isolevel,lattice_size,lattice_color,lattice_fade_color);
    indices  = index_generator(vertices.size());
    
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,EBO1.ID);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size()*sizeof(GLuint),&indices.front(),GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER,VBO1.ID);
    glBufferData(GL_ARRAY_BUFFER,vertices.size()*sizeof(GLfloat),&vertices.front(),GL_STATIC_DRAW);
}


void animate(const std::vector<std::vector<std::vector<std::vector<Links>>>>& lattice,double iso,std::vector<GLfloat>& vertices,std::vector<GLuint>& indices,const VBO& VBO1,const EBO& EBO1,int index) {
    regenerate_triangles(iso,topological_charge(lattice,index%Nt),vertices,indices,VBO1,EBO1);
   // std::this_thread::sleep_for(std::chrono::milliseconds(50));
}


std::vector<GLfloat> lightVertices = {
    -0.1f, -0.1f, 0.1f,
    -0.1f, -0.1f, -0.1f,
    0.1f, -0.1f, -0.1f,
    0.1f, -0.1f, 0.1f,
    -0.1f, 0.1f, 0.1f,
    -0.1f, 0.1f, -0.1f,
    0.1f, 0.1f, -0.1f,
    0.1f, 0.1f, 0.1f  
};
   
std::vector<GLuint> lightIndices = {
    0,1,2,
    0,2,3,
    0,4,7,
    0,7,3,
    3,7,6,
    3,6,2,
    2,6,5,
    2,5,1,
    1,5,4,
    1,4,0,
    4,5,6,
    4,6,7
};

void detectscroll(int yoffset,variable_set& myguys){
   if(yoffset == 1){
        myguys.scrollup();
    } else if(yoffset ==-1){
        myguys.scrolldown();
    }
}
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    detectscroll(yoffset,GUI);
    if(yoffset == -1){
    std::cout << "scrolling" << std::endl;
    }
}

int main() {
   
    init_paulis();

    

    cold_start(lattice);

    temp = Polyakov_loop_real(lattice);

    

    int N = temp.size();
 //   initalize();
    
    vertices =  marching_builder(temp,iso,N,lattice_color,lattice_fade_color);
    indices  = index_generator(vertices.size());

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(Window_width, Window_width, "window_name", NULL, NULL);
    if (window == NULL) {
        std::cout << "window_failed" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }
    glViewport(0,0,Window_width,Window_width);
  



    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);


     FT_Library ft;
    if (FT_Init_FreeType(&ft))
    {
    std::cout << "ERROR::FREETYPE: Could not init FreeType Library" << std::endl;
    return -1;
    }

    FT_Face face;
    if (FT_New_Face(ft, "ResourceFiles/Fonts/W95FA.ttf", 0, &face))
    {
    std::cout << "ERROR::FREETYPE: Failed to load font" << std::endl;  
    return -1;
    }
    FT_Set_Pixel_Sizes(face, 0, 48);  
    if (FT_Load_Char(face, 'X', FT_LOAD_RENDER))
    {
    std::cout << "ERROR::FREETYTPE: Failed to load Glyph" << std::endl;  
    return -1;
    }

    text_initialize(face);

    FT_Done_Face(face);
    FT_Done_FreeType(ft);

    
    Shader textShader("ResourceFiles/Shaders/text.vert","ResourceFiles/Shaders/text.frag");
   
    unsigned int textVAO, textVBO;
    glGenVertexArrays(1, &textVAO);
    glGenBuffers(1, &textVBO);
    glBindVertexArray(textVAO);
    glBindBuffer(GL_ARRAY_BUFFER, textVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 4, NULL, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);      
    
    Shader lightShader("ResourceFiles/Shaders/light.vert","ResourceFiles/Shaders/light.frag");

    VAO lightVAO;
    lightVAO.Bind();

    VBO lightVBO(lightVertices);
    EBO lightEBO(lightIndices);

    lightVAO.LinkAttrib(lightVBO,0,3,GL_FLOAT, 3*sizeof(float),(void*)0);

    lightVAO.Unbind();
    lightVBO.Unbind();
    lightEBO.Unbind();

    glm::vec4 lightColor = glm::vec4(1.0f,1.0f,1.0f,1.0f);
    glm::vec3 lightPos = glm::vec3(5.5f,15.0f,5.5f);
    glm::mat4 lightModel = glm::mat4(1.0f);

    lightModel = glm::translate(lightModel,lightPos);

    Shader simpleShader("ResourceFiles/Shaders/simple.vert","ResourceFiles/Shaders/simple.frag");

    VAO simpleVAO;
    simpleVAO.Bind();

    VBO simpleVBO(box_vertices);
    EBO simpleEBO(box_indices);

    simpleVAO.LinkAttrib(simpleVBO,0,3,GL_FLOAT,3*sizeof(float),(void*)0);
    
    simpleVAO.Unbind();
    simpleVBO.Unbind();
    simpleEBO.Unbind();

    glm::vec3 boxPos = glm::vec3(0.0f,0.0f,0.0f);
    glm::vec4 accentColor = glm::vec4(1.0f,1.0f,1.0f,1.0f);
    glm::mat4 boxModel = glm::mat4(1.0f);

    boxModel = glm::translate(boxModel,boxPos);

	Shader shaderProgram("ResourceFiles/Shaders/default.vert","ResourceFiles/Shaders/default.frag");

	VAO VAO1;
	VAO1.Bind();

	VBO VBO1(vertices);
	EBO EBO1(indices);

	VAO1.LinkAttrib(VBO1,0,3, GL_FLOAT, 12 * sizeof(GLfloat), (void*)0);
	VAO1.LinkAttrib(VBO1,1,3, GL_FLOAT, 12 * sizeof(GLfloat), (void*)(3 * sizeof(GLfloat)));
    VAO1.LinkAttrib(VBO1,2,3, GL_FLOAT, 12 * sizeof(GLfloat), (void*)(6 * sizeof(GLfloat)));
    VAO1.LinkAttrib(VBO1,3,3, GL_FLOAT, 12 * sizeof(GLfloat), (void*)(9 * sizeof(GLfloat)));


    VAO1.Unbind();
	VBO1.Unbind();
	EBO1.Unbind();

    glm::vec3 curvesPos(0.0f,0.0f,0.0f);
    glm::mat4 curvesModel = glm::mat4(1.0f);

    glm::mat4 projection = glm::ortho(0.0f,(float)Window_width,0.0f, (float)Window_width);

    textShader.Activate();
    glUniform3f(glGetUniformLocation(textShader.ID, "textColor"), accentColor.x,accentColor.y,accentColor.z);
    glUniformMatrix4fv(glGetUniformLocation(textShader.ID, "projection"), 1, GL_FALSE, glm::value_ptr(projection));


    simpleShader.Activate();
    glUniformMatrix4fv(glGetUniformLocation(simpleShader.ID, "model"),1,GL_FALSE,glm::value_ptr(boxModel));
    glUniform4f(glGetUniformLocation(simpleShader.ID,"accentColor"),accentColor.x,accentColor.y,accentColor.z,accentColor.w);
  
    lightShader.Activate();
    glUniformMatrix4fv(glGetUniformLocation(lightShader.ID, "model"),1,GL_FALSE,glm::value_ptr(lightModel));
    glUniform4f(glGetUniformLocation(lightShader.ID,"lightColor"),lightColor.x,lightColor.y,lightColor.z,lightColor.w);
    
    
    shaderProgram.Activate();
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram.ID,"model"),1,GL_FALSE,glm::value_ptr(curvesModel));
    glUniform4f(glGetUniformLocation(shaderProgram.ID, "lightColor"), lightColor.x, lightColor.y, lightColor.z,lightColor.w);
	glUniform3f(glGetUniformLocation(shaderProgram.ID,"lightPos"), lightPos.x,lightPos.y,lightPos.z);

    glEnable(GL_DEPTH_TEST);
    glLineWidth(3.0f);


    Camera camera(Window_width,Window_width,glm::vec3(0.0f,0.0f,2.0f));
	//rendering loop

    int iterator = 0;
    while (!glfwWindowShouldClose(window)) {
        glClearColor(0.0f,0.0f,0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glfwSetScrollCallback(window,scroll_callback);
        

        RenderText(textShader,GUI.current_guy, 25.0f, 25.0f, 1.0f, glm::vec3(0.5, 0.8f, 0.2f),textVBO,textVAO);

        camera.Inputs(window);
        camera.updateMatrix(45.0f,0.1f,100.0f);

        lightShader.Activate();
        camera.Matrix(lightShader, "camMatrix");
        
        lightVAO.Bind();
        //glDrawElements(GL_TRIANGLES,lightIndices.size(),GL_UNSIGNED_INT,0); 
   
        
      
        simpleShader.Activate();
        camera.Matrix(simpleShader,"camMatrix");
        simpleVAO.Bind();
        glDrawElements(GL_LINES,box_indices.size(),GL_UNSIGNED_INT,0);


        shaderProgram.Activate();
        glUniform3f(glGetUniformLocation(shaderProgram.ID,"camPos"),camera.Position.x,camera.Position.y,camera.Position.z);
        glUniform3f(glGetUniformLocation(shaderProgram.ID,"camNorm"),camera.Orientation.x,camera.Orientation.y,camera.Orientation.z);

        camera.Matrix(shaderProgram,"camMatrix");
		
        VAO1.Bind();
     
        if(glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS){
            regenerate_triangles(iso,temp,vertices,indices,VBO1,EBO1);
            int currenttype = GUI.current_guy.type;
            if(currenttype == 0){
            *GUI.current_guy.value += iso_speed;
            }else if(currenttype ==1) {
            *GUI.current_guy.value *= 1.1;
            }
        }
        if(glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS){
            regenerate_triangles(iso,temp,vertices,indices,VBO1,EBO1);
            int currenttype = GUI.current_guy.type;
            if(currenttype == 0){
            *GUI.current_guy.value -= iso_speed;
            }else if(currenttype ==1) {
            *GUI.current_guy.value *= 0.9;
            }
        }
        if(glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS){
            full_sweep(lattice,1,beta);
            temp = Polyakov_loop_real(lattice);
            regenerate_triangles(iso,temp,vertices,indices,VBO1,EBO1);
        }
        if(glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS){
            action_minimizing_sweep(lattice,1);
            temp = topological_charge(lattice,0);
            regenerate_triangles(iso,temp,vertices,indices,VBO1,EBO1);
        }
         if(glfwGetKey(window, GLFW_KEY_U) == GLFW_PRESS){
            action_minimizing_sweep(lattice,1);
            temp = Polyakov_loop_real(lattice);
            regenerate_triangles(iso,temp,vertices,indices,VBO1,EBO1);
        }
        if(glfwGetKey(window, GLFW_KEY_Y) == GLFW_PRESS){
            animate(lattice,iso,vertices,indices,VBO1,EBO1,iterator%Nt);
            iterator++;
            std::this_thread::sleep_for(std::chrono::milliseconds(250));


        }
		glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT,0);
   
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

	VAO1.Delete();
	VBO1.Delete();
	EBO1.Delete();
	shaderProgram.Delete();
    lightVAO.Delete();
    lightVBO.Delete();
    lightEBO.Delete();
    lightShader.Delete();
textShader.Delete();

    simpleVAO.Delete();
    simpleVBO.Delete();
    simpleEBO.Delete();
    simpleShader.Delete();

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
