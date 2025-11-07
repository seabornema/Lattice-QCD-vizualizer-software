#include <iostream>
#include <random>
#include <cmath>
#include <ctime>
#include <Eigen/Dense>
#include <SFML/Graphics.hpp>
#include "zupergraph.h"
#include <thread>
#include <complex>


const int L= 16;
const int Nt = 6;
const std::complex<double> I(0.0, 1.0);



Eigen::Matrix2cd sigma_x, sigma_y, sigma_z;
inline int tmod(int a,int b){
    int temp = a % b;
    return (temp<0) ? temp+b : temp;
}

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

const int N_trials_per = 5000;
int sweep_number = 0;
std::vector<double> dist_holder(L/2,0);
std::vector<std::vector<double>> polyakov_holder(L/2,std::vector<double>(N_trials_per,0.0));

double beta = 2.7;
const int window_size = 600;

sf::Font font;

std::vector<std::vector<std::vector<std::vector<Links>>>> lattice(L, std::vector<std::vector<std::vector<Links>>>(L, std::vector<std::vector<Links>>(L, std::vector<Links>(Nt))));
static std::mt19937 rng(std::random_device{}());
#include <omp.h>


double random_range(double a, double b) {
    static thread_local std::mt19937 local_rng( std::random_device{}() ^ (std::hash<std::thread::id>()(std::this_thread::get_id())) );
    std::uniform_real_distribution<double> dist(a,b);
    return dist(local_rng);
}
double gauss_number() {
    static thread_local std::mt19937 local_rng( std::random_device{}() ^ (std::hash<std::thread::id>()(std::this_thread::get_id())) );
    static thread_local std::normal_distribution<double> nd(0.0,1.0);
    return nd(local_rng);
}

double x0_generator(double c,double beta) {
	//looks good
    bool found = false;
    double lambda = 1;
    double r = 10;
    while(r*r > 1.0 - lambda) {
        double r1 = random_range(0.0, 1.0);
        double r2 = random_range(0.0, 1.0);
        double r3 = random_range(0.0, 1.0);

        lambda = -(1.0 / (2.0 * beta * c)) * (std::log(r1) + std::pow(std::cos(2.0 * M_PI * r2), 2) * std::log(r3));
        r = random_range(0.0, 1.0);
    }
    return 1-2*lambda;
}
std::array<double,4> xvec_generator(double x0){
	//looks good
    while(true) {
        double a = random_range(-1,1);
        double b = random_range(-1,1);
        double c = random_range(-1,1);
        double norm = ((a*a) + (b*b) + (c*c));

        if(norm <= 1.0) {
            double cool_const = (std::sqrt(1-(x0*x0)))/norm;
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




double jackKnife(const std::vector<double>& data) {
    int N = data.size();
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    std::vector<double> jk_means(N);
    for (int i = 0; i < N; i++) {
        jk_means[i] = (sum - data[i]) / (N - 1);
    }
    return std::accumulate(jk_means.begin(), jk_means.end(), 0.0) / N;
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
   std::cout << "sweep_#" << sweep_number << " completed" << "\n";
    sweep_number++;
    }

}

std::vector<std::vector<std::vector<std::complex<long>>>> Polyakov_loop(const std::vector<std::vector<std::vector<std::vector<Links>>>>& lattice) {
    
    std::vector<std::vector<std::vector<std::complex<long>>>> temp(L, std::vector<std::vector<std::complex<long>>>(L, std::vector<std::complex<long>>(L, std::complex<long>(0.0, 0.0))));

    #pragma omp parallel for
    for (int i = 0; i < L * L * L; i++) {
        int x = i % L;
        int y = (i / L) % L;
        int z = (i / (L * L)) % L;
        Eigen::Matrix2cd starter = lattice[x][y][z][0].U[3];
        for (int t = 1; t < Nt; t++) {
            starter = starter * lattice[x][y][z][t].U[3];
            
        }
    
        temp[x][y][z] = 0.5*starter.trace().real();
    }

    return temp;
}




std::vector<long> potential_generator(std::vector<std::vector<std::vector<std::vector<Links>>>>& lattice) {
    int halfL = L / 2;
    std::vector<long> PP(halfL, 0.0);
    std::vector<long> num(halfL, 0.0);
    std::vector<std::vector<std::vector<std::complex<long>>>> ploops = Polyakov_loop(lattice);
    
    for (int i = 0; i < halfL; i++) {
        for (int j = 0; j < halfL; j++) {
            for (int k = 0; k < halfL; k++) {
                for (int r = 0; r < halfL; r++) {
                    int ip = (i + r) % L;
                    int jp = (j + r) % L;
                    int kp = (k + r) % L;
                    
                    std::complex<long> P0 = ploops[i][j][k];
               
                    PP[r] += (P0*(std::conj(ploops[ip][j][k])) +P0*(std::conj(ploops[i][jp][k])) +P0*(std::conj(ploops[i][j][kp]))).real();
                    num[r] += 3.0;  
                }
            }
        }
        
    }
    for (int r = 0; r < halfL; r++) {
        if (num[r] != 0.0) {
            PP[r] /= num[r];
        }
        dist_holder[r] = r;
        
    }

    return PP;
}








std::vector<double> potential_holder(L/2,0.0);
std::vector<double> potential_holder_2(L/2,0.0);



int main(){
   font.loadFromFile("/home/seaborne/Desktop/C++/W95FA.ttf");
 
    init_paulis();

    

    std::cout << "initializing" << std::endl;
    beta = 2.7;
    cold_start(lattice);
    

    std::cout << "cold start done" << std::endl;
    
     full_sweep(lattice,500,beta);

    for(int i =0; i < N_trials_per; i++) {
        std::vector<long> holder = potential_generator(lattice);
        for(int j = 0;j < L/2; j ++){
        if(holder[j] < 1e-24){
            polyakov_holder[j][i] = 0;
        }else{  
        polyakov_holder[j][i] = -1*std::log(holder[j]);
        }
        }
        full_sweep(lattice,1,beta);
    }
        for(int j = 0;j < L/2; j ++){
            potential_holder[j] = (jackKnife(polyakov_holder[j]));
        }

for (auto& row : polyakov_holder)
    std::fill(row.begin(), row.end(), 0.0);

    
    std::cout << "initializing" << std::endl;
    beta = 2.0;
    cold_start(lattice);
    

    std::cout << "cold start done" << std::endl;
    //heatbath_evolve_single({0,0,0,0},lattice,0);
    //    return 0;

     full_sweep(lattice,500,beta);
    //return 0;


    for(int i =0; i < N_trials_per; i++) {
        std::vector<long> holder = potential_generator(lattice);
        for(int j = 0;j < L/2; j ++){
       
        polyakov_holder[j][i] = -1*std::log(holder[j]);
        
        }
        full_sweep(lattice,1,beta);
    }
        for(int j = 0;j < L/2; j ++){
            potential_holder_2[j] = (jackKnife(polyakov_holder[j]));
        }



           Camera2D mycam = {0,-5,50,50};

sf::RenderWindow window2(sf::VideoMode(window_size, window_size), "quark-antiquark potential");
    while (window2.isOpen()) {
    sf::Event event;
    while (window2.pollEvent(event)) {
        if (event.type == sf::Event::Closed){
            window2.close();
        }

    }
        mycam = zuper_keyboard_input(1,1.1,mycam);

    
    sf::VertexArray myline = zuper_line(dist_holder,potential_holder,sf::Color(0,0,255,255),mycam,window_size,true);
    sf::VertexArray myline2 = zuper_line(dist_holder,potential_holder_2,sf::Color(255,0,0,255),mycam,window_size,true);
    

    window2.clear(sf::Color::White);

    std::vector<sf::VertexArray> z_grid = zuper_grid(8,window_size,sf::Color(155,155,155,255),mycam);
    for(sf::VertexArray& r : z_grid){;
        window2.draw(r);
    }

    std::vector<sf::Text> z_axis = zuper_axis_labels(8,window_size,20,font,sf::Color::Black,mycam);
    for(sf::Text& r : z_axis){
        window2.draw(r);
    }

   // window2.draw(zuper_x_axis(sf::Color::Black,mycam,window_size));
  //  window2.draw(zuper_y_axis(sf::Color::Black,mycam,window_size));
    window2.draw(myline);
window2.draw(myline2);

    window2.display();
}




    for(int i =0; i<L/2; i++){
    std::cout<< "polyakov:"<<polyakov_holder[i][0] << std::endl;
    std::cout<< "distance:"<<dist_holder[i] << std::endl;

    }


    return 0;

}
