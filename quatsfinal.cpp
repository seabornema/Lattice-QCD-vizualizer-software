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
const int Nt = 8;
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
    std::vector<std::array<double,4>> Q;
};

std::array<double,4> prod(std::array<double,4> A,std::array<double,4> B){
    double x0 = A[0]*B[0] - (A[1]*B[1] + A[2]*B[2] + A[3]*B[3]);
    double a = A[0]*B[1] + B[0]*A[1] - (A[2]*B[3] - A[3]*B[2]);
    double b = A[0]*B[2] + B[0]*A[2] - (A[3]*B[1] - A[1]*B[3]);
    double c = A[0]*B[3] + B[0]*A[3] - (A[1]*B[2] - A[2]*B[1]);

    return {x0,a,b,c};
}

std::array<double,4> scalardiv(std::array<double,4> A, double B){
    return {A[0]/B,A[1]/B,A[2]/B,A[3]/B};
}

std::array<double,4> vec_sum(std::array<double,4> A, std::array<double,4> B) {
    return {A[0]+B[0],A[1]+B[1],A[2]+B[2],A[3]+B[3]};
}
const int N_trials_per = 5000;
int sweep_number = 0;
std::vector<double> dist_holder(L/2,0);
std::vector<std::vector<double>> polyakov_holder(L/2,std::vector<double>(N_trials_per,0.0));

double beta = 2.7;
const int window_size = 600;

sf::Font font;

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
    double rsq = 10;
    double lambda2 = 1;
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

std::array<double,4> dagger(std::array<double,4> A){
    return {A[0],-A[1],-A[2],-A[3]};
}

double det(std::array<double,4> inpu) {
    return ((inpu[0]*inpu[0])+(inpu[1]*inpu[1])+(inpu[2]*inpu[2])+(inpu[3]*inpu[3]));
}



Eigen::Matrix2cd SU2_constructor(const std::array<double,4>& xvec){
    return xvec[0]*Eigen::Matrix2cd::Identity() + xvec[1]*I*sigma_x + xvec[2]*I*sigma_y + xvec[3]*I*sigma_z;
}




std::array<double,4> staples(const std::array<int,4>& position,const std::vector<std::vector<std::vector<std::vector<Links>>>>& lattice,int mu){
    int x = position[0];
    int y = position[1];
    int z = position[2];
    int t = position[3];
    std::array<double,4> sum = {0.0,0.0,0.0,0.0};
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
                
                std::array<double,4> U1 = lattice[tmod(x+U_direction[0],L)][tmod(y+U_direction[1],L)][tmod(z+U_direction[2],L)][tmod(t+U_direction[3],Nt)].Q[j];
                std::array<double,4> U2 = dagger(lattice[tmod(x+V_direction[0],L)][tmod(y+V_direction[1],L)][tmod(z+V_direction[2],L)][tmod(t+V_direction[3],Nt)].Q[mu]);
                std::array<double,4> U3 = dagger(lattice[x][y][z][t].Q[j]);
               
                std::array<double,4> U4 = dagger(lattice[tmod(x+U_direction[0]-V_direction[0],L)][tmod(y+U_direction[1]-V_direction[1],L)][tmod(z+U_direction[2]-V_direction[2],L)][tmod(t+U_direction[3]-V_direction[3],Nt)].Q[j]);
                std::array<double,4> U5 = dagger(lattice[tmod(x-V_direction[0],L)][tmod(y-V_direction[1],L)][tmod(z-V_direction[2],L)][tmod(t-V_direction[3],Nt)].Q[mu]);
                std::array<double,4> U6 = lattice[tmod(x-V_direction[0],L)][tmod(y-V_direction[1],L)][tmod(z-V_direction[2],L)][tmod(t-V_direction[3],Nt)].Q[j];


auto term1 = prod(prod(U1, U2), U3);
auto term2 = prod(prod(U4, U5), U6);
sum = vec_sum(sum, vec_sum(term1, term2));

        }
    }
    return sum;
}

std::array<double,4> sigma_matrix(const std::array<double,4>& staple)
{
    double det_h = (det(staple));
    double norm = std::sqrt(det_h);

    std::array<double,4> Sigma =scalardiv(staple,norm);

    return Sigma;
}
void action_minimizing(const std::array<int,4>& position,std::vector<std::vector<std::vector<std::vector<Links>>>>& lattice,int direction){
       lattice[position[0]][position[1]][position[2]][position[3]].Q[direction] = sigma_matrix(staples(position,lattice,direction));
}
void action_minimizing_sweep(std::vector<std::vector<std::vector<std::vector<Links>>>>& lattice) {
        for(int i = 0; i < L*L*L*Nt*4; i++){
            int dir = i % 4;
            int x = (i / 4) % L;
            int y = (i / (4*L)) % L;
            int z = (i / (4*L*L)) % L;
            int t = (i / (4*L*L*L)) % Nt;
            action_minimizing({x,y,z,t}, lattice, dir);
        }
}
double action_single(std::vector<std::vector<std::vector<std::vector<Links>>>>& lattice, int x,int y,int z,int t){
    double sum =0;
    
    for(int mu = 0; mu < 4; mu ++){
        std::array<int,4> mu_d = {0,0,0,0};
        mu_d[mu]= 1;
        for(int nu =0; nu<mu;nu++){
              std::array<int,4> nu_d = {0,0,0,0};
              nu_d[nu]= 1;
            sum +=  prod(prod(lattice[x][y][z][t].Q[mu],lattice[tmod(x+mu_d[0],L)][tmod(y+mu_d[1],L)][tmod(z+mu_d[2],L)][tmod(t+mu_d[3],Nt)].Q[nu]),prod(dagger(lattice[tmod(x+nu_d[0],L)][tmod(y+nu_d[1],L)][tmod(z+nu_d[2],L)][tmod(t+nu_d[3],Nt)].Q[mu]),dagger(lattice[x][y][z][t].Q[nu])))[0];
        }
    }
    return sum;
}
double sum_action(
    std::vector<std::vector<std::vector<std::vector<Links>>>>& lattice)
{
    double sum = 0.0;
    for (int t = 0; t < Nt; ++t)
    for (int z = 0; z < L;  ++z)
    for (int y = 0; y < L;  ++y)
    for (int x = 0; x < L;  ++x)
        sum += action_single(lattice, x, y, z, t);

    return sum/(L*L*L*Nt);
}


double jackKnife(const std::vector<double>& data) {
    int N = data.size();
    double full_sum = std::accumulate(data.begin(), data.end(), 0.0);

    std::vector<double> jk(N);
    jk.reserve(N);

    for (int i = 0; i < N; i++) {
        jk[i] = (full_sum - data[i]) / (N - 1);
    }

    double mean = std::accumulate(jk.begin(), jk.end(), 0.0) / N;

    for (int i = 0; i < N; i++) {
        double diff = jk[i] - mean;
    }

    return mean;
}

void cold_start(std::vector<std::vector<std::vector<std::vector<Links>>>>& lattice)
{
    for (int x = 0; x < L; ++x){
        for (int y = 0; y < L; ++y){
            for (int z = 0; z < L; ++z){
                for (int t = 0; t < Nt; ++t)
                {
                     lattice[x][y][z][t].Q.resize(4); 
                    for (int mu = 0; mu < 4; ++mu){
                        lattice[x][y][z][t].Q[mu] = {1.0,0.0,0.0,0.0};
                    }
                }
            }      
        }
    }

}

void heatbath_evolve_single(const std::array<int,4>& position,std::vector<std::vector<std::vector<std::vector<Links>>>>& lattice,int direction,double beta){
    std::array<double,4> staps = staples(position,lattice,direction);
    double cval = std::sqrt((det(staps)));
   
    std::array<double,4> temp_guy = xvec_generator(x0_generator(cval,beta));
   
    std::array<double,4> Ut = scalardiv(prod(temp_guy,dagger(staps)),cval);

    lattice[position[0]][position[1]][position[2]][position[3]].Q[direction] = scalardiv(Ut,(std::sqrt(det(Ut))));    
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


std::vector<std::vector<std::vector<double>>> Polyakov_loop(const std::vector<std::vector<std::vector<std::vector<Links>>>>& lattice) {
    
    std::vector<std::vector<std::vector<double>>> temp(L, std::vector<std::vector<double>>(L, std::vector<double>(L,0.0)));
    for (int i = 0; i < L * L * L; i++) {
        int x = i % L;
        int y = (i / L) % L;
        int z = (i / (L * L)) % L;
        std::array<double,4> starter = lattice[x][y][z][0].Q[3];
        for (int t = 1; t < Nt; t++) {
            starter = prod(starter,lattice[x][y][z][t].Q[3]);
        }
    
        temp[x][y][z] = starter[0];
    }

    return temp;
}


std::vector<double> potential_generator(std::vector<std::vector<std::vector<std::vector<Links>>>>& lattice) {
    int halfL = L / 2;
    std::vector<double> PP(halfL, 0.0);
    std::vector<double> num(halfL, 0.0);
    std::vector<std::vector<std::vector<double>>> ploops = Polyakov_loop(lattice);
    
    for (int i = 0; i < halfL; i++) {
        for (int j = 0; j < halfL; j++) {
            for (int k = 0; k < halfL; k++) {
                for (int r = 0; r < halfL; r++) {
                    int ip = (i + r) % L;
                    int jp = (j + r) % L;
                    int kp = (k + r) % L;
                    
                    double P0 = ploops[i][j][k];
               
                    PP[r] += (P0*((ploops[ip][j][k])) +P0*((ploops[i][jp][k])) +P0*((ploops[i][j][kp])));
                    num[r] += 3.0;  
                }
            }
        }
        
    }

for (int r = 0; r < halfL; ++r) {
    if (num[r] > 0.0) PP[r] /= num[r];
    else PP[r] = 0.0;
}
return PP;

}



std::vector<double> potential_holder(L/2,0.0);
std::vector<double> potential_holder_2(L/2,0.0);



int main(){
   font.loadFromFile("/home/seaborne/Desktop/C++/W95FA.ttf");
    init_paulis();
for (int r = 0; r < L/2; ++r) dist_holder[r] = static_cast<double>(r);

    

    std::cout << "initializing" << std::endl;
    cold_start(lattice);
    

    std::cout << "cold start done" << std::endl;
    
/*
   
    for(int i =0; i < N_trials_per; i++) {
        std::vector<double> holder = potential_generator(lattice);
        for(int j = 0;j < L/2; j ++){
        polyakov_holder[j][i] = (holder[j]);
        }
        full_sweep(lattice,1,beta);
    }
        for(int j = 0;j < L/2; j ++){
            potential_holder[j] =-1*std::log(jackKnife(polyakov_holder[j]));
        }

for (auto& row : polyakov_holder)
    std::fill(row.begin(), row.end(), 0.0);

    
    std::cout << "initializing" << std::endl;
    beta = 2.2;
    cold_start(lattice);
    

    std::cout << "cold start done" << std::endl;
    //heatbath_evolve_single({0,0,0,0},lattice,0);
    //    return 0;

     full_sweep(lattice,500,beta);
    //return 0;


    for(int i =0; i < N_trials_per; i++) {
        std::vector<double> holder = potential_generator(lattice);
        for(int j = 0;j < L/2; j ++){
        polyakov_holder[j][i] = (holder[j]);
        }
        full_sweep(lattice,1,beta);
    }
        for(int j = 0;j < L/2; j ++){
            potential_holder_2[j] = -1*std::log(jackKnife(polyakov_holder[j]));
        }

        for(int i =0; i < N_trials_per; i++) {
        std::vector<double> holder = potential_generator(lattice);
        for(int j = 0;j < L/2; j ++){
        polyakov_holder[j][i] = (holder[j]);
        }
    }
*/
           Camera2D mycam = {10,-2,20,20};
   std::vector<double> x;
    std::vector<double> y;
           /*
std::vector<sf::VertexArray> lines_holder;
    std::vector<double> gurbis;
    std::vector<double> shmurbis;
 
    double bmax= 100;
    double binc =10;
    double b0=10;
    for(double b=b0;b<bmax; b+= binc){
    beta = b;
    full_sweep(lattice,500,beta);
    std::vector<double> gurbis;
    std::vector<double> shmurbis;
    for(int i =0; i< 100; i++){
    gurbis.push_back(i);
    shmurbis.push_back(sum_action(lattice));
    action_minimizing_sweep(lattice);
    }
 lines_holder.push_back(zuper_line(gurbis,shmurbis,sf::Color(255*(b/bmax),0,255*(1-(b/bmax)),255),mycam,window_size,true));
}
*/
  full_sweep(lattice,1,beta);
   beta =2.2;
for(int i =0; i< 100; i++){
    x.push_back(i);
    y.push_back(sum_action(lattice));
    full_sweep(lattice,1,beta);
    }

sf::RenderWindow window2(sf::VideoMode(window_size, window_size), "quark-antiquark potential");
    while (window2.isOpen()) {
    sf::Event event;
    while (window2.pollEvent(event)) {
        if (event.type == sf::Event::Closed){
            window2.close();
        }

    }
        mycam = zuper_keyboard_input(1,1.1,mycam);

    sf::VertexArray myline = zuper_line(x,y,sf::Color(0,0,255,255),mycam,window_size,true);
    
   //sf::VertexArray myline = zuper_line(dist_holder,potential_holder,sf::Color(0,0,255,255),mycam,window_size,true);
    //sf::VertexArray myline2 = zuper_line(dist_holder,potential_holder_2,sf::Color(255,0,0,255),mycam,window_size,true);
    

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
 /*   
    for(sf::VertexArray& v : lines_holder){
        window2.draw(v);
    }
 */
   window2.draw(myline);
//window2.draw(myline2);

    window2.display();
}



    for(int i =0; i<L/2; i++){
    std::cout<< "polyakov:"<<polyakov_holder[i][0] << std::endl;
    std::cout<< "distance:"<<dist_holder[i] << std::endl;

    }

    return 0;

}
