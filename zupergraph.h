#ifndef ZUPERGRAPH
#define ZUPERGRAPH

#include <vector>
#include <SFML/Graphics.hpp>
#include <sstream>
#include <iomanip>

#include<SFML/Audio.hpp>
#include <cmath>
#include <string>
struct Camera2D {
    double x;
    double y;
    double PPUX;
    double PPUY; 
};

int zuper_modulus(int a,int b);

std::string clean_number(double value, int precision = 2);

void generate_sound();

std::vector<sf::RectangleShape> zuper_heatmap(
    int window_size,
    const std::vector<std::vector<double>>& input_matrix
);

sf::VertexArray zuper_line(
    const std::vector<double>& x_axis,
    const std::vector<double>& y_axis,
    sf::Color line_color,
    const Camera2D& cam,int win_size,bool isthick
);

sf::VertexArray zuper_x_axis(sf::Color col,Camera2D cam,int win_size);
sf::VertexArray zuper_y_axis(sf::Color col,Camera2D cam,int win_size);

Camera2D zuper_keyboard_input(double speed,double zoomspeed,const Camera2D& cam);

sf::Text zuper_text(sf::Font font,std::string input_text,sf::Color col, double size, sf::Vector2f pos, Camera2D cam, int window_size);

std::vector<sf::VertexArray> zuper_grid(int n,int window_size,sf::Color col,Camera2D cam);

std::vector<sf::Text> zuper_axis_labels(int n, int window_size,int text_size, const sf::Font& font,sf::Color col,Camera2D cam);
double zuper_curve_fit(const std::vector<double>& data_set_y,const std::vector<double>& data_set_x);
#endif 

