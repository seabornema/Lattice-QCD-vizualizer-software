#include "zupergraph.h"




int zuper_modulus(int a,int b) {
    int temp = a % b;
    if(temp < 0) {
        return temp+b;
    }else {
        return temp;
    }
}
std::string clean_number(double value, int precision) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << value;
    std::string s = out.str();

    s.erase(s.find_last_not_of('0') + 1, std::string::npos);
    if (!s.empty() && s.back() == '.') s.pop_back();

    return s;
}

void generate_sound() {
    static sf::Sound sound_global;  
    sf::SoundBuffer buffer;

    const unsigned sampleRate = 44100;
    const double duration = 0.1;
    const double frequency = 200 + rand() % 1800;

    std::vector<sf::Int16> samples(sampleRate * duration);
    for (unsigned i = 0; i < samples.size(); ++i) {
        double t = static_cast<double>(i) / sampleRate;
        samples[i] = 30000 * std::sin(2 * M_PI * frequency * t);
    }

    buffer.loadFromSamples(samples.data(), samples.size(), 1, sampleRate);

    sound_global.setBuffer(buffer);
    sound_global.play();
}


//version 0.3

// for some matrix, matrix[i][j], i will designate the row where j is the collumm, left to right, top to bottom
std::vector<sf::RectangleShape> zuper_heatmap(int window_size, const std::vector<std::vector<double>>& input_matrix) {
    double max = -1e18;
    for(std::vector<double> i : input_matrix) {
        double currmax = *std::max_element(i.begin(),i.end()) ;
        if(currmax > max){
            max = currmax;
        }
    }
    int L = input_matrix.size();
    std::vector<sf::RectangleShape> storage;
    double scaling_factor = window_size/L;
    for (int row = 0; row < L; row++) {
    for (int col = 0; col < L; col++) {
        sf::RectangleShape curr(sf::Vector2f(scaling_factor, scaling_factor));

        curr.setPosition(sf::Vector2f(col * scaling_factor, row * scaling_factor));

        double value = input_matrix[row][col];
        sf::Uint8 alpha = static_cast<sf::Uint8>((255 * value) / max);
        curr.setFillColor(sf::Color(255, 255, 255, alpha));

        storage.push_back(curr);
    }
    }
    return storage;
}


sf::VertexArray zuper_line(const std::vector<double>& x_axis,const std::vector<double>& y_axis,sf::Color line_color,const Camera2D& cam,int win_size,bool isthick) {
    int n = x_axis.size();
    sf::VertexArray temp_line(sf::LineStrip, n);    
    if(isthick == true){
    temp_line = sf::VertexArray(sf::LineStrip, 2*n);
    }
    
    for(int i = 0; i < n; i++) {
        double screen_x = win_size / 2.0 + (cam.PPUX * (x_axis[i] - cam.x));
        double screen_y = win_size / 2.0 - (cam.PPUY * (y_axis[i] - cam.y));

        temp_line[i].position = sf::Vector2f(screen_x,screen_y);
        temp_line[i].color = line_color;
    }
    if(isthick == true){
     for(int i = n-1; i >= 0; i--) {
        double screen_x = win_size / 2.0 + (cam.PPUX * (x_axis[i] - cam.x));
        double screen_y = win_size / 2.0 - (cam.PPUY * (y_axis[i] - cam.y)) - 1;

        temp_line[2*n-i -1].position = sf::Vector2f(screen_x,screen_y);
        temp_line[2*n-i -1].color = line_color;
    }
}
    return temp_line;

}


sf::VertexArray zuper_x_axis(sf::Color col,Camera2D cam,int win_size){
    std::vector<double> tempx = {-1000000,1000000};
    std::vector<double> tempy = {0,0};
    return zuper_line(tempx,tempy,col,cam,win_size,false);
}
sf::VertexArray zuper_y_axis(sf::Color col,Camera2D cam,int win_size){
    std::vector<double> tempy = {-1000000,1000000};
    std::vector<double> tempx = {0,0};
    return zuper_line(tempx,tempy,col,cam,win_size,false);
}
        
Camera2D zuper_keyboard_input(double speed, double zoomspeed, const Camera2D& cam) {
    static sf::Sound sound;          
    static sf::SoundBuffer buffer; 

    Camera2D tempcam = cam;
    bool key_pressed= false;
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::LShift)) {
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left)) { tempcam.x -= speed;key_pressed=true;}
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right)) {tempcam.x += speed;key_pressed=true;}
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up))    {tempcam.y += speed;key_pressed=true;}
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down))  {tempcam.y -= speed;key_pressed=true;}

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Z)) {tempcam.PPUX *= zoomspeed;key_pressed=true;}
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::X)) {tempcam.PPUX /= zoomspeed;key_pressed=true;}
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Q)) {tempcam.PPUY *= zoomspeed;key_pressed=true;}
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::W)) {tempcam.PPUY /= zoomspeed;key_pressed=true;}
    } else {
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left))  {tempcam.x -= speed / tempcam.PPUX;key_pressed=true;}
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right)) {tempcam.x += speed / tempcam.PPUX;key_pressed=true;}
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up))    {tempcam.y += speed / tempcam.PPUY;key_pressed=true;}
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down))  {tempcam.y -= speed / tempcam.PPUY;key_pressed=true;}

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Z)) {tempcam.PPUX += zoomspeed / tempcam.PPUX;key_pressed=true;}
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::X)) {tempcam.PPUX -= zoomspeed / tempcam.PPUX;key_pressed=true;}
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Q)) {tempcam.PPUY += zoomspeed / tempcam.PPUY;key_pressed=true;}
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::W)) {tempcam.PPUY -= zoomspeed / tempcam.PPUY;key_pressed=true;}
    }

    if (key_pressed== true) {
        const unsigned sampleRate = 44100;
        const double duration = 0.1; 
        const double freq = 200 + rand() % 1800;

        std::vector<sf::Int16> samples(sampleRate * duration);
        for (unsigned i = 0; i < samples.size(); ++i) {
            double t = static_cast<double>(i) / sampleRate;
            samples[i] = 30000 * std::sin(2 * M_PI * freq * t);
        }

        buffer.loadFromSamples(samples.data(), samples.size(), 1, sampleRate);
        sound.setBuffer(buffer);
        sound.play();
    }

    return tempcam;
}



sf::Text zuper_text(sf::Font font,std::string input_text,sf::Color col, double size, sf::Vector2f pos, Camera2D cam, int window_size) {

    sf::Text temp;
    temp.setFont(font);
    temp.setFillColor(col);
    temp.setString(input_text);

    float screen_x = window_size / 2.0 + (cam.PPUX * (pos.x - cam.x));
    float screen_y = window_size / 2.0 - (cam.PPUY * (pos.y - cam.y));

    temp.setPosition({screen_x,screen_y});
    temp.setCharacterSize(size);
    return temp;
}

std::vector<sf::VertexArray> zuper_grid(int n,int window_size,sf::Color col,Camera2D cam) {
    std::vector<sf::VertexArray> temp;
    double cell_size = window_size/n;
    for(int i=0; i<n; i++) {
    sf::VertexArray temp_line(sf::Lines,2);
    double xpos =(i-n/2)*cell_size - cam.x*cam.PPUX + window_size/2;
    temp_line[0].position = sf::Vector2f(zuper_modulus(static_cast<int>(std::round(xpos)),window_size),window_size);
    temp_line[1].position = sf::Vector2f(zuper_modulus(static_cast<int>(std::round(xpos)),window_size),-window_size);
    temp_line[0].color = col;
    temp_line[1].color = col;
    temp.push_back(temp_line);
    }
    for(int i=0; i<n; i++) {
    sf::VertexArray temp_line(sf::Lines,2);
    double ypos =(i-n/2)*cell_size + cam.y*cam.PPUY + window_size/2;
    temp_line[0].position = sf::Vector2f(window_size,zuper_modulus(static_cast<int>(std::round(ypos)),window_size));
    temp_line[1].position = sf::Vector2f(-window_size,zuper_modulus(static_cast<int>(std::round(ypos)),window_size));
    temp_line[0].color = col;
    temp_line[1].color = col;
    temp.push_back(temp_line);
    }
    return temp;
}

std::vector<sf::Text> zuper_axis_labels(int n, int window_size,int text_size, const sf::Font& font,sf::Color col,Camera2D cam) {
    std::vector<sf::Text> temp;
    double cell_size = window_size/n;
    for(int i=0; i<n; i++) {
    
    float xpos =(i-n/2)*cell_size - cam.x*cam.PPUX + window_size/2;
    if(std::abs(xpos) < .02) continue;
    xpos = zuper_modulus(xpos,window_size);
    sf::Text temp_text;
    temp_text.setFont(font);
    temp_text.setFillColor(col);
    double num = (xpos/cam.PPUX + cam.x - window_size/(2*cam.PPUX));
    
    temp_text.setString(clean_number(num,2));
    temp_text.setPosition(xpos, window_size/2 + cam.y*cam.PPUY);
    temp_text.setCharacterSize(text_size);

    temp.push_back(temp_text);
    }
    for(int i=0; i<n; i++) {
    double ypos =(i-n/2)*cell_size + cam.y*cam.PPUY + window_size/2;
    ypos = zuper_modulus(ypos,window_size);
    sf::Text temp_text;
    temp_text.setFont(font);
    temp_text.setFillColor(col);
    temp_text.setString(clean_number((-ypos/cam.PPUY + cam.y + window_size/(2*cam.PPUY)),2));
    temp_text.setPosition(window_size/2 - cam.x*cam.PPUX,ypos);
    temp_text.setCharacterSize(text_size);

    temp.push_back(temp_text);
    }
    return temp;
}


double zuper_curve_fit(const std::vector<double>& data_set_y,const std::vector<double>& data_set_x) {

    double num = 0.0; 
    double den = 0.0; 
    
    for (size_t i = 0; i < data_set_y.size(); i++) {
        num += data_set_x[i] * data_set_y[i];
        den += data_set_x[i] * data_set_x[i];
    }

    return num / den;
}

