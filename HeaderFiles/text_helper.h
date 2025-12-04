#ifndef TEXT_CLASS_H
#define TEXT_CLASS_H

#include <string>
#include <glad/glad.h>
#include <glm/glm.hpp>    
#include <ft2build.h>
#include FT_FREETYPE_H  
#include <map>  
#include <vector>
#include "shaderClass.h"  



class Character {
    public:
        GLuint     TextureID;  
        glm::ivec2 Size;       
        glm::ivec2 Bearing;    
        GLuint     Advance;    
};

class fixed_variable {//type means either flat rate of change or exponential
    public:
        std::string name;
        double* value;
        int type;
    fixed_variable(const std::string& name, double* value, int type): name(name), value(value), type(type) {}
};

class variable_set  {
    public:
        std::vector<fixed_variable> myset;
        fixed_variable current_guy;
        int index;
    
    void scrollup(){
        current_guy = myset[(index+1) % myset.size()];
        index = (index+1) % myset.size();
    }
    void scrolldown(){
        if(index ==0){
            current_guy = myset.back();
            index = myset.size();
        }else {
                   index -=1;
            current_guy =myset[index];
     
        }
    }

    variable_set(const std::vector<fixed_variable>& myset,const fixed_variable& curr_guy): myset(myset), current_guy(curr_guy), index(0) {}

};

extern std::map<GLchar, Character> Characters;


void text_initialize(FT_Face face);
void RenderText(Shader &shader,fixed_variable to_display, float x, float y, float scale, glm::vec3 color,unsigned int VBO,unsigned int VAO);

#endif
