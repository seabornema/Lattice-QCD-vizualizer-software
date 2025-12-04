#version 330 core

out vec4 FragColor;


in vec3 color;
in vec3 fadeColor;
in vec3 Normal;
in vec3 crntPos;

uniform vec4 lightColor;
uniform vec3 lightPos;

uniform vec3 camPos;
uniform vec3 camNorm;

void main()
{

	float ambient = 0.40f;

	vec3 normal = normalize(Normal);
	vec3 lightDirection = normalize(-lightPos + crntPos);
	float diffuse = max(dot(normal, lightDirection), 0.0f);

	float blend = clamp(dot(camNorm, normal), 0.0, 1.0);
	vec3 truecolor = mix(color, fadeColor, 1-blend);

	float specularLight = 0.50f;
	vec3 viewDirection = normalize(camPos - crntPos);
	vec3 reflectionDirection = reflect(-lightDirection, normal);
	float specAmount = pow(max(dot(viewDirection, reflectionDirection), 0.0f), 8);
	float specular = specAmount * specularLight;

	FragColor = vec4( truecolor * (diffuse + ambient + specular),1.0f);
}