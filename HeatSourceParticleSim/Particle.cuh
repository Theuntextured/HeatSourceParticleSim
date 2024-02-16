#pragma once
#include "Settings.cuh"

class Particle {
public:
	Particle();

	sf::Vector2f Location;
	sf::Vector2f Velocity;
	int Cell;

	float dvx;
	float dvy;
};

__global__ void ProcessParticle(Particle* p, float dt);

__global__ void ProcessParticleVelocity(Particle* p, float dt);

__global__ void DrawParticle(Particle* p, sf::Uint8* pixels);