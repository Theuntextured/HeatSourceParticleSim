#pragma once
#include "Settings.cuh"
#include "Particle.cuh"
#include "WindowManager.cuh"

class Engine {
public:
	Engine();
	void Tick();

	WindowManager* wm;
	float dt;
private:
	int* d_CellParticleCount;
	int* d_CellParticleIDs;
	Particle* d_Particles;

	sf::Clock TickClock;
	cudaError_t err;
};

__global__ void ResetGrid(int* GridStart);
__global__ void SetupGrid(int* g, int* gi, Particle* p);