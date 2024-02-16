#include "Engine.cuh"

Engine::Engine()
{
	Particle p[ParticleCount];

	cudaMalloc(&d_CellParticleCount, sizeof(int) * GridWidth * GridHeight);
	cudaMalloc(&d_CellParticleIDs, sizeof(int) * GridWidth * GridHeight * MaxParticlesPerCell);
	cudaMalloc(&d_Particles, sizeof(Particle) * ParticleCount);
	cudaMemcpy(d_Particles, p, sizeof(Particle) * ParticleCount, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	wm = new WindowManager();

	TickClock.restart();
	dt = 1;
}

void Engine::Tick()
{
	dt = TickClock.restart().asSeconds();
	if (dt >= 0.01) return;
	//ResetGrid << <(GridCount + 1023) / 1024, 1024 >> > (d_CellParticleCount);
	//SetupGrid << <(ParticleCount + 1023) / 1024, 1024 >> > (d_CellParticleCount, d_CellParticleIDs, d_Particles);
	ProcessParticle << < (ParticleCount + 1023) / 1024, 1024 >> > (d_Particles, dt);
	ERRORCHECKLAST;
	cudaDeviceSynchronize();
	ProcessParticleVelocity << < (ParticleCount + 1023) / 1024, 1024 >> > (d_Particles, dt);
	ERRORCHECKLAST;
	DrawParticle << < (ParticleCount + 1023) / 1024, 1024 >> > (d_Particles, wm->d_pixels);
	ERRORCHECKLAST;
}

__global__ void ResetGrid(int* GridStart)
{
	GETID(GridCount);
	GridStart[id] = 0;
}

__global__ void SetupGrid(int* g, int* gi, Particle* p)
{
	GETID(ParticleCount);
	Particle* pa = &p[id];
	int x = pa->Location.x / WindowWidth * GridWidth;
	int y = pa->Location.y / WindowHeight * GridHeight;
	int c = GridWidth * y + x;
	pa->Cell = c;
	if (g[c] >= MaxParticlesPerCell) return;
	gi[c * MaxParticlesPerCell + g[c]] = id;
	++g[c];
}