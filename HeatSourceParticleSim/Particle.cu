#include "Particle.cuh"

Particle::Particle()
{
	const float v = 1000.0f;
	Location = sf::Vector2f(RAND(ParticleRadius, WindowWidth - ParticleRadius), RAND(ParticleRadius, WindowHeight - ParticleRadius));
	Velocity = sf::Vector2f(RAND(-v, v), RAND(-v, v));
	dvx = 0;
	dvy = 0;
}

__global__ void ProcessParticle(Particle* p, float dt)
{
	GETID(ParticleCount);

	//setup
	float ax = 0;
	float ay = 0;


	//Handle dv
	p[id].Velocity.x += ax * dt;
	p[id].Velocity.y += ay * dt;

	//handle borders
	if (p[id].Location.x - ParticleRadius<= 0 && p[id].Velocity.x < 0) p[id].Velocity.x *= -BorderCollisionElasticity;
	if (p[id].Location.x + ParticleRadius >= WindowWidth && p[id].Velocity.x > 0) p[id].Velocity.x *= -BorderCollisionElasticity;
	if (p[id].Location.y + ParticleRadius >= WindowHeight && p[id].Velocity.y > 0) p[id].Velocity.y *= -BaseCollisionElasticity;
	if (p[id].Location.y - ParticleRadius <= 0 && p[id].Velocity.y < 0) p[id].Velocity.y *= -BorderCollisionElasticity;
	CLAMP(p[id].Location.x, ParticleRadius, WindowWidth - ParticleRadius);
	CLAMP(p[id].Location.y, ParticleRadius, WindowHeight - ParticleRadius);
	
	float npx = p[id].Location.x;
	float npy = p[id].Location.y;

	//handle collision
	for (int i = 0; i < ParticleCount; i++) {
		if (i == id) continue;
		if (((p[i].Location.x - p[id].Location.x) * (p[i].Location.x - p[id].Location.x) + (p[i].Location.y - p[id].Location.y) * (p[i].Location.y - p[id].Location.y)) <= (ParticleRadius * ParticleRadius * 4)) {
			float dist = sqrt((p[i].Location.x - p[id].Location.x) * (p[i].Location.x - p[id].Location.x) + (p[i].Location.y - p[id].Location.y) * (p[i].Location.y - p[id].Location.y));
			float dxn = (p[id].Location.x - p[i].Location.x) / dist;
			float dyn = (p[id].Location.y - p[i].Location.y) / dist;

			float dva = abs(dxn * p[i].Velocity.x + dyn * p[i].Velocity.y); //dot product
			p[id].dvx += dxn * dva;
			p[id].dvy += dyn * dva;
			//printf("%f\n", dva);

			npx += 0.5 * (ParticleRadius * 2 - dist) * dxn;
			npy += 0.5 * (ParticleRadius * 2 - dist) * dyn;
		}
	}

	p[id].Location.x = npx;
	p[id].Location.y = npy;

	p[id].Velocity = 

	//handle gravity
	ay += Gravity * dt;
}

__global__ void ProcessParticleVelocity(Particle* p, float dt)
{
	GETID(ParticleCount);

	//p[id].Velocity.x += p[id].dvx;
	//p[id].Velocity.y += p[id].dvy;

	//handle dx
	p[id].Location.x += p[id].Velocity.x * dt;
	p[id].Location.y += p[id].Velocity.y * dt;

	p[id].dvx = 0;
	p[id].dvy = 0;
}

__global__ void DrawParticle(Particle* p, sf::Uint8* pixels)
{
	GETID(ParticleCount);
	int pos;
	for (int x = p[id].Location.x - ParticleRadius; x < p[id].Location.x + ParticleRadius; x++) {
		for (int y = p[id].Location.y - ParticleRadius; y < p[id].Location.y + ParticleRadius; y++) {
			if (x >= 0 && y >= 0 && x < WindowWidth && y < WindowHeight && (ParticleRadius * ParticleRadius) >= (y - p[id].Location.y) * (y - p[id].Location.y) + (x - p[id].Location.x) * (x - p[id].Location.x)) {
				pos = (WindowWidth * y + x) * 4;
				pixels[pos + 0] = 255;
				pixels[pos + 1] = 255;
				pixels[pos + 2] = 255;
				pixels[pos + 3] = 255;
			}
		}
	}
}