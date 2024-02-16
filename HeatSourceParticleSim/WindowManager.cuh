#pragma once
#include "Settings.cuh"

class WindowManager {
public:
	WindowManager();

	//return value is false if the window has been closed. True otherwise.
	bool Tick(float dt);


	sf::Uint8* pixels;
	sf::Uint8* d_pixels;
private:
	sf::RenderWindow window;
	sf::Texture RenderTexture;
	sf::Sprite RenderSprite;
	cudaError_t err;
};

__global__ void ClearBuffer(sf::Uint8* buffer, float dt);

