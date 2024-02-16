#include "WindowManager.cuh"

WindowManager::WindowManager()
{
	window.create(sf::VideoMode(WindowWidth, WindowHeight), "Temperature Particle Simulation", sf::Style::Close);
	pixels = new sf::Uint8[4 * WindowHeight * WindowWidth];
	err = cudaMalloc(&d_pixels, sizeof(sf::Uint8) * 4 * WindowHeight * WindowWidth);
	ERRORCHECK;
	cudaDeviceSynchronize();
	RenderTexture.create(WindowWidth, WindowHeight);
	RenderTexture.setSmooth(true);
}

bool WindowManager::Tick(float dt)
{
	sf::String title("Temperature Particle simulation ");
	title += std::to_string((int)(1 / dt));
	title += " fps";
	
	window.setTitle(title);
	//manage closing window
	if (!window.isOpen()) return false;
	sf::Event event;
	while (window.pollEvent(event)) {
		if (event.type == sf::Event::Closed) {
			window.close();
			return false;
		}
	}
	//draw
	cudaMemcpy(pixels, d_pixels, sizeof(sf::Uint8) * 4 * WindowHeight * WindowWidth, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	ERRORCHECKLAST;

	RenderTexture.update(pixels);
	RenderSprite.setTexture(RenderTexture);
	window.clear();
	window.draw(RenderSprite);
	window.display();

	ClearBuffer << < (WindowHeight * WindowWidth + 1023) / 1024, 1024 >> > (d_pixels, dt);
	cudaDeviceSynchronize();
	ERRORCHECKLAST;

	return true;
}


__global__ void ClearBuffer(sf::Uint8* buffer, float dt)
{
	GETID(WindowHeight * WindowWidth);

#if WindowClearMethod == 0

	float sum;
	int items;
	int ox = id % WindowWidth;
	int oy = id / WindowWidth;

	for (int n = 0; n < 4; n++) {
		sum = 0;
		items = 0;
		for (int x = ox - 1; x < ox + 2; x++) {
			for (int y = oy - 1; y < oy + 2; y++) {
				if (x >= 0 && y >= 0 && x < WindowWidth && y < WindowHeight && !(y == 0 && x == 0)) {
					sum += buffer[(x + y * WindowWidth) * 4 + n];
					items++;
				}
			}
		}
		buffer[id * 4 + n] = sum / items;
	}
	return;
#elif WindowClearMethod == 1
	id *= 4;
	buffer[id + 0] *= pow(0.5, dt);
	buffer[id + 1] *= pow(0.5, dt);
	buffer[id + 2] *= pow(0.5, dt);
	buffer[id + 3] *= pow(0.5, dt);
	return;
#endif
	id *= 4;
	buffer[id + 0] = 0;
	buffer[id + 1] = 0;
	buffer[id + 2] = 0;
	buffer[id + 3] = 0;
}
