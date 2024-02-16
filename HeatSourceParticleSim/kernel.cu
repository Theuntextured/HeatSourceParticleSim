#include "Engine.cuh"
#include "WindowManager.cuh"
#include "Settings.cuh"
#include <ctime>
#include <cstdlib>

int main() {
	srand(static_cast <unsigned> (time(0)));
	cudaSetDevice(0);

	Engine engine;
	do {
		engine.Tick();
	} while (engine.wm->Tick(engine.dt));
}