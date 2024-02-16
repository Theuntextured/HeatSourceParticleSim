#pragma once
#include <cuda_runtime.h>
#include <math.h>
#include "device_launch_parameters.h"
#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
//COMMON LIB
#define GETID(a) int id = blockIdx.x * blockDim.x + threadIdx.x; \
		if(id >= a) return
#define GridCount GridWidth * GridHeight
#define ERRORCHECKLAST	err = cudaGetLastError();\
					if ( err != cudaSuccess ) {\
						printf("CUDA Error: %s\n", cudaGetErrorString(err));\
						throw;\
					}
#define ERRORCHECK 	if ( err != cudaSuccess ) {\
						printf("CUDA Error: %s\n", cudaGetErrorString(err));\
						throw;\
					}

#define RAND(LO, HI) (LO + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (HI - LO))))
#define MIN(A, B) (A > B ? B : A)
#define MAX(A, B) (A > B ? A : B)
#define CLAMP(VALUE, LO, HI) VALUE = MIN(MAX(LO, VALUE), HI)

//SETTINGS START

#define WindowWidth 1600
#define WindowHeight 900

#define WindowClearMethod 2

#define GridWidth 120
#define GridHeight 90

#define ParticleCount 32
#define ParticleRadius 32

#define Gravity 1000

#define MaxParticlesPerCell 5

#define BorderCollisionElasticity 1
#define BaseCollisionElasticity 0.01

