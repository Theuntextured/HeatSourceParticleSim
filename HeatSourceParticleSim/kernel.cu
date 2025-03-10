#pragma once
#include "CommonIncludes.cuh"
#include "GlobalVariable.cuh"

int main() {
	srand(static_cast <unsigned> (time(0)));
	cudaSetDevice(0);
	const auto p = GlobalVariable<int>(3);
}