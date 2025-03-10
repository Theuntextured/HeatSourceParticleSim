#pragma once
#include "CommonIncludes.cuh"

template <typename T>
class GlobalVariable
{
public:
	//CONSTRUCTORS

	explicit __host__ GlobalVariable(const T& InValue) {
		HostStorage = malloc(sizeof(T));
		memcpy(HostStorage, &InValue, sizeof(T));
		cudaMalloc(&DeviceStorage, sizeof(T));
		cudaMemcpy(DeviceStorage, &HostStorage, sizeof(T), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
	}
	explicit __host__ GlobalVariable() {
		HostStorage = malloc(sizeof(T));
		*HostStorage = T();

		cudaMalloc(&DeviceStorage, sizeof(T));
		cudaMemcpy(DeviceStorage, HostStorage, sizeof(T), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
	}
	//copy
	__host__ GlobalVariable(const GlobalVariable<T>& Other) {
		HostStorage = malloc(sizeof(T));
		memcpy(HostStorage, Other.HostStorage, sizeof(T));
		cudaMalloc(&DeviceStorage, sizeof(T));
		cudaMemcpy(DeviceStorage, &HostStorage, sizeof(T), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
	}
	//move
	__host__ GlobalVariable(const GlobalVariable<T>&& Other) {
		HostStorage = Other->HostStorage;
		DeviceStorage = Other.DeviceStorage;

		Other->HostStorage = nullptr;
		Other->DeviceStorage = nullptr;
	}
	__host__ ~GlobalVariable() {
		free(HostStorage);
		cudaFree(DeviceStorage);
		cudaDeviceSynchronize();
	}

	//CONVERSIONS

	__host__ __device__  operator const T&() const {
		DEVICE_SWITCH(return *DeviceStorage;, return *HostStorage;)
	}

	//STORE AND LOAD

	__host__ __device__ void Store(const T& InValue) {
		DEVICE_SWITCH(
			*DeviceStorage = InValue;
		,
			*HostStorage = InValue;
			cudaMemcpy(DeviceStorage, HostStorage, cudaMemcpyHostToDevice);
		)
	}
	__host__ __device__ const T& Load() {
		return *this;
	}

	//SYNC

	__host__ void SyncWithDevice() {
		cudaMemcpy(HostStorage, DeviceStorage, sizeof(T), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
	}
	

private:
	T* HostStorage;
	T* DeviceStorage;
};

template <typename T>
class GlobalArray {
	explicit __host__ GlobalArray() = default;
	explicit __host__ GlobalArray(const size_t Size) {
		Num = Size;
		Capacity = Size;
		HostData = calloc(Size, sizeof(T));
		cudaMalloc(&DeviceData, Size * sizeof(T));
		cudaDeviceSynchronize();
	}
	__host__ GlobalArray(const T Arr[]) {
		Num = std::size(Arr);
		Capacity = Num;
		HostData = malloc(sizeof(T) * Num);
		memcpy(HostData, Arr, sizeof(T) * Num);
		cudaMalloc(&DeviceData, Num * sizeof(T));
		cudaMemcpy(DeviceData, HostData, Num * sizeof(T));
		cudaDeviceSynchronize();
	}
	__host__ void SetNum(const size_t NewNum);
	std::vector::back;
	std::vector::capacity;
	std::vector::clear;
	std::vector::data;
	std::vector::emplace;
	std::vector::emplace_back;
	std::vector::empty;
	std::vector::erase;
	std::vector::front;
	std::vector::insert;
	std::vector::reserve;
	std::vector::size;
	std::vector::shrink_to_fit;
	std::vector::pop_back;
	std::vector::swap;


private:
	T* HostData = nullptr;
	T* DeviceData = nullptr;

	size_t Num = 0;
	size_t Capacity = 0;
};