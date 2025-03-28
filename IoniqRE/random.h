#pragma once

#include <random>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

class random
{
public:
	static void init();
	static void shutdown() {}

	__host__ static uint32_t uint(uint32_t min = 0, uint32_t max = UINT32_MAX);
	__host__ static float real(float min = 0.0f, float max = 1.0f);

	__device__ static uint32_t uint(curandState* p_rand_state, uint32_t min = 0, uint32_t max = UINT32_MAX);
	__device__ static float real(curandState* p_rand_state, float min = 0.0f, float max = 1.0f);

private:
	random() = default;
	~random() = default;

};
