#include "random.h"

static std::mt19937 g_engine;

void random::init()
{
	g_engine.seed(std::random_device()());
}

__host__ uint32_t random::uint(uint32_t min, uint32_t max)
{
	std::uniform_int_distribution<uint32_t> distribution(min, max);
	return distribution(g_engine);
}

__device__ uint32_t random::uint(curandState* p_rand_state, uint32_t min, uint32_t max)
{
	uint32_t rand_int = curand(p_rand_state);
	return min + (rand_int % (max - min + 1));
}

__host__ float random::real(float min, float max)
{
	std::uniform_int_distribution<int> distribution(0, UINT32_MAX); //uniform real distribution is really bad(_fdlog())
	float t = distribution(g_engine) / (float)UINT32_MAX;
	return t * (max - min) + min;
}

__device__ float random::real(curandState* p_rand_state, float min, float max)
{
	float t = curand(p_rand_state) / (float)UINT32_MAX;
	return t * (max - min) + min;
}
