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

__host__ iqvec random::on_unit_sphere()
{
	// TODO: write an actual uniform distribution the unit sphere
	iqvec result;
	result.w = 0.0f;

	while (true) {
		result.x = random::real(-1.0f, 1.0f);
		result.y = random::real(-1.0f, 1.0f);
		result.z = random::real(-1.0f, 1.0f);
		if (result.length3sq() < 1.0f && !result.is_null3()) {
			return result.normalized3();
		}
	}
}

__host__ iqvec random::on_unit_hemisphere(const iqvec& normal)
{
	iqvec dir = random::on_unit_sphere();
	if (dir.dot3(normal) > 0.0f) {
		return dir;
	}

	return -dir;
}

__device__ float random::real(curandState* p_rand_state, float min, float max)
{
	float t = curand(p_rand_state) / (float)UINT32_MAX;
	return t * (max - min) + min;
}

__device__ iqvec random::on_unit_sphere(curandState* p_rand_state)
{
	// TODO: write an actual uniform distribution the unit sphere
	iqvec result;
	result.w = 0.0f;

	while (true) {
		result.x = random::real(p_rand_state, -1.0f, 1.0f);
		result.y = random::real(p_rand_state, -1.0f, 1.0f);
		result.z = random::real(p_rand_state, -1.0f, 1.0f);
		if (result.length3sq() < 1.0f && !result.is_null3()) {
			return result.normalized3();
		}
	}
}

__device__ iqvec random::on_unit_hemisphere(curandState* p_rand_state, const iqvec& normal)
{
	iqvec dir = random::on_unit_sphere(p_rand_state);
	if (dir.dot3(normal) > 0.0f) {
		return dir;
	}

	return -dir;
}
