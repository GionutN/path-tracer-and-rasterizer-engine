#pragma once

#include "shape.h"
#include "random.h"

class material
{
public:
	__device__ virtual bool scatter(const ray& r_in, const hit_record& hr, iqvec* attenuation, ray* r_out, curandState* local_state) const { return false; }

};

class diffuse : public material
{
public:
	__host__ __device__ diffuse(const iqvec& albedo)
		:
		m_albedo(albedo)
	{}

	__device__ bool scatter(const ray& r_in, const hit_record& hr, iqvec* attenuation, ray* r_out, curandState* local_state) const override;

private:
	iqvec m_albedo;

};
