#pragma once

#include "shape.h"
#include "random.h"
#include "onb.h"

struct scatter_record
{
	iqvec attenuation;
	float pdf_val;
	float cos_law_weight;
};

class material
{
public:
	__device__ virtual bool scatter(const ray& r_in, const hit_record& hr, scatter_record* srec, ray* r_out, curandState* local_state) const { return false; }

protected:
	__device__ virtual float pdf(const iqvec& wo) const { return 1.0f; }

};

class diffuse_uniform : public material
{
public:
	__host__ __device__ diffuse_uniform(const iqvec& albedo)
		:
		m_albedo(albedo)
	{}

	__device__ bool scatter(const ray& r_in, const hit_record& hr, scatter_record* srec, ray* r_out, curandState* local_state) const override;

private:
	__device__ float pdf(const iqvec& wo) const override { return 1 / tau; }

protected:
	iqvec m_albedo;

};

class diffuse_lambertian : public material
{
public:
	__host__ __device__ diffuse_lambertian(const iqvec& albedo)
		:
		m_albedo(albedo)
	{}

	__device__ bool scatter(const ray& r_in, const hit_record& hr, scatter_record* srec, ray* r_out, curandState* local_state) const override;

private:
	__device__ float pdf(const iqvec& wo) const override { return 1.0f; }

protected:
	iqvec m_albedo;

};
