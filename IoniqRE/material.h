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

};

class oren_nayar : public material
{
public:
	__host__ __device__ oren_nayar(const iqvec& albedo, float roughness = 0.0f)
		:
		m_albedo(albedo)
	{
		m_sigma = roughness < 0.0f ? 0.0f : (roughness > 1.0f ? 1.0f : roughness);
	}

	__device__ bool scatter(const ray& r_in, const hit_record& hr, scatter_record* srec, ray* r_out, curandState* local_state) const override;

private:
	__device__ float pdf(const iqvec& wo, const iqvec& normal) const;

private:
	iqvec m_albedo;
	float m_sigma;

};

class emissive : public material
{
public:
	__host__ __device__ emissive(const iqvec& color, float strength)
		:
		m_albedo(color),
		m_strength(strength)
	{}

	__device__ bool scatter(const ray& r_in, const hit_record& hr, scatter_record* srec, ray* r_out, curandState* local_state) const override;

private:
	iqvec m_albedo;
	float m_strength;

};
