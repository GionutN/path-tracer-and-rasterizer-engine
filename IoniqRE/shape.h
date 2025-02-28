#pragma once

#include "ray.h"

class shape
{
public:
	__device__ virtual bool intersect(const ray& r) = 0;
};

class sphere : public shape
{
public:
	__device__ sphere(const iqvec& pos, float radius);

	__device__ bool intersect(const ray& r) override;

private:
	iqvec m_position;
	float m_radius;

};

class triangle : public shape
{
public:
	// v0, v1 and v2 must be in counter-clockwise orientation
	__device__ triangle(const iqvec& v0, const iqvec& v1, const iqvec& v2);

	__device__ bool intersect(const ray& r) override;

private:
	iqvec m_v0;
	iqvec m_v1;
	iqvec m_v2;
};
