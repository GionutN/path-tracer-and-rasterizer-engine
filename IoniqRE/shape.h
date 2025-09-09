#pragma once

#include "ray.h"

struct hit_record
{
	iqvec p;
	iqvec n;
	float t;
	bool front_face;
};

class shape
{
public:
	__device__ virtual bool intersect(const ray& r, hit_record* hr) = 0;
};

class sphere : public shape
{
public:
	__device__ sphere(const iqvec& pos, float radius);

	__device__ bool intersect(const ray& r, hit_record* hr) override;

private:
	iqvec m_position;
	float m_radius;

};

class triangle : public shape
{
public:
	// v0, v1 and v2 must be in counter-clockwise orientation
	__device__ triangle(const iqvec& v0, const iqvec& v1, const iqvec& v2, const iqvec& n);
	__device__ triangle(const iqvec& v0, const iqvec& v1, const iqvec& v2, const iqvec& n0, const iqvec& n1, const iqvec& n2);

	__device__ bool intersect(const ray& r, hit_record* hr) override;

private:
	iqvec m_v0, m_v1, m_v2;	// vertex positions
	iqvec m_n0, m_n1, m_n2;	// vertex normals
};
