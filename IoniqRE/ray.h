#pragma once

#include "iqmath.h"

class ray
{
public:
	__device__ ray(const iqvec& o, const iqvec& d);
	__device__ ray();

	__device__ inline iqvec origin() const { return m_origin; }
	__device__ inline iqvec direction() const { return m_direction; }
	__device__ inline iqvec at(float t) const { return m_origin + t * m_direction; }

private:
	iqvec m_origin;
	iqvec m_direction;

};
