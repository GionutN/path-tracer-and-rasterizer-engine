#pragma once

#include "iqmath.h"

class ray
{
public:
	__device__ ray(const iqvec& o, const iqvec& d)
		:
		m_origin(o),
		m_direction(d)
	{}
	__device__ ray()
		:
		m_origin(iqvec()),
		m_direction(iqvec(0.0f, 0.0f, -1.0f, 0.0f))
	{}

	__device__ inline iqvec get_origin() const { return m_origin; }
	__device__ inline iqvec get_direction() const { return m_direction; }
	__device__ inline iqvec at(float t) const { return m_origin + t * m_direction; }

private:
	iqvec m_origin;
	iqvec m_direction;

};
