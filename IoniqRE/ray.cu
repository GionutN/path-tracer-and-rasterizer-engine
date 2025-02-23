#include "ray.h"

__device__ ray::ray(const iqvec& o, const iqvec& d)
	:
	m_origin(o),
	m_direction(d)
{}
__device__ ray::ray()
	:
	m_origin(iqvec()),
	m_direction(iqvec(0.0f, 0.0f, -1.0f, 0.0f))
{}
