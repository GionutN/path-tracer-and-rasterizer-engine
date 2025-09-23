#pragma once

#include "iqmath.h"

class onb {
public:
	__host__ __device__ onb(const iqvec& n) {
		axis[2] = n.normalized3();
		iqvec a = (fabsf(axis[2].x) > 0.9f) ? iqvec(0.0f, 1.0f, 0.0f, 0.0f) : iqvec(1.0f, 0.0f, 0.0f, 0.0f);
		axis[1] = axis[2].cross3(a).normalize3();
		axis[0] = axis[1].cross3(axis[2]);
	}

	__host__ __device__ const iqvec& u() const { return axis[0]; }
	__host__ __device__ const iqvec& v() const { return axis[1]; }
	__host__ __device__ const iqvec& w() const { return axis[2]; }

	__host__ __device__ iqvec transform_to_world(const iqvec& v) const {
		// transform from basis coordinates to local space
		return (v[0] * axis[0]) + (v[1] * axis[1]) + (v[2] * axis[2]);
	}

private:
	iqvec axis[3];
};
