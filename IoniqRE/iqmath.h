#pragma once

#include <cmath>

#include "vector.h"
#include "matrix.h"

static constexpr float pi = 3.1415926535897932384626433832795f;
static constexpr float tau = 6.283185307179586476925286766559f;
static constexpr float pi_div_2 = 1.5707963267948966192313216916398f;
static constexpr float pi_div_4 = 0.78539816339744830961566084581988f;
static constexpr float one_div_pi = 0.31830988618379067153776752674503f;
static constexpr float one_div_tau = 0.15915494309189533576888376337251f;

__host__ __device__ static float to_degrees(float radians)
{
	return radians * 180.0f / pi;
}

__host__ __device__ static float to_radians(float angle)
{
	return angle * pi / 180.0f;
}

__host__ __device__ static bool is_zero(float val)
{
	return fabsf(val) < 0.000001f;
}
