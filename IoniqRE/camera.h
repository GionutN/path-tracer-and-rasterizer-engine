#pragma once

#include "vector.h"
#include "matrix.h"
#include "ray.h"
#include "random.h"

class camera
{
public:
	__host__ __device__ camera(uint16_t width, uint16_t height, float fovh = 45.0f, float znear = 0.01f, float zfar = 100.0f);

	const iqmat& get_view() const { return m_view; }
	const iqmat& get_projection() const { return m_projection; }

	__host__ __device__ uint16_t get_width() const { return m_width; }
	__host__ __device__ uint16_t get_height() const { return m_height; }

	__device__ ray get_ray(uint16_t x, uint16_t y, curandState* local_state);

private:
	uint16_t m_width;
	uint16_t m_height;
	float m_fovh;

	iqvec m_position = iqvec(0.0f, 0.5f, -3.0f, 0.0f);
	iqvec m_forward = iqvec(0.0f, -0.5f, 3.0f, 0.0f);

	iqmat m_view;
	iqmat m_projection;
	iqmat m_inv_view;
	iqmat m_inv_proj;

};
