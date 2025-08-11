#pragma once

#include "vector.h"
#include "matrix.h"

class camera
{
public:
	camera(uint16_t width, uint16_t height, float fovh = 45.0f, float znear = 0.01f, float zfar = 1000.0f);

	const iqmat& get_view() const { return m_view; }
	const iqmat& get_projection() const { return m_projection; }

private:
	iqvec m_position = iqvec(0.0f, 0.0f, -3.0f, 0.0f);
	iqvec m_forward = iqvec(0.0f, 0.0f, 1.0f, 0.0f);

	iqmat m_view;
	iqmat m_projection;

};
