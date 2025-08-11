#include "camera.h"

#include "iqmath.h"

camera::camera(uint16_t width, uint16_t height, float fovh, float znear, float zfar)
{
	m_view = iqmat::look_at(m_position, m_position + m_forward);
	m_projection = iqmat::perspective((float)width / height, to_radians(fovh), znear, zfar);
	//m_projection = iqmat::orthographic((float)width / height, znear, zfar);
}
