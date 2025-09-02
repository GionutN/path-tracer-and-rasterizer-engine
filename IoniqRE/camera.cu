#include "camera.h"

#include "iqmath.h"

camera::camera(uint16_t width, uint16_t height, float fovh, float znear, float zfar)
	:
	m_width(width),
	m_height(height),
	m_fovh(fovh)
{
	m_view = iqmat::look_at(m_position, m_position + m_forward);
	m_projection = iqmat::perspective((float)width / height, to_radians(fovh), znear, zfar);
	//m_projection = iqmat::orthographic((float)width / height, znear, zfar);

	// these must exist, so no need for sanity checks
	m_inv_view = m_view.inveresed();
	m_inv_proj = m_projection.inveresed();
}

__device__ ray camera::get_ray(uint16_t x, uint16_t y, curandState* local_state)
{
	// go in the reverse order of the rasterizer pipeline steps
	// from screen space to ndc
	const float x_ndc = ((x + random::real(local_state, -0.5f, 0.5f)) / (float)m_width) * 2 - 1;
	const float y_ndc = 1 - ((y + random::real(local_state, -0.5f, 0.5f)) / (float)m_height) * 2;

	// get the near and far plane points, the ray passes through them
	const iqvec p_ndc_near(x_ndc, y_ndc, 0.0f, 1.0f);
	const iqvec p_ndc_far(x_ndc, y_ndc, 1.0f, 1.0f);

	// go from clip space to view space
	iqvec p_view_near = p_ndc_near.transformed(m_inv_proj, iqvec::usage::POINT);
	p_view_near /= p_view_near.w;
	iqvec p_view_far = p_ndc_far.transformed(m_inv_proj, iqvec::usage::POINT);
	p_view_far /= p_view_far.w;

	// and from view space to world space
	iqvec p_world_near = p_view_near.transformed(m_inv_view, iqvec::usage::POINT);
	iqvec p_world_far = p_view_far.transformed(m_inv_view, iqvec::usage::POINT);

	const iqvec dir = p_world_far - p_world_near;
	return ray(p_world_near, dir.normalized3());
}
