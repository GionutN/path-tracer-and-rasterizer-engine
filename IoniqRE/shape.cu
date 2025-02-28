#include "shader.h"
#include "shape.h"

__device__ sphere::sphere(const iqvec& pos, float radius)
	:
	m_position(pos),
	m_radius(radius)
{
}

__device__ bool sphere::intersect(const ray& r)
{
	const iqvec oc = m_position - r.origin();
	const float a = r.direction().length3sq();
	const float b = -2.0f * r.direction().dot3(oc);
	const float c = oc.length3sq() - m_radius * m_radius;
	const float delta = b * b - 4 * a * c;

	// compute the ray's intersection point parameter t
	const float t1 = (-b + sqrtf(delta)) / (2.0f * a);
	return t1 > 0.0f;
	
}

__device__ triangle::triangle(const iqvec& v0, const iqvec& v1, const iqvec& v2)
	:
	m_v0(v0),
	m_v1(v1),
	m_v2(v2)
{
}

__device__ bool triangle::intersect(const ray& r)
{
	// compute the triangle's normal
	const iqvec v0v1 = m_v1 - m_v0;
	const iqvec v0v2 = m_v2 - m_v0;
	const iqvec v1v2 = m_v2 - m_v1;
	const iqvec normal = v0v1.cross3(v0v2);

	// check if the ray's direction and the normal are perpendicular (no intersection point)
	if (is_zero(r.direction().dot3(normal))) {
		return false;
	}

	// compute the free d parameter of the plane formula
	const float d = -normal.dot3(m_v0);

	// compute the ray-plane intersection point
	const float t = -(r.origin().dot3(normal) + d) / r.direction().dot3(normal);
	if (t < 0.0f) {
		return false;
	}
	const iqvec p = r.at(t);

	// edge1 inside-outside test
	iqvec ep = p - m_v0;
	iqvec normal2 = v0v1.cross3(ep);
	if (normal2.dot3(normal) < 0.0f) {
		return false;
	}

	// edge1 inside-outside test
	ep = p - m_v1;
	normal2 = v1v2.cross3(ep);
	if (normal2.dot3(normal) < 0.0f) {
		return false;
	}

	// edge1 inside-outside test
	ep = p - m_v0;
	normal2 = ep.cross3(v0v2);
	if (normal2.dot3(normal) < 0.0f) {
		return false;
	}

	return true;
}
