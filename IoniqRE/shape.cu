#include "shader.h"
#include "shape.h"

#define MOLLER_TRUMBORE 1

__device__ sphere::sphere(const iqvec& pos, float radius)
	:
	m_position(pos),
	m_radius(radius)
{
}

__device__ bool sphere::intersect(const ray& r)
{
	// the coefficient of t^2 is 1 because the ray direction is normalized
	const iqvec oc = m_position - r.origin();
	const float halfb = r.direction().dot3(oc);
	const float c = oc.length3sq() - m_radius * m_radius;
	const float delta = halfb * halfb - c;

	// compute the ray's intersection point parameter t
	if (delta < 0.0f) {
		return false;
	}
	const float t1 = halfb - sqrtf(delta);
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
#if MOLLER_TRUMBORE
	const iqvec v0v1 = m_v1 - m_v0;
	const iqvec v0v2 = m_v2 - m_v0;
	const iqvec pvec = r.direction().cross3(v0v2);
	float det = v0v1.dot3(pvec);

	// check if the ray's direction and the normal are perpendicular (no intersection point)
	if (is_zero(fabs(det))) {
		return false;
	}

	det = 1 / det;
	iqvec tvec = r.origin() - m_v0;
	const float u = tvec.dot3(pvec) * det;
	if (u < 0 || u > 1) {
		return false;
	}

	iqvec qvec = tvec.cross3(v0v1);
	const float v = r.direction().dot3(qvec) * det;
	if (v < 0 || u + v > 1) {
		return false;
	}

	const float t = v0v2.dot3(qvec) * det;
	return true;
#else
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
#endif
}
