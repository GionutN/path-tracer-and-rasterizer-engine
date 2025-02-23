#include "shader.h"
#include "shape.h"

__device__ bool sphere::intersect(const ray& r)
{
	float a = r.direction().length3sq();
	float b = 2.0f * r.origin().dot3(r.direction());
	float c = r.origin().length3sq() - 0.25f;
	float delta = b * b - 4.0f * a * c;
	return delta > 0.0f;
}
