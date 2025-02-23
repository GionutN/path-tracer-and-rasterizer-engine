#pragma once

#include "ray.h"

class shape
{
public:
	__device__ virtual bool intersect(const ray& r) = 0;
};

class sphere : public shape
{
public:
	sphere() = default;

	__device__ bool intersect(const ray& r) override;

};
