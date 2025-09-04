#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

struct mat3x3
{
public:
	__host__ __device__ mat3x3(float _00, float _01, float _02,
		float _10, float _11, float _12,
		float _20, float _21, float _22);
	__host__ __device__ mat3x3(float* data);
	__host__ __device__ mat3x3(float val = 1.0f);

	__host__ __device__ float determinant() const;
	__host__ __device__ mat3x3 inversed() const;
	__host__ __device__ mat3x3& inverse();
	__host__ __device__ mat3x3 transposed() const;
	__host__ __device__ mat3x3& transpose();

public:
	// 4 columns to match the hlsl format
	// float3x3 is actually 3 float4s, so 48 bytes
	float m[3][4];
};

class iqvec;

// hlsl compatible (column major matrix class)
class iqmat
{
public:
	__host__ __device__ iqmat(float _00, float _01, float _02, float _03,
		float _10, float _11, float _12, float _13,
		float _20, float _21, float _22, float _23,
		float _30, float _31, float _32, float _33);
	__host__ __device__ iqmat(float* data);
	__host__ __device__ iqmat(float val = 1.0f);
	__host__ __device__ iqvec row_to_vec(int row) const;
	__host__ __device__ iqvec col_to_vec(int col) const;
	__host__ __device__ static iqmat load3x3(const mat3x3& other, float val = 1.0f);	// the value is the cell [3, 3]
	__host__ __device__ mat3x3 store3x3() const;

	__host__ __device__ iqmat operator*(const iqmat& other) const;
	__host__ __device__ iqmat& operator*=(const iqmat& other);
	__host__ __device__ iqmat operator*(float s) const;
	__host__ __device__ iqmat operator/(float s) const;
	__host__ __device__ iqmat& operator*=(float s);
	__host__ __device__ iqmat& operator/=(float s);
	__host__ __device__ inline friend iqmat operator*(float s, const iqmat& other) {
		return other * s;
	}

	__host__ __device__ iqmat transposed() const;
	__host__ __device__ iqmat& transpose();
	__host__ __device__ float determinant() const;

	__host__ __device__ iqmat inversed() const;
	__host__ __device__ iqmat& inverse();

	__host__ __device__ bool is_identity() const;
	__host__ __device__ bool is_infinite() const;
	__host__ __device__ bool is_nan() const;

	__host__ __device__ static iqmat look_at(const iqvec& eye, const iqvec& focus);
	__host__ __device__ static iqmat orthographic(float aspect_ratio, float znear, float zfar);
	__host__ __device__ static iqmat perspective(float aspect_ratio, float fovh, float znear, float zfar);

	__host__ __device__ static iqmat scale(const iqvec& factor);
	__host__ __device__ static iqmat translate(const iqvec& offset);
	__host__ __device__ static iqmat rotation_x(float radians);
	__host__ __device__ static iqmat rotation_y(float radians);
	__host__ __device__ static iqmat rotation_z(float radians);
	__host__ __device__ static iqmat rotation(float radians, const iqvec& axis);

	float* data() { return &m[0][0]; }
	const float* data() const { return &m[0][0]; }

public:
	float m[4][4];

};
