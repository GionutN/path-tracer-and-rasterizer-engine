#pragma once

#include <cmath>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

#include "matrix.h"

struct vec2
{
	float x;
	float y;

	__host__ __device__ vec2(float x, float y) : x(x), y(y) {}
	__host__ __device__ vec2(float val = 0.0f) { x = y = val; }
};

struct vec3
{
	float x;
	float y;
	float z;

	__host__ __device__ vec3(float x, float y, float z) : x(x), y(y), z(z) {}
	__host__ __device__ vec3(float val = 0.0f) { x = y = z = val; }
};

class iqvec
{
public:
	enum class usage
	{
		INVALID = -1,
		DIRECTION,
		POINT,
		MISCELLANEOUS,
		NUMUSAGES
	};

public:
	__host__ __device__ iqvec(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
	__host__ __device__ iqvec(float val = 0.0f) { x = y = z = w = val; }
	__host__ __device__ static inline iqvec load(const vec2& other, usage u) {
		return iqvec(other.x, other.y, 0.0f, (float)u);
	}
	__host__ __device__ static inline iqvec load(const vec3& other, usage u) {
		return iqvec(other.x, other.y, other.z, (float)u);
	}
	__host__ __device__ inline vec2 store2() const {
		return vec2(x, y);
	}
	__host__ __device__ inline vec3 store3() const {
		return vec3(x, y, z);
	}

	__host__ __device__ inline float& operator[](int i) {
		switch (i) {
		case 0: return x;
		case 1: return y;
		case 2: return z;
		case 3:
		default: return w;
		}
	}
	__host__ __device__ inline float operator[](int i) const {
		switch (i) {
		case 0: return x;
		case 1: return y;
		case 2: return z;
		case 3: return w;
		default: return NAN;
		}
	}
	__host__ __device__ inline iqvec operator-() const {
		return iqvec(-x, -y, -z, -w);
	}
	__host__ __device__ inline iqvec operator-(const iqvec& other) const {
		return iqvec(x - other.x, y - other.y, z - other.z, w - other.w);
	}
	__host__ __device__ inline void operator-=(const iqvec& other) {
		(*this) = (*this) - other;
	}
	__host__ __device__ inline iqvec operator+(const iqvec& other) const {
		return iqvec(x + other.x, y + other.y, z + other.z, w + other.w);
	}
	__host__ __device__ inline void operator+=(const iqvec& other) {
		(*this) = (*this) + other;
	}
	__host__ __device__ inline iqvec operator*(float s) const {
		return iqvec(x * s, y * s, z * s, w * s);
	}
	__host__ __device__ inline friend iqvec operator*(float s, const iqvec& other) {
		return other * s;
	}
	__host__ __device__ inline void operator*=(float s) {
		*this = *this * s;
	}
	__host__ __device__ inline iqvec operator/(float s) const {
		const float inv = 1 / s;
		return (*this) * inv;
	}
	__host__ __device__ inline void operator/=(float s) {
		*this = (*this) / s;
	}
	__host__ __device__ inline iqvec hadamard(const iqvec& other) const {
		return iqvec(x * other.x, y * other.y, z * other.z, w * other.w);
	}

	// 2 component vector operations
	__host__ __device__ inline float dot2(const iqvec& other) const {
		return x * other.x + y * other.y;
	}
	__host__ __device__ inline float length2sq() const {
		return this->dot2(*this);
	}
	__host__ __device__ inline float length2() const {
		return sqrtf(this->length2sq());
	}
	__host__ __device__ inline float angle2(const iqvec& other) {
		const float costheta = this->dot2(other) / sqrtf(this->length2sq() * other.length2sq());
		return acosf(costheta);
	}
	__host__ __device__ iqvec& clamp_length2(float lo, float hi) {
		const float len = this->length2();
		if (len < lo) {
			(*this) *= (lo / len);
			return *this;
		}
		if (len > hi) {
			*this *= (hi / len);
			return *this;
		}
		return *this;
	}
	__host__ __device__ inline iqvec cross2(const iqvec& other) const {
		return iqvec(0.0f, 0.0f, x * other.y - y * other.x, 0.0f);
	}
	__host__ __device__ inline bool equal2(const iqvec& other) const {
		const float epsilon = 0.00001f;
		return (fabsf(x - other.x) < epsilon && fabsf(y - other.y) < epsilon);
	}
	__host__ __device__ inline bool is_null2() const {
		return this->equal2(iqvec());
	}
	__host__ __device__ inline bool is_infinite2() const {
		return (isinf(x) || isinf(y));
	}
	__host__ __device__ inline bool is_nan2() const {
		return (isnan(x) || isnan(y));
	}
	__host__ __device__ inline iqvec normalized2() const {
		if (this->is_null2()) {
			return iqvec();
		}
		return (*this) / this->length2();
	}
	__host__ __device__ inline iqvec& normalize2() {
		*this = this->normalized2();
		return *this;
	}
	__host__ __device__ inline iqvec orthogonal2() const {
		return iqvec(-y, x, 0.0f, 0.0f);
	}
	__host__ __device__ inline iqvec& rotate90() {
		*this = this->orthogonal2();
		return (*this);
	}
	__host__ __device__ inline iqvec reflected2(const iqvec& normal) const {
		const float s = 2.0f * this->dot2(normal);
		return (*this) - s * normal;
	}
	__host__ __device__ inline iqvec& reflect2(const iqvec& normal) {
		*this = this->reflected2(normal);
		return *this;
	}
	__host__ __device__ iqvec refracted2(const iqvec& normal, float n) const {
		const float t = this->dot2(normal);
		const float r = 1.0f - n * n * (1.0f - t * t);
		if (r < 0.0f) {	// total internal reflection
			return this->reflected2(normal);
		}

		const float s = n * t + sqrtf(r);
		return n * (*this) - s * normal;
	}
	__host__ __device__ inline iqvec& refract2(const iqvec& normal, float n) {
		*this = this->refracted2(normal, n);
		return *this;
	}

	// 3 component vector operations
	__host__ __device__ inline float dot3(const iqvec& other) const {
		return x * other.x + y * other.y + z * other.z;
	}
	__host__ __device__ inline float length3sq() const {
		return this->dot3(*this);
	}
	__host__ __device__ inline float length3() const {
		return sqrtf(this->length3sq());
	}
	__host__ __device__ inline float angle3(const iqvec& other) {
		const float costheta = this->dot3(other) / sqrtf(this->length3sq() * other.length3sq());
		return acosf(costheta);
	}
	__host__ __device__ iqvec& clamp_length3(float lo, float hi) {
		const float len = this->length3();
		if (len < lo) {
			(*this) *= (lo / len);
			return *this;
		}
		if (len > hi) {
			*this *= (hi / len);
			return *this;
		}
		return *this;
	}
	__host__ __device__ inline iqvec cross3(const iqvec& other) const {
		return iqvec(y * other.z - z * other.y,
			z * other.x - x * other.z,
			x * other.y - y * other.x,
			0.0f);
	}
	__host__ __device__ inline bool equal3(const iqvec& other) const {
		const float epsilon = 0.00001f;
		return (fabsf(x - other.x) < epsilon && fabsf(y - other.y) < epsilon
			&& fabsf(z - other.z) < epsilon);
	}
	__host__ __device__ inline bool is_null3() const {
		return this->equal3(iqvec());
	}
	__host__ __device__ inline bool is_infinite3() const {
		return (isinf(x) || isinf(y) || isinf(z));
	}
	__host__ __device__ inline bool is_nan3() const {
		return (isnan(x) || isnan(y) || isnan(z));
	}
	__host__ __device__ inline iqvec normalized3() const {
		if (this->is_null3()) {
			return iqvec();
		}
		return (*this) / this->length3();
	}
	__host__ __device__ inline iqvec& normalize3() {
		*this = this->normalized3();
		return *this;
	}
	__host__ __device__ inline iqvec orthogonal3() const {
		return iqvec(y * z, x * z, -2.0f * x * y, 0.0f);
	}
	__host__ __device__ inline iqvec reflected3(const iqvec& normal) const {
		const float s = 2.0f * this->dot3(normal);
		return (*this) - s * normal;
	}
	__host__ __device__ inline iqvec& reflect3(const iqvec& normal) {
		*this = this->reflected3(normal);
		return *this;
	}
	__host__ __device__ iqvec refracted3(const iqvec& normal, float n) const {
		const float t = this->dot3(normal);
		const float r = 1.0f - n * n * (1.0f - t * t);
		if (r < 0.0f) {	// total internal reflection
			return this->reflected3(normal);
		}

		const float s = n * t + sqrtf(r);
		return n * (*this) - s * normal;
	}
	__host__ __device__ inline iqvec& refract3(const iqvec& normal, float n) {
		*this = this->refracted3(normal, n);
		return *this;
	}

	// 4 component vector operations
	__host__ __device__ inline float dot4(const iqvec& other) const {
		return x * other.x + y * other.y + z * other.z + w * other.w;
	}
	__host__ __device__ inline float length4sq() const {
		return this->dot4(*this);
	}
	__host__ __device__ inline float length4() const {
		return sqrtf(this->length4sq());
	}
	__host__ __device__ inline float angle4(const iqvec& other) {
		const float costheta = this->dot4(other) / sqrtf(this->length4sq() * other.length4sq());
		return acosf(costheta);
	}
	__host__ __device__ iqvec& clamp_length4(float lo, float hi) {
		const float len = this->length4();
		if (len < lo) {
			(*this) *= (lo / len);
			return *this;
		}
		if (len > hi) {
			*this *= (hi / len);
			return *this;
		}
		return *this;
	}
	__host__ __device__ inline bool equal4(const iqvec& other) const {
		const float epsilon = 0.00001f;
		return (fabsf(x - other.x) < epsilon && fabsf(y - other.y) < epsilon
			 && fabsf(z - other.z) < epsilon && fabsf(w - other.w) < epsilon);
	}
	__host__ __device__ inline bool is_null4() const {
		return this->equal4(iqvec());
	}
	__host__ __device__ inline bool is_infinite4() const {
		return (isinf(x) || isinf(y) || isinf(z) || isinf(w));
	}
	__host__ __device__ inline bool is_nan4() const {
		return (isnan(x) || isnan(y) || isnan(z) || isnan(w));
	}
	__host__ __device__ inline iqvec normalized4() const {
		if (this->is_null4()) {
			return iqvec();
		}
		return (*this) / this->length4();
	}
	__host__ __device__ inline iqvec& normalize4() {
		*this = this->normalized4();
		return *this;
	}
	__host__ __device__ inline iqvec orthogonal4() const {
		return iqvec(z, w, -x, -y);
	}
	__host__ __device__ inline iqvec reflected4(const iqvec& normal) const {
		const float s = 2.0f * this->dot4(normal);
		return (*this) - s * normal;
	}
	__host__ __device__ inline iqvec& reflect4(const iqvec& normal) {
		*this = this->reflected4(normal);
		return *this;
	}
	__host__ __device__ iqvec refracted4(const iqvec& normal, float n) const {
		const float t = this->dot4(normal);
		const float r = 1.0f - n * n * (1.0f - t * t);
		if (r < 0.0f) {	// total internal reflection
			return this->reflected4(normal);
		}

		const float s = n * t + sqrtf(r);
		return n * (*this) - s * normal;
	}
	__host__ __device__ inline iqvec& refract4(const iqvec& normal, float n) {
		*this = this->refracted4(normal, n);
		return *this;
	}

	__host__ __device__ iqvec swizzle(const std::string& permutation) {
		if (permutation.size() > 4) {
			return iqvec();
		}

		iqvec result;
		for (int i = 0; i < permutation.size(); i++) {
			switch (permutation[i]) {
			case 'x': result[i] = x; break;
			case 'y': result[i] = y; break;
			case 'z': result[i] = z; break;
			case 'w': result[i] = w; break;
			default: return iqvec();
			}
		}

		return result;
	}

	__host__ __device__ iqvec transformed(const iqmat& mat, usage type) const {
		iqvec aux = *this, result;
		switch (type) {
		case usage::POINT: aux.w = 1.0f; break;
		case usage::DIRECTION: aux.w = 0.0f; break;
		}

		result.x = aux.dot4(mat.col_to_vec(0));
		result.y = aux.dot4(mat.col_to_vec(1));
		result.z = aux.dot4(mat.col_to_vec(2));
		result.w = aux.dot4(mat.col_to_vec(3));
		return result;
	}
	__host__ __device__ iqvec& transform(const iqmat& mat, usage type) {
		*this = this->transformed(mat, type);
		return *this;
	}

public:
	float x;
	float y;
	float z;
	float w;
};
