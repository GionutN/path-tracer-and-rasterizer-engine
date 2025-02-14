#pragma once

#include <cmath>
#include <string>

#include "matrix.h"

struct vec2
{
	float x;
	float y;

	vec2(float x, float y) : x(x), y(y) {}
	vec2(float val = 0.0f) { x = y = val; }
};

struct vec3
{
	float x;
	float y;
	float z;

	vec3(float x, float y, float z) : x(x), y(y), z(z) {}
	vec3(float val = 0.0f) { x = y = z = val; }
};

class iqvec
{
public:
	enum class usage
	{
		INVALID,
		POINT,
		DIRECTION,
		MISCELLANEOUS,
		NUMUSAGES
	};

public:
	iqvec(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
	iqvec(float val = 0.0f) { x = y = z = w = val; }
	static inline iqvec load(const vec2& other) {
		return iqvec(other.x, other.y, 0.0f, 0.0f);
	}
	static inline iqvec load(const vec3& other) {
		return iqvec(other.x, other.y, other.z, 0.0f);
	}
	inline vec2 store2() const {
		return vec2(x, y);
	}
	inline vec3 store3() const {
		return vec3(x, y, z);
	}

	inline float& operator[](int i) {
		switch (i) {
		case 0: return x;
		case 1: return y;
		case 2: return z;
		case 3:
		default: return w;
		}
	}
	inline float operator[](int i) const {
		switch (i) {
		case 0: return x;
		case 1: return y;
		case 2: return z;
		case 3: return w;
		default: return NAN;
		}
	}
	inline iqvec operator-() const {
		return iqvec(-x, -y, -z, -w);
	}
	inline iqvec operator-(const iqvec& other) const {
		return iqvec(x - other.x, y - other.y, z - other.z, w - other.w);
	}
	inline void operator-=(const iqvec& other) {
		(*this) = (*this) - other;
	}
	inline iqvec operator+(const iqvec& other) const {
		return iqvec(x + other.x, y + other.y, z + other.z, w + other.w);
	}
	inline void operator+=(const iqvec& other) {
		(*this) = (*this) + other;
	}
	inline iqvec operator*(float s) const {
		return iqvec(x * s, y * s, z * s, w * s);
	}
	inline friend iqvec operator*(float s, const iqvec& other) {
		return other * s;
	}
	inline void operator*=(float s) {
		*this = *this * s;
	}
	inline iqvec operator/(float s) const {
		const float inv = 1 / s;
		return (*this) * inv;
	}
	inline void operator/=(float s) {
		*this = (*this) / s;
	}
	inline iqvec hadamard(const iqvec& other) const {
		return iqvec(x * other.x, y * other.y, z * other.z, w * other.w);
	}

	// 2 component vector operations
	inline float dot2(const iqvec& other) const {
		return x * other.x + y * other.y;
	}
	inline float length2sq() const {
		return this->dot2(*this);
	}
	inline float length2() const {
		return std::sqrtf(this->length2sq());
	}
	inline float angle2(const iqvec& other) {
		const float costheta = this->dot2(other) / std::sqrtf(this->length2sq() * other.length2sq());
		return std::acosf(costheta);
	}
	iqvec& clamp_length2(float lo, float hi) {
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
	inline iqvec cross2(const iqvec& other) const {
		return iqvec(0.0f, 0.0f, x * other.y - y * other.x, 0.0f);
	}
	inline bool equal2(const iqvec& other) const {
		const float epsilon = 0.00001f;
		return (std::fabsf(x - other.x) < epsilon && std::fabsf(y - other.y) < epsilon);
	}
	inline bool is_null2() const {
		return this->equal2(iqvec());
	}
	inline bool is_infinite2() const {
		return (std::isinf(x) || std::isinf(y));
	}
	inline bool is_nan2() const {
		return (std::isnan(x) || std::isnan(y));
	}
	inline iqvec normalized2() const {
		if (this->is_null2()) {
			return iqvec();
		}
		return (*this) / this->length2();
	}
	inline iqvec& normalize2() {
		*this = this->normalized2();
		return *this;
	}
	inline iqvec orthogonal2() const {
		return iqvec(-y, x, 0.0f, 0.0f);
	}
	inline iqvec& rotate90() {
		*this = this->orthogonal2();
		return (*this);
	}
	inline iqvec reflected2(const iqvec& normal) const {
		const float s = 2.0f * this->dot2(normal);
		return (*this) - s * normal;
	}
	inline iqvec& reflect2(const iqvec& normal) {
		*this = this->reflected2(normal);
		return *this;
	}
	iqvec refracted2(const iqvec& normal, float n) const {
		const float t = this->dot2(normal);
		const float r = 1.0f - n * n * (1.0f - t * t);
		if (r < 0.0f) {	// total internal reflection
			return this->reflected2(normal);
		}

		const float s = n * t + std::sqrtf(r);
		return n * (*this) - s * normal;
	}
	inline iqvec& refract2(const iqvec& normal, float n) {
		*this = this->refracted2(normal, n);
		return *this;
	}

	// 3 component vector operations
	inline float dot3(const iqvec& other) const {
		return x * other.x + y * other.y + z * other.z;
	}
	inline float length3sq() const {
		return this->dot3(*this);
	}
	inline float length3() const {
		return std::sqrtf(this->length3sq());
	}
	inline float angle3(const iqvec& other) {
		const float costheta = this->dot3(other) / std::sqrtf(this->length3sq() * other.length3sq());
		return std::acosf(costheta);
	}
	iqvec& clamp_length3(float lo, float hi) {
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
	inline iqvec cross3(const iqvec& other) const {
		return iqvec(y * other.z - z * other.y,
			z * other.x - x * other.z,
			x * other.y - y * other.x,
			0.0f);
	}
	inline bool equal3(const iqvec& other) const {
		const float epsilon = 0.00001f;
		return (std::fabsf(x - other.x) < epsilon && std::fabsf(y - other.y) < epsilon
			&& std::fabsf(z - other.z) < epsilon);
	}
	inline bool is_null3() const {
		return this->equal3(iqvec());
	}
	inline bool is_infinite3() const {
		return (std::isinf(x) || std::isinf(y) || std::isinf(z));
	}
	inline bool is_nan3() const {
		return (std::isnan(x) || std::isnan(y) || std::isnan(z));
	}
	inline iqvec normalized3() const {
		if (this->is_null3()) {
			return iqvec();
		}
		return (*this) / this->length3();
	}
	inline iqvec& normalize3() {
		*this = this->normalized3();
		return *this;
	}
	inline iqvec orthogonal3() const {
		return iqvec(y * z, x * z, -2.0f * x * y, 0.0f);
	}
	inline iqvec reflected3(const iqvec& normal) const {
		const float s = 2.0f * this->dot3(normal);
		return (*this) - s * normal;
	}
	inline iqvec& reflect3(const iqvec& normal) {
		*this = this->reflected3(normal);
		return *this;
	}
	iqvec refracted3(const iqvec& normal, float n) const {
		const float t = this->dot3(normal);
		const float r = 1.0f - n * n * (1.0f - t * t);
		if (r < 0.0f) {	// total internal reflection
			return this->reflected3(normal);
		}

		const float s = n * t + std::sqrtf(r);
		return n * (*this) - s * normal;
	}
	inline iqvec& refract3(const iqvec& normal, float n) {
		*this = this->refracted3(normal, n);
		return *this;
	}

	// 4 component vector operations
	inline float dot4(const iqvec& other) const {
		return x * other.x + y * other.y + z * other.z + w * other.w;
	}
	inline float length4sq() const {
		return this->dot4(*this);
	}
	inline float length4() const {
		return std::sqrtf(this->length4sq());
	}
	inline float angle4(const iqvec& other) {
		const float costheta = this->dot4(other) / std::sqrtf(this->length4sq() * other.length4sq());
		return std::acosf(costheta);
	}
	iqvec& clamp_length4(float lo, float hi) {
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
	inline bool equal4(const iqvec& other) const {
		const float epsilon = 0.00001f;
		return (std::fabsf(x - other.x) < epsilon && std::fabsf(y - other.y) < epsilon
			&& std::fabsf(z - other.z) < epsilon && std::fabsf(w - other.w) < epsilon);
	}
	inline bool is_null4() const {
		return this->equal4(iqvec());
	}
	inline bool is_infinite4() const {
		return (std::isinf(x) || std::isinf(y) || std::isinf(z) || std::isinf(w));
	}
	inline bool is_nan4() const {
		return (std::isnan(x) || std::isnan(y) || std::isnan(z) || std::isnan(w));
	}
	inline iqvec normalized4() const {
		if (this->is_null4()) {
			return iqvec();
		}
		return (*this) / this->length4();
	}
	inline iqvec& normalize4() {
		*this = this->normalized4();
		return *this;
	}
	inline iqvec orthogonal4() const {
		return iqvec(z, w, -x, -y);
	}
	inline iqvec reflected4(const iqvec& normal) const {
		const float s = 2.0f * this->dot4(normal);
		return (*this) - s * normal;
	}
	inline iqvec& reflect4(const iqvec& normal) {
		*this = this->reflected4(normal);
		return *this;
	}
	iqvec refracted4(const iqvec& normal, float n) const {
		const float t = this->dot4(normal);
		const float r = 1.0f - n * n * (1.0f - t * t);
		if (r < 0.0f) {	// total internal reflection
			return this->reflected4(normal);
		}

		const float s = n * t + std::sqrtf(r);
		return n * (*this) - s * normal;
	}
	inline iqvec& refract4(const iqvec& normal, float n) {
		*this = this->refracted4(normal, n);
		return *this;
	}

	iqvec swizzle(const std::string& permutation) {
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

	iqvec transformed(const iqmat& mat, usage type) const {
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
	iqvec& transform(const iqmat& mat, usage type) {
		*this = this->transformed(mat, type);
		return *this;
	}

public:
	float x;
	float y;
	float z;
	float w;
};
