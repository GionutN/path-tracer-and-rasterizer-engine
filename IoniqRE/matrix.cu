#include "matrix.h"

#include "vector.h"

__host__ __device__ iqmat::iqmat(float _00, float _01, float _02, float _03,
	float _10, float _11, float _12, float _13,
	float _20, float _21, float _22, float _23,
	float _30, float _31, float _32, float _33) {
	m[0][0] = _00; m[0][1] = _01; m[0][2] = _02; m[0][3] = _03;
	m[1][0] = _10; m[1][1] = _11; m[1][2] = _12; m[1][3] = _13;
	m[2][0] = _20; m[2][1] = _21; m[2][2] = _22; m[2][3] = _23;
	m[3][0] = _30; m[3][1] = _31; m[3][2] = _32; m[3][3] = _33;
}
__host__ __device__ iqmat::iqmat(float* data) {
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			m[i][j] = data[i * 4 + j];
		}
	}
}
__host__ __device__ iqmat::iqmat(float val) {
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			m[i][j] = 0.0f;
		}
	}

	m[0][0] = m[1][1] = m[2][2] = m[3][3] = val;
}
__host__ __device__ iqvec iqmat::row_to_vec(int row) const {
	return iqvec(m[row][0], m[row][1], m[row][2], m[row][3]);
}
__host__ __device__ iqvec iqmat::col_to_vec(int col) const {
	return iqvec(m[0][col], m[1][col], m[2][col], m[3][col]);
}

__host__ __device__ iqmat iqmat::operator*(const iqmat& other) const {
	return iqmat(this->row_to_vec(0).dot4(this->col_to_vec(0)),
		this->row_to_vec(0).dot4(this->col_to_vec(1)),
		this->row_to_vec(0).dot4(this->col_to_vec(2)),
		this->row_to_vec(0).dot4(this->col_to_vec(3)),

		this->row_to_vec(1).dot4(this->col_to_vec(0)),
		this->row_to_vec(1).dot4(this->col_to_vec(1)),
		this->row_to_vec(1).dot4(this->col_to_vec(2)),
		this->row_to_vec(1).dot4(this->col_to_vec(3)),

		this->row_to_vec(2).dot4(this->col_to_vec(0)),
		this->row_to_vec(2).dot4(this->col_to_vec(1)),
		this->row_to_vec(2).dot4(this->col_to_vec(2)),
		this->row_to_vec(2).dot4(this->col_to_vec(3)),

		this->row_to_vec(3).dot4(this->col_to_vec(0)),
		this->row_to_vec(3).dot4(this->col_to_vec(1)),
		this->row_to_vec(3).dot4(this->col_to_vec(2)),
		this->row_to_vec(3).dot4(this->col_to_vec(3)));
}
__host__ __device__ iqmat& iqmat::operator*=(const iqmat& other) {
	*this = (*this) * other;
	return *this;
}
__host__ __device__ iqmat iqmat::operator*(float s) const {
	iqmat result;
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			result.m[i][j] = this->m[i][j] * s;
		}
	}

	return result;
}
__host__ __device__ iqmat iqmat::operator/(float s) const {
	s = 1 / s;
	return (*this) * s;
}
__host__ __device__ iqmat& iqmat::operator*=(float s) {
	*this = (*this) * s;
	return *this;
}
__host__ __device__ iqmat& iqmat::operator/=(float s) {
	s = 1 / s;
	*this = (*this) * s;
	return *this;
}

__host__ __device__ iqmat iqmat::transposed() const {
	iqmat result;
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			result.m[j][i] = this->m[i][j];
		}
	}

	return result;
}
__host__ __device__ iqmat& iqmat::transpose() {
	*this = this->transposed();
	return *this;
}
__host__ __device__ float iqmat::determinant() const {
	return
		m[0][3] * m[1][2] * m[2][1] * m[3][0] - m[0][2] * m[1][3] * m[2][1] * m[3][0] -
		m[0][3] * m[1][1] * m[2][2] * m[3][0] + m[0][1] * m[1][3] * m[2][2] * m[3][0] +
		m[0][2] * m[1][1] * m[2][3] * m[3][0] - m[0][1] * m[1][2] * m[2][3] * m[3][0] -
		m[0][3] * m[1][2] * m[2][0] * m[3][1] + m[0][2] * m[1][3] * m[2][0] * m[3][1] +
		m[0][3] * m[1][0] * m[2][2] * m[3][1] - m[0][0] * m[1][3] * m[2][2] * m[3][1] -
		m[0][2] * m[1][0] * m[2][3] * m[3][1] + m[0][0] * m[1][2] * m[2][3] * m[3][1] +
		m[0][3] * m[1][1] * m[2][0] * m[3][2] - m[0][1] * m[1][3] * m[2][0] * m[3][2] -
		m[0][3] * m[1][0] * m[2][1] * m[3][2] + m[0][0] * m[1][3] * m[2][1] * m[3][2] +
		m[0][1] * m[1][0] * m[2][3] * m[3][2] - m[0][0] * m[1][1] * m[2][3] * m[3][2] -
		m[0][2] * m[1][1] * m[2][0] * m[3][3] + m[0][1] * m[1][2] * m[2][0] * m[3][3] +
		m[0][2] * m[1][0] * m[2][1] * m[3][3] - m[0][0] * m[1][2] * m[2][1] * m[3][3] -
		m[0][1] * m[1][0] * m[2][2] * m[3][3] + m[0][0] * m[1][1] * m[2][2] * m[3][3];
}

__host__ __device__ iqmat iqmat::inveresed() const {
	float det = this->determinant();
	if (fabsf(det) < 0.00001f) {
		return iqmat(INFINITY);
	}

	float inv[16];
	float m2[16];
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			m2[i * 4 + j] = m[i][j];
		}
	}

	inv[0] = m2[5] * m2[10] * m2[15] -
		m2[5] * m2[11] * m2[14] -
		m2[9] * m2[6] * m2[15] +
		m2[9] * m2[7] * m2[14] +
		m2[13] * m2[6] * m2[11] -
		m2[13] * m2[7] * m2[10];

	inv[4] = -m2[4] * m2[10] * m2[15] +
		m2[4] * m2[11] * m2[14] +
		m2[8] * m2[6] * m2[15] -
		m2[8] * m2[7] * m2[14] -
		m2[12] * m2[6] * m2[11] +
		m2[12] * m2[7] * m2[10];

	inv[8] = m2[4] * m2[9] * m2[15] -
		m2[4] * m2[11] * m2[13] -
		m2[8] * m2[5] * m2[15] +
		m2[8] * m2[7] * m2[13] +
		m2[12] * m2[5] * m2[11] -
		m2[12] * m2[7] * m2[9];

	inv[12] = -m2[4] * m2[9] * m2[14] +
		m2[4] * m2[10] * m2[13] +
		m2[8] * m2[5] * m2[14] -
		m2[8] * m2[6] * m2[13] -
		m2[12] * m2[5] * m2[10] +
		m2[12] * m2[6] * m2[9];

	inv[1] = -m2[1] * m2[10] * m2[15] +
		m2[1] * m2[11] * m2[14] +
		m2[9] * m2[2] * m2[15] -
		m2[9] * m2[3] * m2[14] -
		m2[13] * m2[2] * m2[11] +
		m2[13] * m2[3] * m2[10];

	inv[5] = m2[0] * m2[10] * m2[15] -
		m2[0] * m2[11] * m2[14] -
		m2[8] * m2[2] * m2[15] +
		m2[8] * m2[3] * m2[14] +
		m2[12] * m2[2] * m2[11] -
		m2[12] * m2[3] * m2[10];

	inv[9] = -m2[0] * m2[9] * m2[15] +
		m2[0] * m2[11] * m2[13] +
		m2[8] * m2[1] * m2[15] -
		m2[8] * m2[3] * m2[13] -
		m2[12] * m2[1] * m2[11] +
		m2[12] * m2[3] * m2[9];

	inv[13] = m2[0] * m2[9] * m2[14] -
		m2[0] * m2[10] * m2[13] -
		m2[8] * m2[1] * m2[14] +
		m2[8] * m2[2] * m2[13] +
		m2[12] * m2[1] * m2[10] -
		m2[12] * m2[2] * m2[9];

	inv[2] = m2[1] * m2[6] * m2[15] -
		m2[1] * m2[7] * m2[14] -
		m2[5] * m2[2] * m2[15] +
		m2[5] * m2[3] * m2[14] +
		m2[13] * m2[2] * m2[7] -
		m2[13] * m2[3] * m2[6];

	inv[6] = -m2[0] * m2[6] * m2[15] +
		m2[0] * m2[7] * m2[14] +
		m2[4] * m2[2] * m2[15] -
		m2[4] * m2[3] * m2[14] -
		m2[12] * m2[2] * m2[7] +
		m2[12] * m2[3] * m2[6];

	inv[10] = m2[0] * m2[5] * m2[15] -
		m2[0] * m2[7] * m2[13] -
		m2[4] * m2[1] * m2[15] +
		m2[4] * m2[3] * m2[13] +
		m2[12] * m2[1] * m2[7] -
		m2[12] * m2[3] * m2[5];

	inv[14] = -m2[0] * m2[5] * m2[14] +
		m2[0] * m2[6] * m2[13] +
		m2[4] * m2[1] * m2[14] -
		m2[4] * m2[2] * m2[13] -
		m2[12] * m2[1] * m2[6] +
		m2[12] * m2[2] * m2[5];

	inv[3] = -m2[1] * m2[6] * m2[11] +
		m2[1] * m2[7] * m2[10] +
		m2[5] * m2[2] * m2[11] -
		m2[5] * m2[3] * m2[10] -
		m2[9] * m2[2] * m2[7] +
		m2[9] * m2[3] * m2[6];

	inv[7] = m2[0] * m2[6] * m2[11] -
		m2[0] * m2[7] * m2[10] -
		m2[4] * m2[2] * m2[11] +
		m2[4] * m2[3] * m2[10] +
		m2[8] * m2[2] * m2[7] -
		m2[8] * m2[3] * m2[6];

	inv[11] = -m2[0] * m2[5] * m2[11] +
		m2[0] * m2[7] * m2[9] +
		m2[4] * m2[1] * m2[11] -
		m2[4] * m2[3] * m2[9] -
		m2[8] * m2[1] * m2[7] +
		m2[8] * m2[3] * m2[5];

	inv[15] = m2[0] * m2[5] * m2[10] -
		m2[0] * m2[6] * m2[9] -
		m2[4] * m2[1] * m2[10] +
		m2[4] * m2[2] * m2[9] +
		m2[8] * m2[1] * m2[6] -
		m2[8] * m2[2] * m2[5];

	det = 1 / det;
	for (int i = 0; i < 16; i++)
		inv[i] *= det;
	return iqmat(inv);
}

__host__ __device__ iqmat& iqmat::inverse() {
	*this = this->inveresed();
	return *this;
}

__host__ __device__ bool iqmat::is_identity() const {
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			if (fabsf(m[i][j]) > 0.00001f) {
				return false;
			}
			if (i == j && fabsf(m[i][i] - 1.0f) > 0.00001f) {
				return false;
			}
		}
	}

	return true;
}
__host__ __device__ bool iqmat::is_infinite() const {
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			if (isinf(m[i][j])) {
				return true;
			}
		}
	}

	return false;
}
__host__ __device__ bool iqmat::is_nan() const {
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			if (isnan(m[i][j])) {
				return true;
			}
		}
	}

	return false;
}

__host__ __device__ iqmat iqmat::look_at(const iqvec& eye, const iqvec& focus) {
	const iqvec aux = iqvec(0.0f, 1.0f, 0.0f, 0.0f);
	iqvec forward = (eye - focus).normalize3();
	iqvec right = aux.cross3(forward);
	iqvec up = forward.cross3(right);
	return iqmat(right.x, right.y, right.z, 0.0f,
		up.x, up.y, up.z, 0.0f,
		forward.x, forward.y, forward.z, 0.0f,
		eye.x, eye.y, eye.z, 1.0f);
}
__host__ __device__ iqmat iqmat::orthographic(float width, float height, float near, float far) {
	if (near < 0.0f || far < 0.0f || fabsf(near - far) < 0.00001f) {
		return iqmat(INFINITY);
	}

	iqmat result(0.0f);
	result.m[0][0] = 2 / width;
	result.m[1][1] = 2 / height;
	result.m[2][2] = 2 / (far - near);
	result.m[3][3] = 1.0f;
	result.m[3][2] = (near + far) / (near - far);
	return result;
}
__host__ __device__ iqmat iqmat::perspective(float aspect_ratio, float fovh, float near, float far) {
	if (near < 0.0f || far < 0.0f || fabsf(near - far) < 0.00001f) {
		return iqmat(INFINITY);
	}

	const float top = tanf(fovh / 2) * near;
	const float bottom = -top;
	const float right = top * aspect_ratio;
	const float left = -right;

	iqmat result(0.0f);
	result.m[0][0] = 2 * near / (right - left);
	result.m[1][1] = 2 * near / (top - bottom);
	result.m[2][0] = (right + left) / (right - left);
	result.m[2][1] = (top + bottom) / (top - bottom);
	result.m[2][2] = (near + far) / (near - far);
	result.m[2][3] = -1.0f;
	result.m[3][2] = 2 * near * far / (near - far);
	return result;
}

__host__ __device__ iqmat iqmat::scale(const iqvec& factor) {
	iqmat result;
	result.m[0][0] = factor.x;
	result.m[1][1] = factor.y;
	result.m[2][2] = factor.z;
	return result;
}
__host__ __device__ iqmat iqmat::translate(const iqvec& offset) {
	iqmat result;
	result.m[3][0] = offset.x;
	result.m[3][1] = offset.y;
	result.m[3][2] = offset.z;
	return result;
}
__host__ __device__ iqmat iqmat::rotation_x(float angle) {
	iqmat result;
	const float s = sinf(angle);
	const float c = cosf(angle);

	result.m[1][1] = c;
	result.m[1][2] = s;
	result.m[2][2] = -s;
	result.m[2][2] = c;
	return result;
}
__host__ __device__ iqmat iqmat::rotation_y(float angle) {
	iqmat result;
	const float s = sinf(angle);
	const float c = cosf(angle);

	result.m[0][0] = c;
	result.m[0][2] = -s;
	result.m[2][0] = s;
	result.m[2][2] = c;
	return result;
}
__host__ __device__ iqmat iqmat::rotation_z(float angle) {
	iqmat result;
	const float s = sinf(angle);
	const float c = cosf(angle);

	result.m[0][0] = c;
	result.m[0][1] = s;
	result.m[1][0] = -s;
	result.m[1][1] = c;
	return result;
}
__host__ __device__ iqmat iqmat::rotation(float angle, const iqvec& axis) {
	iqmat result;
	const float s = sinf(angle);
	const float c = cosf(angle);

	result.m[0][0] = c + axis.x * axis.x * (1.0f - c);
	result.m[0][1] = axis.y * axis.x * (1.0f - c) + axis.z * s;
	result.m[0][2] = axis.z * axis.x * (1.0f - c) - axis.y * s;

	result.m[1][0] = axis.x * axis.y * (1.0f - c) - axis.z * s;
	result.m[1][1] = c + axis.y * axis.y * (1.0f - c);
	result.m[1][2] = axis.z * axis.y * (1.0f - c) + axis.x * s;

	result.m[2][0] = axis.x * axis.z * (1.0f - c) + axis.y * s;
	result.m[2][1] = axis.y * axis.z * (1.0f - c) - axis.x * s;
	result.m[2][2] = c + axis.z * axis.z * (1.0f - c);
	return result;
}
