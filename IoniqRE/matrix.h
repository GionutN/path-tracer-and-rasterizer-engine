#pragma once

#include <cmath>

class iqvec;

class iqmat
{
public:
	iqmat(float _00, float _01, float _02, float _03,
		float _10, float _11, float _12, float _13,
		float _20, float _21, float _22, float _23,
		float _30, float _31, float _32, float _33);
	iqmat(float* data);
	iqmat(float val = 1.0f);
	iqvec row_to_vec(int row) const;
	iqvec col_to_vec(int col) const;

	iqmat operator*(const iqmat& other) const;
	iqmat& operator*=(const iqmat& other);
	iqmat operator*(float s) const;
	iqmat operator/(float s) const;
	iqmat& operator*=(float s);
	iqmat& operator/=(float s);
	inline friend iqmat operator*(float s, const iqmat& other) {
		return other * s;
	}

	iqmat transposed() const;
	iqmat& transpose();
	float determinant() const;

	iqmat inveresed() const;
	iqmat& inverse();

	bool is_identity() const;
	bool is_infinite() const;
	bool is_nan() const;

	static iqmat look_at(const iqvec& eye, const iqvec& focus);
	static iqmat orthographic(float width, float height, float near, float far);
	static iqmat perspective(float aspect_ratio, float fovh, float near, float far);

	static iqmat scale(const iqvec& factor);
	static iqmat translate(const iqvec& offset);
	static iqmat rotation_x(float angle);
	static iqmat rotation_y(float angle);
	static iqmat rotation_z(float angle);
	static iqmat rotation(float angle, const iqvec& axis);

public:
	float m[4][4];

};
