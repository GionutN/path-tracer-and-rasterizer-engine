#pragma once

#include <string>

#include "iqmath.h"

class model
{
	friend class scene;

public:
	model() {}
	model(const std::string& mesh_name)
		:
		m_mesh(mesh_name)
	{}

	const std::string& get_mesh_name() const { return m_mesh; }
	const iqvec& get_translation() const { return m_translation; }
	const iqvec& get_scale() const { return m_scale; }
	const iqmat& get_transform() const { return m_transform; }

private:
	void set_mesh_name(const std::string& name) { m_mesh = name; }

private:
	std::string m_mesh;

	iqvec m_translation = 0.0f;
	iqvec m_rotation = 0.0f;
	iqvec m_scale = 1.0f;
	iqmat m_transform = 1.0f;	// for caching

};
