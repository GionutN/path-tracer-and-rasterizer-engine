#pragma once

#include <string>

#include "iqmath.h"

class model
{
public:
	model(const std::string& mesh_name)
		:
		m_mesh(mesh_name)
	{}

	const std::string& get_mesh_name() const { return m_mesh; }
	const iqvec& get_translation() const { return m_translation; }
	const iqvec& get_scale() const { return m_scale; }
	const iqmat& get_transform() const { return m_transform; }

private:
	std::string m_mesh;

	iqvec m_translation;
	iqvec m_rotation;
	iqvec m_scale;
	iqmat m_transform;	// for caching

};
