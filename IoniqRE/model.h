#pragma once

#include <string>

#include "iqmath.h"

class model
{
public:
	model(size_t mesh_idx)
		:
		m_mesh_index(mesh_idx)
	{}

private:
	size_t m_mesh_index;

	iqmat m_translation;
	iqmat m_rotation;
	iqmat m_scale;
	iqmat m_transform;	// for caching

};
