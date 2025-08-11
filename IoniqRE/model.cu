#include "model.h"

void model::set_transforms(const iqvec& scale, const iqvec& rotation, const iqvec& translation)
{
	m_scale = scale;
	m_rotation = rotation;
	m_translation = translation;
	this->recompute_transform();
}

void model::recompute_transform()
{
	// for direct3d and iqmat the order is  scale*rotation*translation
	iqmat s = iqmat::scale(m_scale);
	iqmat r = iqmat::rotation_x(m_rotation.x) * iqmat::rotation_y(m_rotation.y) * iqmat::rotation_z(m_rotation.z);
	iqmat t = iqmat::translate(m_translation);
	m_transform = s * r * t;
}
