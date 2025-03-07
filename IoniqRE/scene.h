#pragma once

#include <vector>
#include <array>

#include "mesh.h"
#include "shape.h"

class scene
{
public:
	// represents the structure that needs to be sent to the gpu for rendering
	struct gpu_packet
	{
		vertex* vertices;
		UINT* indices;
		UINT* model_types;
	};

public:
	scene();

	const std::vector<mesh>& meshes() const { return m_models; }

	void add(const mesh& m);
	void change(const mesh& m);
	gpu_packet build_packet() const;

private:
	std::vector<mesh> m_models;
	std::array<UINT, 2> m_model_types;

	size_t m_vertices = 0;	// total number of vertices
	size_t m_indices = 0;	// total number of indices

};
