#include "scene.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include "renderer.h"

scene::scene()
	:
	m_model_types({0, 0})
{
}

void scene::add(const mesh& m)
{
	m_models.push_back(m);
	m_model_types[(size_t)m.get_type()]++;

	m_vertices += m.get_vertices().size();
	m_indices += m.get_indices().size();
}

void scene::change(const mesh& m)
{
	m_vertices -= m_models[0].get_vertices().size();
	m_indices -= m_models[0].get_indices().size();
	m_models[0] = m;
	m_vertices += m.get_vertices().size();
	m_indices += m.get_indices().size();
}

scene::gpu_packet scene::build_packet() const
{
	if (m_vertices == 0) {
		return { nullptr, nullptr, nullptr };
	}

	gpu_packet packet;
	packet.vertices = new vertex[m_vertices];
	packet.indices = new UINT[m_indices + 1];
	packet.indices[0] = m_indices;	// set the number of indices as the first element in the array
	packet.model_types = new UINT[2];

	// bundle all the scene's vertices and indices in one big array
	size_t vert_idx = 0, index_idx = 1;
	UINT index_offset = 0;	// shift the indices of the model by the total number of vertices from the previously added meshes
	for (const auto& m : m_models) {
		for (const auto& v : m.get_vertices()) {
			packet.vertices[vert_idx++] = v;
		}

		// CCW sent to the gpu
		for (const auto& i : m.get_indices()) {
			packet.indices[index_idx++] = i + index_offset;	// weird compiler warning
		}
		index_offset += (UINT)m.get_vertices().size();
	}

	packet.model_types[(size_t)mesh::type::TRIANGLES] = m_model_types[(size_t)mesh::type::TRIANGLES];
	packet.model_types[(size_t)mesh::type::SPHERES] = m_model_types[(size_t)mesh::type::SPHERES];

	// copy all the arrays to gpu memory;
	cudaError cderr;
	vertex* vertices;
	UINT* indices;
	UINT* model_types;

	RENDERER_THROW_CUDA(cudaMalloc((void**)&vertices, m_vertices * sizeof(vertex)));
	RENDERER_THROW_CUDA(cudaMalloc((void**)&indices, (m_indices + 1) * sizeof(UINT)));
	RENDERER_THROW_CUDA(cudaMalloc((void**)&model_types, (size_t)mesh::type::NUMTYPES * sizeof(mesh::type)));
	RENDERER_THROW_CUDA(cudaMemcpy(vertices, packet.vertices, m_vertices * sizeof(vertex), cudaMemcpyHostToDevice));
	RENDERER_THROW_CUDA(cudaMemcpy(indices, packet.indices, (m_indices + 1) * sizeof(UINT), cudaMemcpyHostToDevice));
	RENDERER_THROW_CUDA(cudaMemcpy(model_types, packet.model_types, (size_t)mesh::type::NUMTYPES * sizeof(mesh::type), cudaMemcpyHostToDevice));

	// delete and reassign the cpu memory data
	delete[] packet.vertices; packet.vertices = vertices;
	delete[] packet.indices; packet.indices = indices;
	delete[] packet.model_types; packet.model_types = model_types;

	return packet;
}
