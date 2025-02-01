#include "mesh.h"

#include "renderer.h"

mesh::mesh(const std::vector<vertex>& verts, const std::vector<UINT>& ids)
	:
	m_vertices(verts),
	m_indices(ids)
{
	this->setup_mesh();
}

mesh::~mesh()
{
	m_vertices.clear();
	m_indices.clear();
}

void mesh::setup_mesh()
{
	renderer::get()->create_buffer<vertex>(D3D11_BIND_VERTEX_BUFFER, m_vertices.data(), m_vertices.size(), m_vbuff);
	renderer::get()->create_buffer<UINT>(D3D11_BIND_INDEX_BUFFER, m_indices.data(), m_indices.size(), m_ibuff);
}

triangle::triangle()
{
	m_vertices = {
		{ 0.0f,  0.5f },
		{ 0.5f, -0.5f },
		{-0.5f, -0.5f }
	};

	m_indices = {
		0, 1, 2
	};

	this->setup_mesh();
}

quad::quad()
{
	m_vertices = {
		{-0.5f,  0.5f },
		{ 0.5f,  0.5f },
		{ 0.5f, -0.5f },
		{-0.5f, -0.5f },
	};

	m_indices = {
		0, 1, 3,
		1, 2, 3
	};

	this->setup_mesh();
}
