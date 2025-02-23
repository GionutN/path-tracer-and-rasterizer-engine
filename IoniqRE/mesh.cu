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

void mesh::bind() const
{
	UINT stride = sizeof(vertex), offset = 0;
	RENDERER_CTX->IASetVertexBuffers(0, 1, m_vbuff.GetAddressOf(), &stride, &offset);
	RENDERER_CTX->IASetIndexBuffer(m_ibuff.Get(), DXGI_FORMAT_R32_UINT, offset);
}

void mesh::draw() const
{
	RENDERER_CTX->DrawIndexed((UINT)m_indices.size(), 0, 0);
}

void mesh::setup_mesh()
{
	HRESULT hr;

	// create the vertex buffer
	D3D11_BUFFER_DESC bdesc = {};
	bdesc.ByteWidth = (UINT)(sizeof(vertex) * m_vertices.size());
	bdesc.Usage = D3D11_USAGE_DEFAULT;
	bdesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bdesc.CPUAccessFlags = 0;
	bdesc.MiscFlags = 0;
	bdesc.StructureByteStride = sizeof(vertex);

	D3D11_SUBRESOURCE_DATA bdata = {};
	bdata.pSysMem = m_vertices.data();
	RENDERER_THROW_FAILED(RENDERER_DEV->CreateBuffer(&bdesc, &bdata, &m_vbuff));

	// create the index buffer
	bdesc.ByteWidth = (UINT)(sizeof(UINT) * m_indices.size());
	bdesc.Usage = D3D11_USAGE_DEFAULT;
	bdesc.BindFlags = D3D11_BIND_INDEX_BUFFER;
	bdesc.CPUAccessFlags = 0;
	bdesc.MiscFlags = 0;
	bdesc.StructureByteStride = sizeof(UINT);

	bdata.pSysMem = m_indices.data();
	RENDERER_THROW_FAILED(RENDERER_DEV->CreateBuffer(&bdesc, &bdata, &m_ibuff));
}

tri::tri()
{
	m_vertices = {
		{vec2( 0.0f,  0.5f)},
		{vec2( 0.5f, -0.5f)},
		{vec2(-0.5f, -0.5f)}
	};

	m_indices = {
		0, 1, 2
	};

	this->setup_mesh();
}

quad::quad()
{
	m_vertices = {
		{vec2(-0.5f,  0.5f)},
		{vec2( 0.5f,  0.5f)},
		{vec2( 0.5f, -0.5f)},
		{vec2(-0.5f, -0.5f)},
	};

	m_indices = {
		0, 1, 3,
		1, 2, 3
	};

	this->setup_mesh();
}

reg_polygon::reg_polygon(UINT vertices)
{
	vertices = vertices > 2 ? vertices : 3;
	float theta = tau / vertices;
	m_vertices.emplace_back(vec2());
	
	iqvec vertex(0.5, 0.0f, 0.0f, 0.0f);
	m_vertices.emplace_back(vertex.store2());
	iqmat transform = iqmat::rotation_z(theta);

	for (UINT i = 1; i < vertices; i++) {
		vertex.transform(transform, iqvec::usage::POINT);
		m_vertices.emplace_back(vertex.store2());
	}

	for (UINT i = 1; i < vertices; i++) {
		m_indices.emplace_back(i);
		m_indices.emplace_back(0);
		m_indices.emplace_back(i + 1);
	}
	// add the last triangle
	m_indices.emplace_back((UINT)m_vertices.size() - 1);
	m_indices.emplace_back(0);
	m_indices.emplace_back(1);

	this->setup_mesh();
}
