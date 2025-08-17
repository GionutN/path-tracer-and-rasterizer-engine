#include "mesh.h"

#include "renderer.h"

mesh::mesh(const std::vector<vertex>& verts, const std::vector<UINT>& ids)
	:
	m_vertices(verts),
	m_indices(ids)
{
	this->setup_mesh();
}

mesh::mesh(type t)
	:
	m_mesh_type(t)
{
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
	vec3 normal(0.0f, 0.0f, -1.0f);
	m_vertices = {
		{ vec3( 0.0f,  0.5f, 0.0f), normal },
		{ vec3( 0.5f, -0.5f, 0.0f), normal },
		{ vec3(-0.5f, -0.5f, 0.0f), normal }
	};

	m_indices = {
		0, 1, 2
	};

	this->setup_mesh();
}

quad::quad()
{
	vec3 normal(0.0f, 0.0f, -1.0f);
	m_vertices = {
		{ vec3(-0.5f, -0.5f, 0.0f), normal },
		{ vec3( 0.5f, -0.5f, 0.0f), normal },
		{ vec3( 0.5f,  0.5f, 0.0f), normal },
		{ vec3(-0.5f,  0.5f, 0.0f), normal }
	};

	m_indices = {
		0, 3, 1,
		1, 3, 2
	};

	this->setup_mesh();
}

reg_polygon::reg_polygon(UINT vertices)
{
	// this builds the polygon just like the one made of the n-th roots of unity
	vertices = vertices > 2 ? vertices : 3;
	float theta = tau / vertices;
	vec3 normal(0.0f, 0.0f, -1.0f);
	m_vertices.emplace_back(vec3(), normal);
	
	iqvec vertex(0.5f, 0.0f, 0.0f, 0.0f);
	m_vertices.emplace_back(vertex.store3(), normal);
	iqmat transform = iqmat::rotation_z(theta);

	for (UINT i = 1; i < vertices; i++) {
		vertex.transform(transform, iqvec::usage::POINT);
		m_vertices.emplace_back(vertex.store3(), normal);
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

cube::cube()
{
	m_vertices = {
		// -Z (back)
		{vec3(-0.5f, -0.5f, -0.5f), vec3(0.0f, 0.0f, -1.0f)},	// a 0
		{vec3( 0.5f, -0.5f, -0.5f), vec3(0.0f, 0.0f, -1.0f)},	// b 1
		{vec3( 0.5f,  0.5f, -0.5f), vec3(0.0f, 0.0f, -1.0f)},	// c 2
		{vec3(-0.5f,  0.5f, -0.5f), vec3(0.0f, 0.0f, -1.0f)},	// d 3

		// +Z (front)
		{vec3(-0.5f, -0.5f,  0.5f), vec3(0.0f, 0.0f, 1.0f)},	// a' 4
		{vec3( 0.5f, -0.5f,  0.5f), vec3(0.0f, 0.0f, 1.0f)},	// b' 5
		{vec3( 0.5f,  0.5f,  0.5f), vec3(0.0f, 0.0f, 1.0f)},	// c' 6 
		{vec3(-0.5f,  0.5f,  0.5f), vec3(0.0f, 0.0f, 1.0f)},	// d' 7

		// -X (left)
		{vec3(-0.5f, -0.5f,  0.5f), vec3(-1.0f, 0.0f, 0.0f)},	// a' 4
		{vec3(-0.5f,  0.5f, -0.5f), vec3(-1.0f, 0.0f, 0.0f)},	// d 3
		{vec3(-0.5f, -0.5f, -0.5f), vec3(-1.0f, 0.0f, 0.0f)},	// a 0
		{vec3(-0.5f,  0.5f,  0.5f), vec3(-1.0f, 0.0f, 0.0f)},	// d' 7

		// +X (right)
		{vec3(0.5f, -0.5f, -0.5f), vec3(1.0f, 0.0f, 0.0f)},	// b 1
		{vec3(0.5f,  0.5f,  0.5f), vec3(1.0f, 0.0f, 0.0f)},	// c' 6 
		{vec3(0.5f, -0.5f,  0.5f), vec3(1.0f, 0.0f, 0.0f)},	// b' 5
		{vec3(0.5f,  0.5f, -0.5f), vec3(1.0f, 0.0f, 0.0f)},	// c 2

		// -Y (bottom)
		{vec3(-0.5f, -0.5f,  0.5f), vec3(0.0f, -1.0f, 0.0f)},	// a' 4
		{vec3( 0.5f, -0.5f, -0.5f), vec3(0.0f, -1.0f, 0.0f)},	// b 1
		{vec3( 0.5f, -0.5f,  0.5f), vec3(0.0f, -1.0f, 0.0f)},	// b' 5
		{vec3(-0.5f, -0.5f, -0.5f), vec3(0.0f, -1.0f, 0.0f)},	// a 0

		// +Y (top)
		{vec3(-0.5f,  0.5f, -0.5f), vec3(0.0f, 1.0f, 0.0f)},	// d 3
		{vec3( 0.5f,  0.5f,  0.5f), vec3(0.0f, 1.0f, 0.0f)},	// c' 6 
		{vec3( 0.5f,  0.5f, -0.5f), vec3(0.0f, 1.0f, 0.0f)},	// c 2
		{vec3(-0.5f,  0.5f,  0.5f), vec3(0.0f, 1.0f, 0.0f)},	// d' 7
	};

	m_indices = {
		// -Z (back)
		0,2,1,  0,3,2,
		// +Z (front)
		5,7,4,  5,6,7,
		// -X (left)
		8,9,10,  8,11,9,
		// +X (right)
		12,13,14,  12,15,13,
		// -Y (bottom)
		16,17,18,  16,19,17,
		// +Y (top)
		20,21,22,  20,23,21
	};

	this->setup_mesh();
}

// TODO:
// check if the mesh is correct under lighting
uv_sphere::uv_sphere(bool flat, UINT segments, UINT rings, mesh::type t)
{
	// y = sin(theta)
	// x = cos(theta) * cos(phi)
	// z = cos(theta) * sin(phi)

	// TODO: add mesh data for a flat shaded uv sphere

	// segments is the number of vertices in the xOy plane
	// rings is the number of segments along the z axis (rings + 1 vertices in the z axis direction)
	segments = segments > 2 ? segments : 3;
	rings = rings > 2 ? rings : 3;

	const float theta = pi / rings;		// polar angle
	const float phi = tau / segments;	// azimuthal angle

	const iqvec bottom(0.0f, -1.0f, 0.0f, 1.0f);
	const iqvec top(0.0f, 1.0f, 0.0f, 1.0f);
	const iqmat polar_tr = iqmat::rotation_z(theta);
	const iqmat azimuthal_tr = iqmat::rotation_y(phi);

	// build the sphere bottom -> top (-y to +y)
	iqvec crt_polar = bottom;
	for (UINT i = 1; i < rings; i++) {
		crt_polar.transform(polar_tr, iqvec::usage::POINT);
		m_vertices.emplace_back(crt_polar.store3(), crt_polar.store3());

		// at each ring, rotate around the y axis to get all the segments
		iqvec crt_azimuthal = crt_polar;
		for (UINT j = 1; j < segments; j++) {
			crt_azimuthal.transform(azimuthal_tr, iqvec::usage::POINT);
			m_vertices.emplace_back(crt_azimuthal.store3(), crt_azimuthal.store3());
		}
	}

	// finally add the top and bottom
	m_vertices.emplace_back(bottom.store3(), bottom.store3());
	m_vertices.emplace_back(top.store3(), top.store3());

	// all rings beside the top and bottom ones are made of quads
	// build them first
	for (UINT i = 0; i < rings - 2; i++) {
		for (UINT j = 0; j < segments - 1; j++) {
			m_indices.emplace_back(i * segments + j);
			m_indices.emplace_back((i + 1) * segments + j + 1);
			m_indices.emplace_back(i * segments + j + 1);

			m_indices.emplace_back(i * segments + j);
			m_indices.emplace_back((i + 1) * segments + j);
			m_indices.emplace_back((i + 1) * segments + j + 1);
		}

		// add the last quad after the loop to avoid working with modulos
		m_indices.emplace_back((i + 1) * segments - 1);
		m_indices.emplace_back((i + 1) * segments);
		m_indices.emplace_back(i * segments);

		m_indices.emplace_back((i + 1) * segments - 1);
		m_indices.emplace_back((i + 2) * segments - 1);
		m_indices.emplace_back((i + 1) * segments);
	}

	UINT top_idx = (UINT)m_vertices.size() - 1;
	UINT bottom_idx = top_idx - 1;
	// add the last 2 rings, made of triangles
	for (UINT i = 0; i < segments - 1; i++) {
		// the bottom ring
		m_indices.emplace_back(bottom_idx);
		m_indices.emplace_back(i);
		m_indices.emplace_back(i + 1);

		// the top ring
		m_indices.emplace_back(top_idx);
		m_indices.emplace_back(m_vertices.size() - i - 3);
		m_indices.emplace_back(m_vertices.size() - i - 4);
	}

	// add the last triangles in both of the rings to avoid modulos
	m_indices.emplace_back(bottom_idx);
	m_indices.emplace_back(segments - 1);
	m_indices.emplace_back(0);

	m_indices.emplace_back(top_idx);
	m_indices.emplace_back(m_vertices.size() - segments - 2);
	m_indices.emplace_back(m_vertices.size() - 3);

	this->setup_mesh();
}
