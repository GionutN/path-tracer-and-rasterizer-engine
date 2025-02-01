#pragma once

#include "ioniq_windows.h"
#include <d3d11.h>
#include <wrl.h>

#include <vector>

#include "core.h"

struct vertex
{
	real x;
	real y;
};

class mesh
{
public:
	mesh(const std::vector<vertex>& verts, const std::vector<UINT>& ids);
	mesh() {}
	virtual ~mesh();

	ID3D11Buffer* const* get_vertex_buffer() const { return m_vbuff.GetAddressOf(); }
	ID3D11Buffer* get_index_buffer()  const { return m_ibuff.Get(); }
	UINT get_num_indices() const { return (UINT)m_indices.size(); }

protected:
	void setup_mesh();

protected:
	Microsoft::WRL::ComPtr<ID3D11Buffer> m_vbuff;
	Microsoft::WRL::ComPtr<ID3D11Buffer> m_ibuff;

	std::vector<vertex> m_vertices;
	std::vector<UINT>   m_indices;

};

class triangle : public mesh
{
public:
	triangle();

};

class quad : public mesh
{
public:
	quad();
};
