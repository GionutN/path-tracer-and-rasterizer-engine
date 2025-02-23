#pragma once

#include "ioniq_windows.h"
#include <d3d11.h>
#include <wrl.h>

#include <vector>

#include "core.h"
#include "iqmath.h"

struct vertex
{
	vec2 pos;
	vertex(const vec2& pos) : pos(pos) {}
	vertex() = default;

	static constexpr D3D11_INPUT_ELEMENT_DESC vertex_layout = { "POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA };
	static D3D11_INPUT_ELEMENT_DESC const* get_vertex_layout() {
		return &vertex_layout;
	}
};

class mesh
{
public:
	mesh(const std::vector<vertex>& verts, const std::vector<UINT>& ids);
	mesh() = default;
	virtual ~mesh();

	void bind() const;
	void draw() const;

protected:
	void setup_mesh();

protected:
	Microsoft::WRL::ComPtr<ID3D11Buffer> m_vbuff;
	Microsoft::WRL::ComPtr<ID3D11Buffer> m_ibuff;

	std::vector<vertex> m_vertices;
	std::vector<UINT>   m_indices;

};

class tri : public mesh
{
public:
	tri();

};

class quad : public mesh
{
public:
	quad();
};

class reg_polygon : public mesh
{
public:
	reg_polygon(UINT vertices);
};
