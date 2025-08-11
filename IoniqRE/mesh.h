#pragma once

#include "ioniq_windows.h"
#include <d3d11.h>
#include <wrl.h>

#include <vector>

#include "core.h"
#include "iqmath.h"

struct vertex
{
	vec3 pos;
	vertex(const vec3& pos) : pos(pos) {}
	vertex() = default;

	static constexpr D3D11_INPUT_ELEMENT_DESC vertex_layout = { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA };
	static D3D11_INPUT_ELEMENT_DESC const* get_vertex_layout() {
		return &vertex_layout;
	}
};

class mesh
{
public:
	// used for different ray intersection algorithms
	enum type
	{
		INVALID = -1,
		TRIANGLES,
		SPHERES,
		NUMTYPES
	};

public:
	mesh(const std::vector<vertex>& verts, const std::vector<UINT>& ids);
	mesh(type t = type::TRIANGLES);
	virtual ~mesh();

	inline const std::vector<vertex>& get_vertices() const { return m_vertices; }
	inline const std::vector<UINT>& get_indices()    const { return m_indices; }
	inline type get_type() const { return m_mesh_type; }

	void bind() const;
	void draw() const;

protected:
	void setup_mesh();

protected:
	Microsoft::WRL::ComPtr<ID3D11Buffer> m_vbuff;
	Microsoft::WRL::ComPtr<ID3D11Buffer> m_ibuff;

	std::vector<vertex> m_vertices;
	std::vector<UINT>   m_indices;
	type m_mesh_type;

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

class cube : public mesh
{
public:
	cube();
};
