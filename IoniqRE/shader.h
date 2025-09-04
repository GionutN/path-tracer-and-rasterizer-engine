#pragma once

#include "ioniq_windows.h"
#include <d3d11.h>
#include <wrl.h>

#include <string>

#include "iqmath.h"

class shader
{
public:
	struct cbuffer
	{
		mat3x3 normal_mat;
		iqmat model;
		iqmat view;
		iqmat projection;
	};

public:
	shader(const std::wstring& vertex_path, const std::wstring& pixel_path);
	shader() = default;
	~shader() = default;

	void bind() const;
	void update_transform(const iqmat& tr);
	void update_view_proj(const iqmat& view, const iqmat& proj);

private:
	Microsoft::WRL::ComPtr<ID3D11VertexShader> m_vshader;
	Microsoft::WRL::ComPtr<ID3D11PixelShader> m_pshader;
	Microsoft::WRL::ComPtr<ID3D11InputLayout> m_layout;
	Microsoft::WRL::ComPtr<ID3D11Buffer> m_transform_cbuffer;

	cbuffer cb = { 1.0f, 1.0f, 1.0f, 1.0f };

};
