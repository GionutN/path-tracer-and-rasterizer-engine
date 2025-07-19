#pragma once

#include "ioniq_windows.h"
#include <d3d11.h>
#include <wrl.h>

#include <string>

#include "iqmath.h"

class shader
{
public:
	shader(const std::wstring& vertex_path, const std::wstring& pixel_path);
	shader() = default;
	~shader() = default;

	void bind() const;
	void update_transform(const iqmat& tr);

private:
	Microsoft::WRL::ComPtr<ID3D11VertexShader> m_vshader;
	Microsoft::WRL::ComPtr<ID3D11PixelShader> m_pshader;
	Microsoft::WRL::ComPtr<ID3D11InputLayout> m_layout;
	Microsoft::WRL::ComPtr<ID3D11Buffer> m_transform_cbuffer;

	iqmat m_transform = 1.0f;

};
