#pragma once

#include "ioniq_windows.h"
#include <d3d11.h>
#include <wrl.h>

#include <string>

class shader
{
public:
	shader(const std::wstring& vertex_path, const std::wstring& pixel_path);
	shader() = default;
	~shader() = default;

	void bind() const;

private:
	Microsoft::WRL::ComPtr<ID3D11VertexShader> m_vshader;
	Microsoft::WRL::ComPtr<ID3D11PixelShader> m_pshader;
	Microsoft::WRL::ComPtr<ID3D11InputLayout> m_layout;

};
