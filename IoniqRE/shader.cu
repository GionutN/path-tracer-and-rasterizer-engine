#include "shader.h"

#include <d3dcompiler.h>

#include "renderer.h"

namespace wrl = Microsoft::WRL;

shader::shader(const std::wstring& vertex_path, const std::wstring& pixel_path)
{
	HRESULT hr;

	wrl::ComPtr<ID3DBlob> blob;
	RENDERER_THROW_FAILED(D3DReadFileToBlob(pixel_path.c_str(), &blob));
	RENDERER_THROW_FAILED(RENDERER_DEV->CreatePixelShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &m_pshader));

	RENDERER_THROW_FAILED(D3DReadFileToBlob(vertex_path.c_str(), &blob));
	RENDERER_THROW_FAILED(RENDERER_DEV->CreateVertexShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &m_vshader));

	RENDERER_THROW_FAILED(RENDERER_DEV->CreateInputLayout(vertex::get_vertex_layout(), 1, blob->GetBufferPointer(), blob->GetBufferSize(), &m_layout));
}

void shader::bind() const
{
	RENDERER_CTX->VSSetShader(m_vshader.Get(), nullptr, 0);
	RENDERER_CTX->IASetInputLayout(m_layout.Get());
	RENDERER_CTX->PSSetShader(m_pshader.Get(), nullptr, 0);
}
