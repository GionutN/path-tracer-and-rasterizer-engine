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

	RENDERER_THROW_FAILED(RENDERER_DEV->CreateInputLayout(vertex::get_vertex_layout(), 2, blob->GetBufferPointer(), blob->GetBufferSize(), &m_layout));

	D3D11_BUFFER_DESC cb_desc = {};
	cb_desc.ByteWidth = sizeof(cbuffer);
	cb_desc.Usage = D3D11_USAGE_DYNAMIC;
	cb_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	cb_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

	D3D11_SUBRESOURCE_DATA cb_subd = {};
	cb_subd.pSysMem = &(this->cb);
	RENDERER_THROW_FAILED(RENDERER_DEV->CreateBuffer(&cb_desc, &cb_subd, &m_transform_cbuffer));
}

void shader::bind() const
{
	// update the constant buffer before binding
	HRESULT hr;
	D3D11_MAPPED_SUBRESOURCE mapped;
	RENDERER_THROW_FAILED(RENDERER_CTX->Map(m_transform_cbuffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped));
	std::memcpy(mapped.pData, &cb, sizeof(shader::cbuffer));
	RENDERER_CTX->Unmap(m_transform_cbuffer.Get(), 0);

	RENDERER_CTX->VSSetShader(m_vshader.Get(), nullptr, 0);
	RENDERER_CTX->IASetInputLayout(m_layout.Get());
	RENDERER_CTX->PSSetShader(m_pshader.Get(), nullptr, 0);
	RENDERER_CTX->VSSetConstantBuffers(0, 1, m_transform_cbuffer.GetAddressOf());
}

void shader::update_transform(const iqmat& tr)
{
	// this needs to be transposed for whatever reason
	cb.model = tr.transposed();
}

void shader::update_view_proj(const iqmat& view, const iqmat& proj)
{
	// these must also be transposed
	cb.view = view.transposed();
	cb.projection = proj.transposed();
}
