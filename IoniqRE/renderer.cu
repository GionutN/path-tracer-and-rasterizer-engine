#include "renderer.h"

#include <d3dcompiler.h>

#include <sstream>

#include "mesh.h"

static renderer* g_renderer;

namespace wrl = Microsoft::WRL;

void renderer::init(const ref<window>& wnd)
{
	if (!g_renderer) {
		g_renderer = new renderer(wnd);
	}
}

void renderer::shutdown()
{
	if (g_renderer) {
		delete g_renderer;
		g_renderer = nullptr;
	}
}

renderer* renderer::get()
{
	return g_renderer;
}

void renderer::begin_frame()
{
	float clear[4];
	clear[0] = (float)m_clear[0];
	clear[1] = (float)m_clear[1];
	clear[2] = (float)m_clear[2];
	clear[3] = (float)m_clear[3];

	m_imctx->ClearRenderTargetView(m_target.Get(), clear);
}

void renderer::end_frame()
{
	HRESULT hr;
	// copy from a multi-sampled texture to a non-sampled texture
	m_imctx->ResolveSubresource(m_nonmsaa_intermediate_texture.Get(), 0, m_msaa_target_texture.Get(), 0, m_output_format);

	// copy to the back buffer for presenting
	wrl::ComPtr<ID3D11Texture2D> back_buffer;
	RENDERER_THROW_FAILED(m_swchain->GetBuffer(0, __uuidof(ID3D11Texture2D), &back_buffer));
	m_imctx->CopyResource(back_buffer.Get(), m_nonmsaa_intermediate_texture.Get());

	hr = m_swchain->Present(1, 0);	// 1 for vsync;
	if (hr == DXGI_ERROR_DEVICE_REMOVED) {
		throw RENDERER_EXCEPTION(m_device->GetDeviceRemovedReason());
	}
}

void renderer::draw_scene(const mesh& mesh)
{
	// bind the vertex shader
	m_imctx->VSSetShader(vs.Get(), nullptr, 0);
	// set the input layout
	m_imctx->IASetInputLayout(layout.Get());

	// bind the pixel shader
	m_imctx->PSSetShader(ps.Get(), nullptr, 0);
	// set the render target
	m_imctx->OMSetRenderTargets(1, m_target.GetAddressOf(), nullptr);

	//m_imctx->Draw(3, 0);
	//m_imctx->DrawIndexed(6, 0, 0);

	this->bind_draw_mesh(mesh);
}

void renderer::bind_mesh(const mesh& mesh)
{
	UINT stride = sizeof(vertex), offset = 0;
	m_imctx->IASetVertexBuffers(0, 1, mesh.get_vertex_buffer(), &stride, &offset);
	m_imctx->IASetIndexBuffer(mesh.get_index_buffer(), DXGI_FORMAT_R32_UINT, offset);
}

void renderer::draw_mesh(const mesh& mesh)
{
	m_imctx->DrawIndexed(mesh.get_num_indices(), 0, 0);
}

void renderer::bind_draw_mesh(const mesh& mesh)
{
	this->bind_mesh(mesh);
	this->draw_mesh(mesh);
}

void renderer::set_triangle()
{
	HRESULT hr;

	// basic vertex structure and data
	// ALWAYS IN CLOCKWISE ORDER
	const Vertex verts[4] = {
		{ -0.5f,  0.5f },
		{  0.5f,  0.5f },
		{  0.5f, -0.5f },
		{ -0.5f, -0.5f }
	};

	// create the vertex buffer
	D3D11_BUFFER_DESC vbdesc = {};
	vbdesc.ByteWidth = sizeof(verts);
	vbdesc.Usage = D3D11_USAGE_DEFAULT;
	vbdesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	vbdesc.CPUAccessFlags = 0;
	vbdesc.MiscFlags = 0;
	vbdesc.StructureByteStride = sizeof(Vertex);

	D3D11_SUBRESOURCE_DATA vbdata = {};
	vbdata.pSysMem = verts;
	RENDERER_THROW_FAILED(m_device->CreateBuffer(&vbdesc, &vbdata, &vb));

	const UINT indices[6] = {
		0, 1, 3,
		1, 2, 3
	};
	// create the index buffer
	D3D11_BUFFER_DESC ibdesc = {};
	ibdesc.ByteWidth = sizeof(indices);
	ibdesc.Usage = D3D11_USAGE_DEFAULT;
	ibdesc.BindFlags = D3D11_BIND_INDEX_BUFFER;
	ibdesc.CPUAccessFlags = 0;
	ibdesc.MiscFlags = 0;
	ibdesc.StructureByteStride = sizeof(UINT);

	D3D11_SUBRESOURCE_DATA ibdata = {};
	ibdata.pSysMem = indices;
	RENDERER_THROW_FAILED(m_device->CreateBuffer(&ibdesc, &ibdata, &ib));

	// set primitive topology
	m_imctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	//create pixel shader
	wrl::ComPtr<ID3DBlob> blob;
	RENDERER_THROW_FAILED(D3DReadFileToBlob(L"pixel_shader.cso", &blob));
	RENDERER_THROW_FAILED(m_device->CreatePixelShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &ps));

	//create vertex shader
	RENDERER_THROW_FAILED(D3DReadFileToBlob(L"vertex_shader.cso", &blob));
	RENDERER_THROW_FAILED(m_device->CreateVertexShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &vs));

	// create the input layouts
	D3D11_INPUT_ELEMENT_DESC indesc[] = {
		{"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA }
	};
	RENDERER_THROW_FAILED(m_device->CreateInputLayout(indesc, (UINT)std::size(indesc), blob->GetBufferPointer(), blob->GetBufferSize(), &layout));
}

renderer::renderer(const ref<window>& wnd)
	:
	m_wnd(wnd),
	m_samples(8),
	m_output_format(DXGI_FORMAT_B8G8R8A8_UNORM)
{
	m_clear[0] = 0.0;
	m_clear[1] = 0.0;
	m_clear[2] = 0.0;
	m_clear[3] = 1.0;

	HRESULT hr;

	DXGI_SWAP_CHAIN_DESC swdesc = {};
	swdesc.BufferDesc.Width = m_wnd->width;
	swdesc.BufferDesc.Height = m_wnd->height;
	swdesc.BufferDesc.RefreshRate.Numerator = 60;
	swdesc.BufferDesc.RefreshRate.Denominator = 1;
	swdesc.BufferDesc.Format = m_output_format;	// use BGRA8 instead of RGBA8 for automatic color space conversion (sRGB)
	swdesc.BufferDesc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;
	swdesc.BufferDesc.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;
	swdesc.SampleDesc.Count = 1;
	swdesc.SampleDesc.Quality = 0;
	swdesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	swdesc.BufferCount = 2;	// triple-buffering, at least 2 for swap_effect_flip_x
	swdesc.OutputWindow = m_wnd->get_handle();
	swdesc.Windowed = TRUE;
	swdesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;	// this does NOT support msaa directly on the back buffer
	swdesc.Flags = 0;

	UINT device_flags = D3D11_CREATE_DEVICE_SINGLETHREADED;
#ifdef _DEBUG
	device_flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
	D3D_FEATURE_LEVEL version;	// used for d3d11 support check
	RENDERER_THROW_FAILED(D3D11CreateDeviceAndSwapChain(nullptr, 
		D3D_DRIVER_TYPE_HARDWARE, nullptr, 
		device_flags, 
		nullptr, 0, 
		D3D11_SDK_VERSION, 
		&swdesc, &m_swchain ,
		&m_device, 
		&version, 
		&m_imctx));

	if (version < D3D_FEATURE_LEVEL_11_0) {
		throw RENDERER_CUSTOMEXCEPTION("Direct3D 11 not supported");
	}

	// check for multisampling support
	RENDERER_THROW_FAILED(m_device->CheckMultisampleQualityLevels(m_output_format, m_samples, &m_quality));
	m_quality--;	// always use the maximul quality level, quality - 1

	// create the texture for msaa rendering
	D3D11_TEXTURE2D_DESC msaa_tex_desc = {};
	msaa_tex_desc.Width = m_wnd->width;
	msaa_tex_desc.Height = m_wnd->height;
	msaa_tex_desc.MipLevels = 1;	// multi-sampled
	msaa_tex_desc.ArraySize = 1;
	msaa_tex_desc.Format = m_output_format;
	msaa_tex_desc.SampleDesc.Count = m_samples;
	msaa_tex_desc.SampleDesc.Quality = m_quality;
	msaa_tex_desc.Usage = D3D11_USAGE_DEFAULT;
	msaa_tex_desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
	msaa_tex_desc.CPUAccessFlags = 0;
	msaa_tex_desc.MiscFlags = 0;
	RENDERER_THROW_FAILED(m_device->CreateTexture2D(&msaa_tex_desc, nullptr, &m_msaa_target_texture));

	// make a non multi-sampled texture
	D3D11_TEXTURE2D_DESC nonmsaa_tex_desc = {};
	nonmsaa_tex_desc.Width = m_wnd->width;
	nonmsaa_tex_desc.Height = m_wnd->height;
	nonmsaa_tex_desc.MipLevels = 1;
	nonmsaa_tex_desc.ArraySize = 1;
	nonmsaa_tex_desc.Format = m_output_format;
	nonmsaa_tex_desc.SampleDesc.Count = 1;	// non multisampled
	nonmsaa_tex_desc.SampleDesc.Quality = 0;
	nonmsaa_tex_desc.Usage = D3D11_USAGE_DEFAULT;
	nonmsaa_tex_desc.BindFlags = 0;
	nonmsaa_tex_desc.CPUAccessFlags = 0;
	nonmsaa_tex_desc.MiscFlags = 0;
	RENDERER_THROW_FAILED(m_device->CreateTexture2D(&nonmsaa_tex_desc, nullptr, &m_nonmsaa_intermediate_texture));

	// set the description for the render target view
	D3D11_RENDER_TARGET_VIEW_DESC rtv_desc = {};
	rtv_desc.Format = msaa_tex_desc.Format;
	rtv_desc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DMS;
	rtv_desc.Texture2D.MipSlice = 0;
	RENDERER_THROW_FAILED(m_device->CreateRenderTargetView(m_msaa_target_texture.Get(), &rtv_desc, &m_target));

	// set the viewport
	D3D11_VIEWPORT vport = {};
	vport.TopLeftX = 0.0f;
	vport.TopLeftY = 0.0f;
	vport.Width = (FLOAT)m_wnd->width;
	vport.Height = (FLOAT)m_wnd->height;
	vport.MinDepth = 0.0f;
	vport.MaxDepth = 1.0f;
	m_imctx->RSSetViewports(1, &vport);

	set_triangle();
}

renderer::exception::exception(int line, const std::string& file, HRESULT hr, const std::string& custom_desc)
	:
	ioniq_exception(line, file),
	_hr(hr),
	_custom_desc(custom_desc)
{

}

const char* renderer::exception::what() const
{
	std::ostringstream oss;
	oss << get_type() << std::endl;
	oss << "[Error Code]: " << std::hex << _hr << std::dec << std::endl;
	oss << "[Description]: " << get_description() << std::endl;
	oss << get_origin();
	m_what_buffer = oss.str();
	return m_what_buffer.c_str();
}

std::string renderer::exception::get_description() const
{
	if (_hr == 0) {
		return _custom_desc;
	}

	char* msg = nullptr;
	DWORD len = FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
		nullptr, _hr, 0, reinterpret_cast<LPSTR>(&msg), 0, nullptr);
	if (len == 0) {
		return "Unknown error code";
	}

	std::string out = msg;
	LocalFree(msg);
	return out;
}
