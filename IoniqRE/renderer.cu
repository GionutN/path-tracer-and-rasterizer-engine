#include "renderer.h"

#include <sstream>

#include <d3dcompiler.h>

static renderer* g_renderer;

namespace wrl = Microsoft::WRL;

void renderer::init(HWND hWnd)
{
	if (!g_renderer) {
		g_renderer = new renderer(hWnd);
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
	hr = m_swchain->Present(1, 0);	// 1 for vsync;
	if (hr == DXGI_ERROR_DEVICE_REMOVED) {
		throw RENDERER_EXCEPTION(m_device->GetDeviceRemovedReason());
	}
}

void renderer::draw_triangle()
{
	// bind the vertex buffer
	const UINT stride = sizeof(Vertex), offset = 0;
	m_imctx->IASetVertexBuffers(0, 1, vb.GetAddressOf(), &stride, &offset);

	// bind the vertex shader
	m_imctx->VSSetShader(vs.Get(), nullptr, 0);

	m_imctx->Draw(3, 0);
}

void renderer::set_triangle()
{
	HRESULT hr;

	// basic vertex structure and data
	const Vertex verts[3] = {
		{ 0.0,  0.5},
		{ 0.5, -0.5},
		{-0.5, -0.5}
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

	//create vertex shader
	wrl::ComPtr<ID3DBlob> blob;
	RENDERER_THROW_FAILED(D3DReadFileToBlob(L"vertex_shader.cso", &blob));
	RENDERER_THROW_FAILED(m_device->CreateVertexShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &vs));
}

renderer::renderer(HWND hWnd)
{
	m_clear[0] = 0.0;
	m_clear[1] = 0.0;
	m_clear[2] = 0.0;
	m_clear[3] = 1.0;

	HRESULT hr;

	DXGI_SWAP_CHAIN_DESC swdesc = {};
	swdesc.BufferDesc.Width = 0;
	swdesc.BufferDesc.Height = 0;
	swdesc.BufferDesc.RefreshRate.Numerator = 60;
	swdesc.BufferDesc.RefreshRate.Denominator = 1;
	swdesc.BufferDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;	// use this instead of rgba for automatic color space conversion (sRGB)
	swdesc.BufferDesc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;
	swdesc.BufferDesc.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;
	swdesc.SampleDesc.Count = 1;
	swdesc.SampleDesc.Quality = 0;
	swdesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	swdesc.BufferCount = 2;	// triple-buffering, at least 2 for swap_effect_flip_x
	swdesc.OutputWindow = hWnd;
	swdesc.Windowed = TRUE;
	// TODO: add multi-sample antialiasing by first rendering to a multisampled texture
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

	wrl:: ComPtr<ID3D11Resource> back_buffer;
	RENDERER_THROW_FAILED(m_swchain->GetBuffer(0, __uuidof(ID3D11Resource), &back_buffer));
	RENDERER_THROW_FAILED(m_device->CreateRenderTargetView(back_buffer.Get(), nullptr, &m_target));

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
	oss << "[Error Code]: " << _hr << std::endl;
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
