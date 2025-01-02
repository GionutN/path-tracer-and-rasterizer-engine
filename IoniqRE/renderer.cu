#include "renderer.h"

static renderer* g_renderer;

using namespace Microsoft::WRL;

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
	float clear[4] = { 0.5f, 0.0f, 1.0f, 1.0f };
	m_imctx->ClearRenderTargetView(m_target.Get(), clear);
}

void renderer::end_frame()
{
	m_swchain->Present(1, 0);	// 1 for vsync;
}

renderer::renderer(HWND hWnd)
{
	DXGI_SWAP_CHAIN_DESC swdesc = {};
	swdesc.BufferDesc.Width = 0;
	swdesc.BufferDesc.Height = 0;
	swdesc.BufferDesc.RefreshRate.Numerator = 60;
	swdesc.BufferDesc.RefreshRate.Denominator = 1;
	swdesc.BufferDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;	// use this instead of rgba for automatic color space conversion (sRGB)
	swdesc.BufferDesc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;
	swdesc.BufferDesc.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;
	// TODO: add multi-sample antialiasing
	// for runtime change of the number of samples, the swap-chain must be destroyed and created with the new SampleDesc
	swdesc.SampleDesc.Count = 1;
	swdesc.SampleDesc.Quality = 0;
	swdesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	swdesc.BufferCount = 1;	// double-buffering
	swdesc.OutputWindow = hWnd;
	swdesc.Windowed = TRUE;
	swdesc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
	swdesc.Flags = 0;

	D3D_FEATURE_LEVEL version;	// used for d3d11 support check
	D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 
		D3D11_CREATE_DEVICE_SINGLETHREADED, nullptr, 0, 
		D3D11_SDK_VERSION, 
		&swdesc, &m_swchain ,&m_device, &version, &m_imctx);

	ComPtr<ID3D11Resource> back_buffer;
	m_swchain->GetBuffer(0, __uuidof(ID3D11Resource), &back_buffer);
	m_device->CreateRenderTargetView(back_buffer.Get(), nullptr, &m_target);
}
