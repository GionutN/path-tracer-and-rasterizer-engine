#include "renderer_base.h"

#include <sstream>

static renderer_base* g_engine_base = nullptr;

void renderer_base::init(const ref<window>& wnd)
{
	if (!g_engine_base) {
		g_engine_base = new renderer_base(wnd);
	}
}

void renderer_base::shutdown()
{
	if (g_engine_base) {
		delete g_engine_base;
		g_engine_base = nullptr;
	}
}

renderer_base* renderer_base::get()
{
	return g_engine_base;
}

renderer_base::renderer_base(const ref<window>& wnd)
	:
	m_wnd(wnd),
	m_clear(iqvec(0.0f, 0.0f, 0.0f, 1.0f)),
	m_output_format(DXGI_FORMAT_B8G8R8A8_UNORM)
{
	HRESULT hr;

	DXGI_SWAP_CHAIN_DESC swdesc = {};
	swdesc.BufferDesc.Width  = window::width;
	swdesc.BufferDesc.Height = window::height;
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
		&swdesc, &m_swchain,
		&m_device,
		&version,
		&m_imctx));

	if (version < D3D_FEATURE_LEVEL_11_0) {
		throw RENDERER_CUSTOMEXCEPTION("Direct3D 11 not supported");
	}
}

renderer_base::hr_exception::hr_exception(int line, const std::string& file, HRESULT hr, const std::string& custom_desc)
	:
	ioniq_exception(line, file),
	_hr(hr),
	_custom_desc(custom_desc)
{

}

const char* renderer_base::hr_exception::what() const
{
	std::ostringstream oss;
	oss << get_type() << std::endl;
	oss << "[Error Code]: " << std::hex << _hr << std::dec << std::endl;
	oss << "[Description]: " << get_description() << std::endl;
	oss << get_origin();
	m_what_buffer = oss.str();
	return m_what_buffer.c_str();
}

std::string renderer_base::hr_exception::get_description() const
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

renderer_base::cuda_exception::cuda_exception(int line, const std::string& file, cudaError err)
	:
	ioniq_exception(line, file),
	_err(err)
{

}

const char* renderer_base::cuda_exception::what() const
{
	std::ostringstream oss;
	oss << get_type() << std::endl;
	oss << "[Error Code]: " << _err << std::endl;
	oss << "[Name]: " << cudaGetErrorName(_err) << std::endl;
	oss << "[Description]: " << cudaGetErrorString(_err) << std::endl;
	oss << get_origin();
	m_what_buffer = oss.str();
	return m_what_buffer.c_str();
}
