#include "renderer.h"

#include <sstream>
#include <d3dcompiler.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "mesh.h"
#include "shader.h"

static renderer* g_renderer;

namespace wrl = Microsoft::WRL;

void renderer::init(const ref<window>& wnd)
{
	if (!g_renderer) {
		g_renderer = new renderer(wnd);
		RENDERER->m_background = std::make_shared<quad>();
		RENDERER->m_bg_shader = std::make_shared<shader>(L"bg_vert_shader.cso", L"bg_pixel_shader.cso");
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
	if (m_crt_engine != m_old_engine) {
		m_old_engine = m_crt_engine;
	}

	float clear[4];
	clear[0] = (float)m_clear[0];
	clear[1] = (float)m_clear[1];
	clear[2] = (float)m_clear[2];
	clear[3] = (float)m_clear[3];
	switch (m_old_engine) {
	case engine::RASTERIZER:
		m_imctx->OMSetRenderTargets(1, m_rttarget.GetAddressOf(), nullptr);
		m_imctx->ClearRenderTargetView(m_rttarget.Get(), clear);
		break;
	case engine::PATHTRACER:
		m_imctx->OMSetRenderTargets(1, m_pttarget.GetAddressOf(), nullptr);
		break;
	}
}

void renderer::end_frame()
{
	HRESULT hr;
	wrl::ComPtr<ID3D11Texture2D> back_buffer;

	switch (m_old_engine) {
	case engine::RASTERIZER:
		// copy from a multi-sampled texture to a non-multisampled texture
		m_imctx->ResolveSubresource(m_nonmsaa_intermediate_texture.Get(), 0, m_msaa_target_texture.Get(), 0, m_output_format);

		// copy to the back buffer for presenting
		RENDERER_THROW_FAILED(m_swchain->GetBuffer(0, __uuidof(ID3D11Texture2D), &back_buffer));
		m_imctx->CopyResource(back_buffer.Get(), m_nonmsaa_intermediate_texture.Get());
		break;
	case engine::PATHTRACER:
		// update the texture only when the pathtracer updates the pixel buffer
		if (m_ptimage_updated) {
			RENDERER_THROW_FAILED(m_imctx->Map(m_pttexture.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &m_mapped_texture));

			// setup parameteres for copying
			pixel* dest = (pixel*)(m_mapped_texture.pData);
			const size_t dest_pitch = m_mapped_texture.RowPitch / sizeof(pixel);
			const size_t src_pitch = m_wnd->width;
			const size_t row_bytes = src_pitch * sizeof(pixel);
			// perform the copy line-by-line
			for (size_t i = 0; i < m_wnd->height; i++)
			{
				memcpy(&dest[i * dest_pitch], &m_host_pixel_buffer[i * src_pitch], row_bytes);
			}
			// release the adapter memory
			m_imctx->Unmap(m_pttexture.Get(), 0);

			m_ptimage_updated = false;
		}

		// render offscreen scene texture to back buffer
		const UINT stride = 16, offset = 0;
		m_imctx->IASetVertexBuffers(0, 1, m_ptvertex_buffer.GetAddressOf(), &stride, &offset);
		m_imctx->IASetInputLayout(m_ptlayout.Get());
		m_imctx->VSSetShader(m_ptvertex_shader.Get(), nullptr, 0);
		m_imctx->PSSetShader(m_ptpixel_shader.Get(), nullptr, 0);
		m_imctx->PSSetShaderResources(0, 1, m_pttexture_view.GetAddressOf());
		m_imctx->PSSetSamplers(0, 1, m_pttexture_sampler.GetAddressOf());
		m_imctx->Draw(6, 0);
		break;
	}

	hr = m_swchain->Present(1, 0);	// 1 for vsync;
	if (hr == DXGI_ERROR_DEVICE_REMOVED) {
		throw RENDERER_EXCEPTION(m_device->GetDeviceRemovedReason());
	}
}

__global__ void render(renderer::pixel* fb, int width, int height)
{
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	if (y >= height || x >= width) {
		return;
	}
	int pixelid = y * width + x;
	fb[pixelid].r = (uint8_t)(256.0f * y / height);
	fb[pixelid].g = (uint8_t)(256.0f * x / width);
	fb[pixelid].b = 50;
	fb[pixelid].a = 255;
}

void renderer::draw_scene(const std::vector<mesh>& scene, const std::vector<shader>& shaders, float dt)
{
	cudaError cderr;
	static float time = 0.0f;
	time += dt;

	switch (m_old_engine) {
	case engine::RASTERIZER:
		m_background->bind();
		m_bg_shader->bind();
		m_background->draw();

		scene[0].bind();
		shaders[0].bind();
		scene[0].draw();
		break;
	case engine::PATHTRACER:
		// let the compute kernel run for half a second, then retrieve the computed image
		if (time > 0.5f) {
			time = 0.0f;
			UINT tx = 16, ty = 16;
			UINT num_pixels = m_wnd->height * m_wnd->width;

			dim3 blocks(m_wnd->width / tx + 1, m_wnd->height / ty + 1);
			dim3 threads(tx, ty);
			RENDERER_THROW_CUDA(cudaGetLastError());
			size_t fbsize = num_pixels * sizeof(pixel);
			RENDERER_THROW_CUDA(cudaDeviceSynchronize());	// synchronize before calling the kernel, so that it runs for half a second
			RENDERER_THROW_CUDA(cudaMemcpy(m_host_pixel_buffer, m_dev_pixel_buffer, fbsize, cudaMemcpyDeviceToHost));
			m_ptimage_updated = true;
			render<<<blocks, threads>>>(m_dev_pixel_buffer, m_wnd->width, m_wnd->height);
		}
		break;
	}
}

renderer::renderer(const ref<window>& wnd)
	:
	m_wnd(wnd),
	m_samples(8),
	m_output_format(DXGI_FORMAT_B8G8R8A8_UNORM),
	m_crt_engine(engine::PATHTRACER),
	m_old_engine(engine::PATHTRACER),
	m_ptimage_updated(true)
{
	m_clear[0] = 0.0f;
	m_clear[1] = 0.0f;
	m_clear[2] = 0.0f;
	m_clear[3] = 1.0f;

	HRESULT hr;
	cudaError cderr;

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

	RENDERER_THROW_FAILED(m_device->CheckMultisampleQualityLevels(m_output_format, m_samples, &m_quality));
	m_quality--;	// always use the maximum quality level, quality - 1

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
	RENDERER_THROW_FAILED(m_device->CreateRenderTargetView(m_msaa_target_texture.Get(), &rtv_desc, &m_rttarget));

	// set the viewport
	D3D11_VIEWPORT vport = {};
	vport.TopLeftX = 0.0f;
	vport.TopLeftY = 0.0f;
	vport.Width = (FLOAT)m_wnd->width;
	vport.Height = (FLOAT)m_wnd->height;
	vport.MinDepth = 0.0f;
	vport.MaxDepth = 1.0f;
	m_imctx->RSSetViewports(1, &vport);
	m_imctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	// now create the render target for the path-tracer
	// needed to output the final buffer
	wrl::ComPtr<ID3D11Texture2D> back_buffer;
	RENDERER_THROW_FAILED(m_swchain->GetBuffer(0, __uuidof(ID3D11Texture2D), &back_buffer));
	RENDERER_THROW_FAILED(m_device->CreateRenderTargetView(back_buffer.Get(), nullptr, &m_pttarget));

	D3D11_TEXTURE2D_DESC pttex_desc = {};
	pttex_desc.Width = m_wnd->width;
	pttex_desc.Height = m_wnd->height;
	pttex_desc.MipLevels = 1;
	pttex_desc.ArraySize = 1;
	pttex_desc.Format = m_output_format;
	pttex_desc.SampleDesc.Count = 1;	// non multisampled
	pttex_desc.SampleDesc.Quality = 0;
	pttex_desc.Usage = D3D11_USAGE_DYNAMIC;
	pttex_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	pttex_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	pttex_desc.MiscFlags = 0;
	RENDERER_THROW_FAILED(m_device->CreateTexture2D(&pttex_desc, nullptr, &m_pttexture));

	D3D11_SHADER_RESOURCE_VIEW_DESC pttex_view = {};
	pttex_view.Format = m_output_format;
	pttex_view.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
	pttex_view.Texture2D.MipLevels = 1;
	RENDERER_THROW_FAILED(m_device->CreateShaderResourceView(m_pttexture.Get(), &pttex_view, &m_pttexture_view));

	wrl::ComPtr<ID3DBlob> blob;
	RENDERER_THROW_FAILED(D3DReadFileToBlob(L"PTPixelShader.cso", &blob));
	RENDERER_THROW_FAILED(m_device->CreatePixelShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &m_ptpixel_shader));

	RENDERER_THROW_FAILED(D3DReadFileToBlob(L"PTVertexShader.cso", &blob));
	RENDERER_THROW_FAILED(m_device->CreateVertexShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &m_ptvertex_shader));

	D3D11_INPUT_ELEMENT_DESC pt_vertex_layout[2] = {
		{"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA},
		{"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 8, D3D11_INPUT_PER_VERTEX_DATA}
	};
	RENDERER_THROW_FAILED(m_device->CreateInputLayout(pt_vertex_layout, (UINT)std::size(pt_vertex_layout), blob->GetBufferPointer(), blob->GetBufferSize(), &m_ptlayout));

	const float vertices[] = {
		-1.0f,  1.0f, 0.0f, 0.0f,
		 1.0f,  1.0f, 1.0f, 0.0f,
		 1.0f, -1.0f, 1.0f, 1.0f,

		 1.0f, -1.0f, 1.0f, 1.0f,
		-1.0f, -1.0f, 0.0f, 1.0f,
		-1.0f,  1.0f, 0.0f, 0.0f
	};
	D3D11_BUFFER_DESC pt_bdesc = {};
	pt_bdesc.ByteWidth = (UINT)sizeof(vertices);
	pt_bdesc.Usage = D3D11_USAGE_DEFAULT;
	pt_bdesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	pt_bdesc.CPUAccessFlags = 0;
	pt_bdesc.MiscFlags = 0;
	pt_bdesc.StructureByteStride = 4 * (UINT)sizeof(float);

	D3D11_SUBRESOURCE_DATA pt_bdata = {};
	pt_bdata.pSysMem = vertices;
	RENDERER_THROW_FAILED(m_device->CreateBuffer(&pt_bdesc, &pt_bdata, &m_ptvertex_buffer));

	D3D11_SAMPLER_DESC smpl_desc = {};
	smpl_desc.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
	smpl_desc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
	smpl_desc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
	smpl_desc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
	smpl_desc.ComparisonFunc = D3D11_COMPARISON_NEVER;
	smpl_desc.MinLOD = 0.0f;
	smpl_desc.MaxLOD = D3D11_FLOAT32_MAX;
	RENDERER_THROW_FAILED(m_device->CreateSamplerState(&smpl_desc, &m_pttexture_sampler));

	size_t num_pixels = m_wnd->width * m_wnd->height;
	size_t fbsize = num_pixels * sizeof(pixel);
	RENDERER_THROW_CUDA(cudaMalloc((void**)&m_dev_pixel_buffer, fbsize));
	m_host_pixel_buffer = new pixel[num_pixels];
	memset(m_host_pixel_buffer, 0, fbsize);
	cudaMemset(m_dev_pixel_buffer, 0, fbsize);
}

renderer::~renderer()
{
	cudaError cderr;

	RENDERER_THROW_CUDA(cudaDeviceSynchronize());
	if (m_dev_pixel_buffer) {
		RENDERER_THROW_CUDA(cudaFree(m_dev_pixel_buffer));
		m_dev_pixel_buffer = nullptr;
	}
	if (m_host_pixel_buffer) {
		delete[] m_host_pixel_buffer;
		m_host_pixel_buffer = nullptr;
	}
	cudaDeviceReset();
}

renderer::hr_exception::hr_exception(int line, const std::string& file, HRESULT hr, const std::string& custom_desc)
	:
	ioniq_exception(line, file),
	_hr(hr),
	_custom_desc(custom_desc)
{

}

const char* renderer::hr_exception::what() const
{
	std::ostringstream oss;
	oss << get_type() << std::endl;
	oss << "[Error Code]: " << std::hex << _hr << std::dec << std::endl;
	oss << "[Description]: " << get_description() << std::endl;
	oss << get_origin();
	m_what_buffer = oss.str();
	return m_what_buffer.c_str();
}

std::string renderer::hr_exception::get_description() const
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

renderer::cuda_exception::cuda_exception(int line, const std::string& file, cudaError err)
	:
	ioniq_exception(line, file),
	_err(err)
{
	
}

const char* renderer::cuda_exception::what() const
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
