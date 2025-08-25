#include "rasterizer.h"

#include <set>

static rasterizer* g_rasterizer = nullptr;

namespace wrl = Microsoft::WRL;

void rasterizer::init()
{
	if (!g_rasterizer) {
		g_rasterizer = new rasterizer();
		g_rasterizer->m_background = std::make_shared<quad>();
		g_rasterizer->m_bg_shader = std::make_shared<shader>(L"bg_vert_shader.cso", L"bg_pixel_shader.cso");
	}
}

void rasterizer::shutdown()
{
	if (g_rasterizer) {
		delete g_rasterizer;
		g_rasterizer = nullptr;
	}
}

rasterizer* rasterizer::get()
{
	return g_rasterizer;
}

rasterizer::rasterizer()
	:
	m_samples(4)
{
	HRESULT hr;
	renderer_base* rnd_base = RENDERER;

	RENDERER_THROW_FAILED(rnd_base->device()->CheckMultisampleQualityLevels(rnd_base->pixel_format(), m_samples, &m_quality));
	m_quality--;	// always use the maximum quality level, quality - 1

	// create the texture for msaa rendering
	D3D11_TEXTURE2D_DESC msaa_tex_desc = {};
	msaa_tex_desc.Width = window::width;
	msaa_tex_desc.Height = window::height;
	msaa_tex_desc.MipLevels = 1;	// multisampled
	msaa_tex_desc.ArraySize = 1;
	msaa_tex_desc.Format = rnd_base->pixel_format();
	msaa_tex_desc.SampleDesc.Count = m_samples;
	msaa_tex_desc.SampleDesc.Quality = m_quality;
	msaa_tex_desc.Usage = D3D11_USAGE_DEFAULT;
	msaa_tex_desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
	msaa_tex_desc.CPUAccessFlags = 0;
	msaa_tex_desc.MiscFlags = 0;
	RENDERER_THROW_FAILED(rnd_base->device()->CreateTexture2D(&msaa_tex_desc, nullptr, &m_msaa_target_texture));

	// make a non multi-sampled texture
	D3D11_TEXTURE2D_DESC nonmsaa_tex_desc = {};
	nonmsaa_tex_desc.Width = window::width;
	nonmsaa_tex_desc.Height = window::height;
	nonmsaa_tex_desc.MipLevels = 1;
	nonmsaa_tex_desc.ArraySize = 1;
	nonmsaa_tex_desc.Format = rnd_base->pixel_format();
	nonmsaa_tex_desc.SampleDesc.Count = 1;	// non multisampled
	nonmsaa_tex_desc.SampleDesc.Quality = 0;
	nonmsaa_tex_desc.Usage = D3D11_USAGE_DEFAULT;
	nonmsaa_tex_desc.BindFlags = 0;
	nonmsaa_tex_desc.CPUAccessFlags = 0;
	nonmsaa_tex_desc.MiscFlags = 0;
	RENDERER_THROW_FAILED(rnd_base->device()->CreateTexture2D(&nonmsaa_tex_desc, nullptr, &m_nonmsaa_intermediate_texture));

	// set the description for the render target view
	D3D11_RENDER_TARGET_VIEW_DESC rtv_desc = {};
	rtv_desc.Format = msaa_tex_desc.Format;
	rtv_desc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DMS;
	RENDERER_THROW_FAILED(rnd_base->device()->CreateRenderTargetView(m_msaa_target_texture.Get(), &rtv_desc, &m_target));

	// build the depth stencil buffer
	// disable stencil for now
	D3D11_DEPTH_STENCIL_DESC ds_desc = {};
	ds_desc.DepthEnable = TRUE;
	ds_desc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
	ds_desc.DepthFunc = D3D11_COMPARISON_LESS;
	ds_desc.StencilEnable = FALSE;
	RENDERER_THROW_FAILED(rnd_base->device()->CreateDepthStencilState(&ds_desc, &m_ds_state));
	rnd_base->context()->OMSetDepthStencilState(m_ds_state.Get(), 1);

	// create the depth stencil texture (data is written to a texture in direct3d11)
	D3D11_TEXTURE2D_DESC ds_tex_desc = {};
	ds_tex_desc.Width = window::width;
	ds_tex_desc.Height = window::height;
	ds_tex_desc.MipLevels = 1;	// multisampled
	ds_tex_desc.ArraySize = 1;
	ds_tex_desc.Format = DXGI_FORMAT_D32_FLOAT;
	ds_tex_desc.SampleDesc.Count = m_samples;
	ds_tex_desc.SampleDesc.Quality = m_quality;
	ds_tex_desc.Usage = D3D11_USAGE_DEFAULT;
	ds_tex_desc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
	ds_tex_desc.CPUAccessFlags = 0;
	ds_tex_desc.MiscFlags = 0;
	RENDERER_THROW_FAILED(rnd_base->device()->CreateTexture2D(&ds_tex_desc, nullptr, &m_ds_tex));

	// create the depth stencil view
	D3D11_DEPTH_STENCIL_VIEW_DESC dsv_desc = {};
	dsv_desc.Format = DXGI_FORMAT_D32_FLOAT;
	dsv_desc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DMS;
	RENDERER_THROW_FAILED(rnd_base->device()->CreateDepthStencilView(m_ds_tex.Get(), &dsv_desc, &m_ds_view));

	// set the viewport
	D3D11_VIEWPORT vport = {};
	vport.TopLeftX = 0.0f;
	vport.TopLeftY = 0.0f;
	vport.Width =  (FLOAT)window::width;
	vport.Height = (FLOAT)window::height;
	vport.MinDepth = 0.0f;
	vport.MaxDepth = 1.0f;
	rnd_base->context()->RSSetViewports(1, &vport);
	rnd_base->context()->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	D3D11_RASTERIZER_DESC rd = {};
	//rd.FillMode = D3D11_FILL_WIREFRAME;
	rd.FillMode = D3D11_FILL_SOLID;
	rd.CullMode = D3D11_CULL_BACK;
	rd.FrontCounterClockwise = FALSE;
	rd.DepthClipEnable = TRUE;
	RENDERER_THROW_FAILED(rnd_base->device()->CreateRasterizerState(&rd, &m_rs_state));
	rnd_base->context()->RSSetState(m_rs_state.Get());
}

void rasterizer::begin_frame()
{
	const iqvec clear_col = RENDERER->clear_color();
	float clear[4] = { clear_col[0], clear_col[1], clear_col[2], clear_col[3] };
	RENDERER->context()->OMSetRenderTargets(1, m_target.GetAddressOf(), m_ds_view.Get());
	RENDERER->context()->ClearDepthStencilView(m_ds_view.Get(), D3D11_CLEAR_DEPTH, 1.0f, 0);	// only clear the depth buffer for now
	RENDERER->context()->ClearRenderTargetView(m_target.Get(), clear);
}

void rasterizer::end_frame()
{
	HRESULT hr;
	renderer_base* rnd_base = RENDERER;

	// copy from a multi-sampled texture to a non-multisampled texture
	rnd_base->context()->ResolveSubresource(m_nonmsaa_intermediate_texture.Get(), 0, m_msaa_target_texture.Get(), 0, rnd_base->pixel_format());

	// copy to the back buffer for presenting
	wrl::ComPtr<ID3D11Texture2D> back_buffer;
	RENDERER_THROW_FAILED(rnd_base->swap_chain()->GetBuffer(0, __uuidof(ID3D11Texture2D), &back_buffer));
	rnd_base->context()->CopyResource(back_buffer.Get(), m_nonmsaa_intermediate_texture.Get());

	hr = rnd_base->swap_chain()->Present(1, 0);	// 1 for vsync;
	if (hr == DXGI_ERROR_DEVICE_REMOVED) {
		throw RENDERER_EXCEPTION(rnd_base->device()->GetDeviceRemovedReason());
	}
}

void rasterizer::draw_scene(const scene& scene, std::vector<shader>& shaders, float dt)
{
	//m_background->bind();
	//m_bg_shader->bind();
	//m_background->draw();	// instead of a simple quad, add hdri or cubemap support

	const std::set<model*, scene::model_comparator>& models = scene.get_models();	// these are the models sorted with respect to the mesh name
	std::string last_mesh_name = "";
	for (const auto& m : models) {
		// if instancing is used, this allows the shared mesh to be bound only once
		if (m->get_mesh_name() != last_mesh_name) {
			last_mesh_name = m->get_mesh_name();
			scene.get_mesh(last_mesh_name).bind();
		}

		shaders[0].update_transform(m->get_transform());
		shaders[0].bind();
		scene.get_mesh(last_mesh_name).draw();
	}
}
