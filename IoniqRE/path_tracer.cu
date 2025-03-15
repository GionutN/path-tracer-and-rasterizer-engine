#include "path_tracer.h"

#include <d3dcompiler.h>

static path_tracer* g_path_tracer = nullptr;

namespace wrl = Microsoft::WRL;

void path_tracer::init()
{
	if (!g_path_tracer) {
		g_path_tracer = new path_tracer();
	}
}

void path_tracer::shutdown()
{
	if (g_path_tracer) {
		delete g_path_tracer;
		g_path_tracer = nullptr;
	}
}

path_tracer* path_tracer::get()
{
	return g_path_tracer;
}

void path_tracer::begin_frame()
{
	RENDERER->context()->OMSetRenderTargets(1, m_target.GetAddressOf(), nullptr);
}

void path_tracer::end_frame()
{
	HRESULT hr;
	renderer_base* rnd_base = RENDERER;

	// update the texture only when the pathtracer updates the pixel buffer
	if (m_ptimage_updated) {
		RENDERER_THROW_FAILED(rnd_base->context()->Map(m_texture.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &m_mapped_texture));

		// setup parameteres for copying
		pixel* dest = (pixel*)(m_mapped_texture.pData);
		const size_t dest_pitch = m_mapped_texture.RowPitch / sizeof(pixel);
		const size_t src_pitch = window::width;
		const size_t row_bytes = src_pitch * sizeof(pixel);
		// perform the copy line-by-line
		for (size_t i = 0; i < window::height; i++)
		{
			memcpy(&dest[i * dest_pitch], &m_host_pixel_buffer[i * src_pitch], row_bytes);
		}
		// release the adapter memory
		RENDERER->context()->Unmap(m_texture.Get(), 0);

		m_ptimage_updated = false;
	}

	// render offscreen scene texture to back buffer
	const UINT stride = 16, offset = 0;
	RENDERER->context()->IASetVertexBuffers(0, 1, m_vertex_buffer.GetAddressOf(), &stride, &offset);
	RENDERER->context()->IASetInputLayout(m_layout.Get());
	RENDERER->context()->VSSetShader(m_vertex_shader.Get(), nullptr, 0);
	RENDERER->context()->PSSetShader(m_pixel_shader.Get(), nullptr, 0);
	RENDERER->context()->PSSetShaderResources(0, 1, m_texture_view.GetAddressOf());
	RENDERER->context()->PSSetSamplers(0, 1, m_texture_sampler.GetAddressOf());
	RENDERER->context()->Draw(6, 0);

	hr = rnd_base->swap_chain()->Present(1, 0);	// 1 for vsync;
	if (hr == DXGI_ERROR_DEVICE_REMOVED) {
		throw RENDERER_EXCEPTION(rnd_base->device()->GetDeviceRemovedReason());
	}
}

__device__ iqvec path_tracer::ray_color(const ray& r, scene::gpu_packet packet)
{
	if (packet.vertices != nullptr) {
		for (UINT i = 0; i < packet.indices[0]; i += 3) {
			// rebuild in CW order
			iqvec v0 = iqvec::load(packet.vertices[packet.indices[i + 3]].pos, iqvec::usage::POINT);
			iqvec v1 = iqvec::load(packet.vertices[packet.indices[i + 2]].pos, iqvec::usage::POINT);
			iqvec v2 = iqvec::load(packet.vertices[packet.indices[i + 1]].pos, iqvec::usage::POINT);

			triangle tr(v0, v1, v2);
			if (tr.intersect(r)) {
				return iqvec(1.0f, 0.0f, 0.0f, 0.0f);
			}
		}
	}

	iqvec dir = r.direction().normalize3();
	const float t = (dir.y + 1.0f) * 0.5f;
	return (1.0f - t) * iqvec(1.0f) + t * iqvec(0.5f, 0.7f, 1.0f, 0.0f);
}

__global__ static void render_kernel(path_tracer::pixel* fb, int width, int height, scene::gpu_packet packet)
{
	const int y = threadIdx.y + blockIdx.y * blockDim.y;
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	if (y >= height || x >= width) {
		return;
	}
	// compute the viewing direction of each pixel
	const float aspect_ratio = (float)width / height;
	iqvec center = iqvec(0.0f, 0.0f, 2.0f, 1.0f);
	iqvec viewport = iqvec(0.0f, 0.0f, 1.0f, 0.0f);
	iqvec viewport_u = iqvec(aspect_ratio, 0.0f, 0.0f, 0.0f);
	iqvec viewport_v = iqvec(0.0f, -1.0f, 0.0f, 0.0f);
	iqvec du = viewport_u / width;
	iqvec dv = viewport_v / height;
	iqvec topleft = center - viewport - (viewport_u + viewport_v) / 2.0f;
	iqvec pixel00 = topleft + 0.5f * (du + dv);
	iqvec crt_pixel = pixel00 + x * du + y * dv;
	ray r = ray(center, crt_pixel - center);

	iqvec color = path_tracer::ray_color(r, packet);
	color.x = color.x > 1.0f ? 1.0f : (color.x < 0.0f ? 0.0f : color.x);
	color.y = color.y > 1.0f ? 1.0f : (color.y < 0.0f ? 0.0f : color.y);
	color.z = color.z > 1.0f ? 1.0f : (color.z < 0.0f ? 0.0f : color.z);

	const int pixelid = y * width + x;
	fb[pixelid].r = (uint8_t)(255.0f * color.x);
	fb[pixelid].g = (uint8_t)(255.0f * color.y);
	fb[pixelid].b = (uint8_t)(255.0f * color.z);
	fb[pixelid].a = 255;
}

void path_tracer::draw_scene(const scene& scene, const std::vector<shader>& shaders, float dt)
{
	cudaError cderr;

	static float time = 0.0f;
	time += dt;

	// let the compute kernel run for half a second, then retrieve the computed image
	if (time > 0.5f) {
		time = 0.0f;

		UINT tx = 16, ty = 16;
		UINT num_pixels = window::height * window::width;

		dim3 blocks(window::width / tx + 1, window::height / ty + 1);
		dim3 threads(tx, ty);
		RENDERER_THROW_CUDA(cudaGetLastError());
		size_t fbsize = num_pixels * sizeof(pixel);

		// copy the framebuffer from device to host
		RENDERER_THROW_CUDA(cudaDeviceSynchronize());	// synchronize before calling the kernel, so that it runs for half a second
		RENDERER_THROW_CUDA(cudaMemcpy(m_host_pixel_buffer, m_dev_pixel_buffer, fbsize, cudaMemcpyDeviceToHost));
		m_ptimage_updated = true;

		// TODO:
		// path-tracer multi-sampling (ever-converging)
		// model class? that refers to an array of meshes and shaders, for instancing
		// refering based on a name (giving names to meshes and materials)
		// sphere mesh class

		// copy the scene data from host to device only if changed
		if (scene.modified()) {
			if (d_packet.vertices != nullptr) {
				RENDERER_THROW_CUDA(cudaFree(d_packet.vertices));
				RENDERER_THROW_CUDA(cudaFree(d_packet.indices));
				RENDERER_THROW_CUDA(cudaFree(d_packet.model_types));
			}
			d_packet = scene.build_packet();
		}
		render_kernel<<<blocks, threads>>>(m_dev_pixel_buffer, window::width, window::height, d_packet);
	}
}

path_tracer::path_tracer()
{
	HRESULT hr;
	cudaError cderr;
	renderer_base* rnd_base = RENDERER;

	wrl::ComPtr<ID3D11Texture2D> back_buffer;
	RENDERER_THROW_FAILED(rnd_base->swap_chain()->GetBuffer(0, __uuidof(ID3D11Texture2D), &back_buffer));
	RENDERER_THROW_FAILED(rnd_base->device()->CreateRenderTargetView(back_buffer.Get(), nullptr, &m_target));

	D3D11_TEXTURE2D_DESC tex_desc = {};
	tex_desc.Width  = window::width;
	tex_desc.Height = window::height;
	tex_desc.MipLevels = 1;
	tex_desc.ArraySize = 1;
	tex_desc.Format = rnd_base->pixel_format();
	tex_desc.SampleDesc.Count = 1;	// non multisampled
	tex_desc.SampleDesc.Quality = 0;
	tex_desc.Usage = D3D11_USAGE_DYNAMIC;
	tex_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	tex_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	tex_desc.MiscFlags = 0;
	RENDERER_THROW_FAILED(rnd_base->device()->CreateTexture2D(&tex_desc, nullptr, &m_texture));

	D3D11_SHADER_RESOURCE_VIEW_DESC tex_view = {};
	tex_view.Format = tex_desc.Format;
	tex_view.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
	tex_view.Texture2D.MipLevels = 1;
	RENDERER_THROW_FAILED(rnd_base->device()->CreateShaderResourceView(m_texture.Get(), &tex_view, &m_texture_view));

	wrl::ComPtr<ID3DBlob> blob;
	RENDERER_THROW_FAILED(D3DReadFileToBlob(L"PTPixelShader.cso", &blob));
	RENDERER_THROW_FAILED(rnd_base->device()->CreatePixelShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &m_pixel_shader));

	RENDERER_THROW_FAILED(D3DReadFileToBlob(L"PTVertexShader.cso", &blob));
	RENDERER_THROW_FAILED(rnd_base->device()->CreateVertexShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &m_vertex_shader));

	D3D11_INPUT_ELEMENT_DESC vertex_layout[2] = {
		{"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA},
		{"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 8, D3D11_INPUT_PER_VERTEX_DATA}
	};
	RENDERER_THROW_FAILED(rnd_base->device()->CreateInputLayout(vertex_layout, (UINT)std::size(vertex_layout), blob->GetBufferPointer(), blob->GetBufferSize(), &m_layout));

	const float vertices[] = {
		-1.0f,  1.0f, 0.0f, 0.0f,
		 1.0f,  1.0f, 1.0f, 0.0f,
		 1.0f, -1.0f, 1.0f, 1.0f,

		 1.0f, -1.0f, 1.0f, 1.0f,
		-1.0f, -1.0f, 0.0f, 1.0f,
		-1.0f,  1.0f, 0.0f, 0.0f
	};
	D3D11_BUFFER_DESC bdesc = {};
	bdesc.ByteWidth = (UINT)sizeof(vertices);
	bdesc.Usage = D3D11_USAGE_DEFAULT;
	bdesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bdesc.CPUAccessFlags = 0;
	bdesc.MiscFlags = 0;
	bdesc.StructureByteStride = 4 * (UINT)sizeof(float);

	D3D11_SUBRESOURCE_DATA bdata = {};
	bdata.pSysMem = vertices;
	RENDERER_THROW_FAILED(rnd_base->device()->CreateBuffer(&bdesc, &bdata, &m_vertex_buffer));

	D3D11_SAMPLER_DESC smpl_desc = {};
	smpl_desc.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
	smpl_desc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
	smpl_desc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
	smpl_desc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
	smpl_desc.ComparisonFunc = D3D11_COMPARISON_NEVER;
	smpl_desc.MinLOD = 0.0f;
	smpl_desc.MaxLOD = D3D11_FLOAT32_MAX;
	RENDERER_THROW_FAILED(rnd_base->device()->CreateSamplerState(&smpl_desc, &m_texture_sampler));

	size_t num_pixels = window::width * window::height;
	size_t fbsize = num_pixels * sizeof(pixel);
	RENDERER_THROW_CUDA(cudaMalloc((void**)&m_dev_pixel_buffer, fbsize));
	m_host_pixel_buffer = new pixel[num_pixels];
	memset(m_host_pixel_buffer, 0, fbsize);
	RENDERER_THROW_CUDA(cudaMemset(m_dev_pixel_buffer, 0, fbsize));
}
