#include "path_tracer.h"

#include <d3dcompiler.h>
#include <device_launch_parameters.h>

#include "random.h"
#include "material.h"
#include "iqmath.h"

static constexpr UINT threads = 16;

static path_tracer* g_path_tracer = nullptr;

namespace wrl = Microsoft::WRL;

void path_tracer::init(camera* cam)
{
	if (!g_path_tracer) {
		g_path_tracer = new path_tracer(cam);
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

__global__ static void renderer_init_kernel(UINT width, UINT height, curandState* rand_states)
{
	const UINT y = threadIdx.y + blockIdx.y * blockDim.y;
	const UINT x = threadIdx.x + blockIdx.x * blockDim.x;
	if (y >= height || x >= width) {
		return;
	}
	const UINT pixelid = y * width + x;
	// each thread gets same seed, a different sequence number, no offset
	curand_init(1984, pixelid, 0, &rand_states[pixelid]);
}

path_tracer::path_tracer(camera* cam)
	:
	m_blocks_per_grid(window::width / threads + 1, window::height / threads + 1),
	m_threads_per_block(threads, threads),
	m_num_pixels(window::height * window::width),
	m_fbsize(window::height * window::width * sizeof(pixel)),
	m_camera(cam)
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

	// build the quad used as the canvas
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

	RENDERER_THROW_CUDA(cudaMalloc((void**)&m_dev_pixel_buffer, m_fbsize));
	m_host_pixel_buffer = new pixel[m_num_pixels];
	memset(m_host_pixel_buffer, 0, m_fbsize);
	// clear the accumulated device pixel buffer
	RENDERER_THROW_CUDA(cudaMalloc((void**)&m_dev_linear_color_buffer, 4 * m_fbsize));	// 4 floats per channel insted of 1
	RENDERER_THROW_CUDA(cudaMemset(m_dev_pixel_buffer, 0, m_fbsize));
	RENDERER_THROW_CUDA(cudaMemset(m_dev_linear_color_buffer, 0, 4 * m_fbsize));
	m_crt_frame = 0ui64;
	
	RENDERER_THROW_CUDA(cudaMalloc((void**)&m_dev_rand_state, m_num_pixels * sizeof(curandState)));
	renderer_init_kernel<<<m_blocks_per_grid, m_threads_per_block>>>(window::width, window::height, m_dev_rand_state);
	RENDERER_THROW_CUDA(cudaGetLastError());
	RENDERER_THROW_CUDA(cudaDeviceSynchronize());
}

path_tracer::~path_tracer()
{
	cudaDeviceSynchronize();
	if (m_dev_pixel_buffer) {
		cudaFree(m_dev_pixel_buffer);
		m_dev_pixel_buffer = nullptr;
	}
	if (m_dev_linear_color_buffer) {
		cudaFree(m_dev_linear_color_buffer);
		m_dev_linear_color_buffer = nullptr;
	}
	if (m_host_pixel_buffer) {
		delete[] m_host_pixel_buffer;
		m_host_pixel_buffer = nullptr;
	}
	if (m_dev_rand_state) {
		cudaFree(m_dev_rand_state);
		m_dev_rand_state = nullptr;
	}
	cudaDeviceReset();
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
	if (m_image_updated) {
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

		m_image_updated = false;
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

__device__ iqvec path_tracer::pixel_shader(const ray& r, const hit_record& hr)
{
	iqvec light_color = 1.0f;
	iqvec ambient_color(0.62f, 0.84f, 1.0f, 0.0f);	// this is the clear color
	iqvec albedo(1.0f, 0.0f, 0.0f, 0.0f);
	iqvec final_color = 0.0f;
	float ambient_strength = 0.2f;

	iqvec ambient = ambient_strength * ambient_color;

	iqvec light_dir = iqvec(1.0f, 0.0f, 1.0f, 0.0f).normalize3();
	float diffuse = fmaxf(-hr.n.dot3(light_dir), 0.0f);
	iqvec diffuse_col = diffuse * light_color;

	final_color = (ambient + diffuse_col).hadamard(albedo);

	return final_color;
}

__device__ iqvec path_tracer::ray_color(const ray& r, scene::gpu_packet packet, curandState* local_state)
{
	// this is the integrator
	// the base formula is Lo = Li * bsdf * (normal.dot(wi)) / pdf
	// Lo/i is the ray color, the bsdf value is the color returned from the surface
	// wi is the direction from the surface of the scattered ray
	// wo is the direction from the surface to the camera
	// pdf is the value of the pdf with which wi was sampled wrt wo

	const int max_depth = 5;
	const float t_min = 0.000001f, t_max = 999.99f;

	scatter_record ray_stack[max_depth] = {};
	ray crt_ray = r;

	int crt_depth;

	oren_nayar mat(iqvec(0.5f, 0.5f, 0.5f, 0.0f), 1.0f);
	emissive light(1.0f, 10.0f);

	// to avoid recursion, go through the rays and add them to a stack
	for (crt_depth = 0; crt_depth < max_depth; crt_depth++) {
		hit_record final_hr;
		float closest_hit = t_max;
		bool hit = false;

		for (UINT i = 0; i < packet.num_drawcalls[mesh::type::TRIANGLES]; i++) {
			const UINT mesh_id = packet.tri_mesh_dcs[i].mesh_id;
			const iqmat transform = packet.tri_mesh_dcs[i].transform;
			const iqmat normal_matrix = iqmat::load3x3(transform.store3x3().transpose().inverse());

			const scene::gpu_packet::tri_mesh m = packet.tri_meshes[mesh_id];
			for (UINT j = 0; j < m.num_indices; j += 3) {
				// in CW order
				iqvec v0 = iqvec::load(m.vertices[m.indices[j + 0]].pos, iqvec::usage::POINT).transform(transform);
				iqvec v1 = iqvec::load(m.vertices[m.indices[j + 1]].pos, iqvec::usage::POINT).transform(transform);
				iqvec v2 = iqvec::load(m.vertices[m.indices[j + 2]].pos, iqvec::usage::POINT).transform(transform);
				iqvec n0 = iqvec::load(m.vertices[m.indices[j + 0]].normal, iqvec::usage::DIRECTION).transform(normal_matrix);
				iqvec n1 = iqvec::load(m.vertices[m.indices[j + 1]].normal, iqvec::usage::DIRECTION).transform(normal_matrix);
				iqvec n2 = iqvec::load(m.vertices[m.indices[j + 2]].normal, iqvec::usage::DIRECTION).transform(normal_matrix);

				// add normals here
				triangle tr(v0, v1, v2, n0, n1, n2);
				hit_record hr;
				if (tr.intersect(crt_ray, t_min, closest_hit, &hr)) {
					closest_hit = hr.t;
					final_hr = hr;
					final_hr.mat = &light;
					hit = true;
				}
			}
		}

		// TODO: add here intersection checks for other shape primitives
		for (UINT i = 0; i < packet.num_drawcalls[mesh::type::SPHERES]; i++) {
			sphere s(packet.sphere_dcs[i].center, packet.sphere_dcs[i].radius);

			hit_record hr;
			if (s.intersect(crt_ray, t_min, closest_hit, &hr)) {
				closest_hit = hr.t;
				final_hr = hr;
				final_hr.mat = &mat;
				hit = true;
			}
		}

		if (hit) {
			ray r_out;
			if (final_hr.mat->scatter(crt_ray, final_hr, &ray_stack[crt_depth], &r_out, local_state)) {
				crt_ray = r_out;
			}
			else {
				crt_depth++;
				break;
			}
		}
		else {
			iqvec dir = crt_ray.direction();
			const float a = (dir.y + 1.0f) * 0.5f;
			ray_stack[crt_depth].attenuation = (1.0f - a) * iqvec(1.0f) + a * iqvec(0.5f, 0.7f, 1.0f, 0.0f);
			ray_stack[crt_depth].pdf_val = 1.0f;
			ray_stack[crt_depth].cos_law_weight = 1.0f;
			crt_depth++;
			break;
		}
			
	}

	// finally go backwards in the stack to get the final color
	iqvec final_color = ray_stack[crt_depth - 1].cos_law_weight / ray_stack[crt_depth - 1].pdf_val * ray_stack[crt_depth - 1].attenuation;
	for (int depth = crt_depth - 2; depth >= 0; depth--) {
		final_color = final_color.hadamard(ray_stack[depth].cos_law_weight / ray_stack[depth].pdf_val * ray_stack[depth].attenuation);
	}

	return final_color;
	
}

__global__ static void render_kernel(path_tracer::pixel* fb, iqvec* lin_fb, size_t num_frame, camera* cam, scene::gpu_packet packet, curandState* rand_states)
{
	const UINT y = threadIdx.y + blockIdx.y * blockDim.y;
	const UINT x = threadIdx.x + blockIdx.x * blockDim.x;
	if (y >= cam->get_height() || x >= cam->get_width()) {
		return;
	}
	// compute the viewing direction of each pixel
	const UINT pixelid = y * cam->get_width() + x;
	curandState* local_state = &rand_states[pixelid];

	iqvec path_color;
	ray r = cam->get_ray(x, y, local_state);

	iqvec color = path_tracer::ray_color(r, packet, local_state);
	color.x = color.x > 1.0f ? 1.0f : (color.x < 0.0f ? 0.0f : color.x);
	color.y = color.y > 1.0f ? 1.0f : (color.y < 0.0f ? 0.0f : color.y);
	color.z = color.z > 1.0f ? 1.0f : (color.z < 0.0f ? 0.0f : color.z);
	path_color += color;

	// check for NaNs
	if (color.x != color.x) color.x = 0.0f;
	if (color.y != color.y) color.y = 0.0f;
	if (color.z != color.z) color.z = 0.0f;

	// have the float buffer as accumulator to avoid losing information when casting to uint8_t
	lin_fb[pixelid].x = path_color.x / num_frame + lin_fb[pixelid].x * ((num_frame - 1) / (float)num_frame);
	lin_fb[pixelid].y = path_color.y / num_frame + lin_fb[pixelid].y * ((num_frame - 1) / (float)num_frame);
	lin_fb[pixelid].z = path_color.z / num_frame + lin_fb[pixelid].z * ((num_frame - 1) / (float)num_frame);

	path_tracer::pixel p;
	p.r = (uint8_t)(255.0f * sqrtf(lin_fb[pixelid].x));
	p.g = (uint8_t)(255.0f * sqrtf(lin_fb[pixelid].y));
	p.b = (uint8_t)(255.0f * sqrtf(lin_fb[pixelid].z));
	p.a = 255;
	fb[pixelid] = p;
}

void path_tracer::draw_scene(const scene& scene, std::vector<shader>& shaders, float dt)
{
	cudaError cderr;

	static float time = 0.0f;
	time += dt;

	// let the compute kernel run for half a second, then retrieve the computed image
	// there is a problem here, the timer gets updated only when the currently selected engine is the path tracer
	// on engine switch to this one, the kernel call must happen right away
	if (time > 0.25f) {
		time = 0.0f;

		// copy the framebuffer from device to host
		RENDERER_THROW_CUDA(cudaDeviceSynchronize());	// synchronize before calling the kernel, so that it runs for half a second
		RENDERER_THROW_CUDA(cudaGetLastError());

		RENDERER_THROW_CUDA(cudaMemcpy(m_host_pixel_buffer, m_dev_pixel_buffer, m_fbsize, cudaMemcpyDeviceToHost));
		m_image_updated = true;

		// copy the scene data from host to device only if changed
		if (scene.modified()) {
			scene.free_packet(&d_packet);
			d_packet = scene.build_packet();
		}
		// reset the buffer when the kernel is not running
		if (m_pending_reset) {
			//scene.free_packet(&d_packet);
			//d_packet = scene.build_packet();
			RENDERER_THROW_CUDA(cudaMemset(m_dev_pixel_buffer, 0, m_fbsize));
			m_crt_frame = 0;
			m_pending_reset = false;
		}
		m_crt_frame++;
		render_kernel<<<m_blocks_per_grid, m_threads_per_block>>> (m_dev_pixel_buffer, m_dev_linear_color_buffer, m_crt_frame, m_camera, d_packet, m_dev_rand_state);
	}
}
