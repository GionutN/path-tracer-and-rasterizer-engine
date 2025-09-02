#pragma once

#include <curand_kernel.h>

#include "renderer_base.h"
#include "renderer_template.h"

#include "camera.h"

class path_tracer : public renderer_template
{
public:
	struct pixel
	{
		uint8_t b;
		uint8_t g;
		uint8_t r;
		uint8_t a;
	};

public:
	static void init(camera* cam);
	static void shutdown();
	static path_tracer* get();

	void begin_frame() override;
	void end_frame() override;
	void draw_scene(const scene& scene, std::vector<shader>& shaders, float dt) override;

	__device__ static iqvec ray_color(const ray& r, scene::gpu_packet packet);

	inline void reset() { m_pending_reset = true; }

private:
	path_tracer(camera* cam);
	~path_tracer();

private:
	Microsoft::WRL::ComPtr<ID3D11RenderTargetView> m_target;
	Microsoft::WRL::ComPtr<ID3D11Texture2D> m_texture;
	Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_texture_view;
	Microsoft::WRL::ComPtr<ID3D11SamplerState> m_texture_sampler;
	Microsoft::WRL::ComPtr<ID3D11PixelShader> m_pixel_shader;
	Microsoft::WRL::ComPtr<ID3D11VertexShader> m_vertex_shader;
	Microsoft::WRL::ComPtr<ID3D11InputLayout> m_layout;
	Microsoft::WRL::ComPtr<ID3D11Buffer> m_vertex_buffer;
	D3D11_MAPPED_SUBRESOURCE m_mapped_texture;

	const dim3 m_threads_per_block;
	const dim3 m_blocks_per_grid;
	const size_t m_fbsize;
	const size_t m_num_pixels;
	curandState* m_dev_rand_state;
	camera* m_camera;

	pixel* m_host_pixel_buffer;
	pixel* m_dev_pixel_buffer;	// ever-converging, should set a limit
	size_t m_crt_frame;	// the current traced frame
	scene::gpu_packet d_packet;
	bool m_image_updated;
	bool m_pending_reset = false;	// flag for resetting to avoid data race when clearing the device pixel buffer

};
