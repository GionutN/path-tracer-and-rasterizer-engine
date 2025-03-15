#pragma once

#include "renderer_base.h"
#include "renderer_template.h"

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
	static void init();
	static void shutdown();
	static path_tracer* get();

	void begin_frame() override;
	void end_frame() override;
	void draw_scene(const scene& scene, const std::vector<shader>& shaders, float dt) override;

	__device__ static iqvec ray_color(const ray& r, scene::gpu_packet packet);

private:
	path_tracer();
	~path_tracer() = default;

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
	pixel* m_dev_pixel_buffer;
	pixel* m_host_pixel_buffer;
	scene::gpu_packet d_packet = { nullptr, nullptr, nullptr };

	bool m_ptimage_updated;

};
