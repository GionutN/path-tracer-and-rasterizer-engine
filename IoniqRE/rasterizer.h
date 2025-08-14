#pragma once

#include "renderer_template.h"
#include "renderer_base.h"

class rasterizer : public renderer_template
{
public:
	static void init();
	static void shutdown();
	static rasterizer* get();

	void begin_frame() override;
	void end_frame() override;
	void draw_scene(const scene& scene, std::vector<shader>& shaders, float dt) override;

private:
	rasterizer();
	~rasterizer() = default;

private:
	Microsoft::WRL::ComPtr<ID3D11RenderTargetView> m_target;
	Microsoft::WRL::ComPtr<ID3D11Texture2D> m_msaa_target_texture;
	Microsoft::WRL::ComPtr<ID3D11Texture2D> m_nonmsaa_intermediate_texture;
	Microsoft::WRL::ComPtr<ID3D11RasterizerState> m_rs_state;

	UINT m_samples, m_quality;

	// make these pointers so that they do not get constructed before the renderer gets constructed
	ref<mesh> m_background;
	ref<shader> m_bg_shader;
};
