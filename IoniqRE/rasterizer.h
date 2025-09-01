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
	Microsoft::WRL::ComPtr<ID3D11DepthStencilState> m_ds_state;
	Microsoft::WRL::ComPtr<ID3D11Texture2D> m_ds_tex;
	Microsoft::WRL::ComPtr<ID3D11DepthStencilView> m_ds_view;

	UINT m_samples, m_quality;
};
