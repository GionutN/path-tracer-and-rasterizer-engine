#pragma once

#include "ioniq_windows.h"
#include <d3d11.h>
#include <wrl.h>

#include "core.h"
#include "ioniq_exception.h"
#include "window.h"
#include "shader.h"
#include "mesh.h"

#define RENDERER_THROW_FAILED(fcall) if (FAILED(hr = (fcall))) throw renderer::exception(__LINE__, __FILE__, hr)
#define RENDERER_EXCEPTION(hr) renderer::exception(__LINE__, __FILE__, (hr))	// used for device_removed exception
#define RENDERER_CUSTOMEXCEPTION(desc) renderer::exception(__LINE__, __FILE__, 0, (desc))

#define RENDERER renderer::get()
#define RENDERER_DEV renderer::get()->get_device_view()
#define RENDERER_CTX renderer::get()->get_context_view()

class renderer
{
public:
	class exception : public ioniq_exception
	{
	public:
		exception(int line, const std::string& file, HRESULT hr, const std::string& custom_desc = "");

		const char* what() const override;
		inline const char* get_type() const override { return "Ioniq Renderer Exception"; }

	private:
		std::string get_description() const;

	private:
		HRESULT _hr;
		std::string _custom_desc;

	};

public:
	static void init(const ref<window>& wnd);
	static void shutdown();
	static renderer* get();

	void begin_frame();
	void end_frame();
	inline void set_clear_color(real* col) { m_clear[0] = col[0]; m_clear[1] = col[1]; m_clear[2] = col[2]; }

	inline ID3D11Device* get_device_view() const { return m_device.Get(); }
	inline ID3D11DeviceContext* get_context_view() const { return m_imctx.Get(); }

	void draw_scene(const std::vector<mesh>& scene, const std::vector<shader>& shaders);

private:
	renderer(const ref<window>& wnd);
	~renderer() = default;

private:
	ref<window> m_wnd;
	Microsoft::WRL::ComPtr<ID3D11Device> m_device;
	Microsoft::WRL::ComPtr<ID3D11DeviceContext> m_imctx;	// immediate context, no calls to d3d from multiple threads yet
	Microsoft::WRL::ComPtr<IDXGISwapChain> m_swchain;
	Microsoft::WRL::ComPtr<ID3D11RenderTargetView> m_target;
	Microsoft::WRL::ComPtr<ID3D11Texture2D> m_msaa_target_texture;
	Microsoft::WRL::ComPtr<ID3D11Texture2D> m_nonmsaa_intermediate_texture;

private:
	real m_clear[4] = {};
	UINT m_samples, m_quality;
	DXGI_FORMAT m_output_format;

	// make these pointers so that they do not get constructed before the renderer gets constructed
	ref<mesh> m_background;
	ref<shader> m_bg_shader;

};
