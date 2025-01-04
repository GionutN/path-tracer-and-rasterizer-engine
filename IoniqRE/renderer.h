#pragma once

#include "ioniq_windows.h"
#include <d3d11.h>
#include <wrl.h>

#include "core.h"
#include "ioniq_exception.h"

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

	struct Vertex
	{
		real x;
		real y;
	};

public:
	static void init(HWND hWnd);
	static void shutdown();
	static renderer* get();

	void begin_frame();
	void end_frame();
	inline void set_clear_color(real* col) { m_clear[0] = col[0]; m_clear[1] = col[1]; m_clear[2] = col[2]; }

	void draw_triangle();

private:
	void set_triangle();

private:
	renderer(HWND hWnd);
	~renderer() = default;

private:
	Microsoft::WRL::ComPtr<ID3D11Device> m_device;
	Microsoft::WRL::ComPtr<ID3D11DeviceContext> m_imctx;	// immediate context, no calls to d3d from multiple threads yet
	Microsoft::WRL::ComPtr<IDXGISwapChain> m_swchain;
	Microsoft::WRL::ComPtr<ID3D11RenderTargetView> m_target;

	Microsoft::WRL::ComPtr<ID3D11Buffer> vb;
	Microsoft::WRL::ComPtr<ID3D11VertexShader> vs;
	Microsoft::WRL::ComPtr<ID3D11PixelShader> ps;

private:
	real m_clear[4] = {};

};

#define RENDERER_THROW_FAILED(fcall) if (FAILED(hr = (fcall))) throw renderer::exception(__LINE__, __FILE__, hr)
#define RENDERER_EXCEPTION(hr) renderer::exception(__LINE__, __FILE__, (hr))	// used for device_removed exception
#define RENDERER_CUSTOMEXCEPTION(desc) renderer::exception(__LINE__, __FILE__, 0, (desc))
