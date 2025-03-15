#pragma once

#include "ioniq_windows.h"
#include <d3d11.h>
#include <wrl.h>
#include "iqmath.h"
#include "core.h"
#include "ioniq_exception.h"
#include "window.h"

#define RENDERER_THROW_FAILED(fcall) if (FAILED(hr = (fcall))) throw renderer_base::hr_exception(__LINE__, __FILE__, hr)
#define RENDERER_EXCEPTION(hr) renderer_base::hr_exception(__LINE__, __FILE__, (hr))	// used for device_removed exception
#define RENDERER_CUSTOMEXCEPTION(desc) renderer_base::hr_exception(__LINE__, __FILE__, 0, (desc))
#define RENDERER_THROW_CUDA(fcall) if (cderr = (fcall)) throw renderer_base::cuda_exception(__LINE__, __FILE__, cderr)

#define RENDERER renderer_base::get()
#define RENDERER_DEV renderer_base::get()->device()
#define RENDERER_CTX renderer_base::get()->context()

class renderer_base
{
public:
	class hr_exception : public ioniq_exception
	{
	public:
		hr_exception(int line, const std::string& file, HRESULT hr, const std::string& custom_desc = "");

		const char* what() const override;
		inline const char* get_type() const override { return "Ioniq Direct3D Exception"; }

	private:
		std::string get_description() const;

	private:
		HRESULT _hr;
		std::string _custom_desc;

	};

	class cuda_exception : public ioniq_exception
	{
	public:
		cuda_exception(int line, const std::string& file, cudaError err);

		const char* what() const override;
		inline const char* get_type() const override { return "Ioniq CUDA Exception"; }

	private:
		cudaError _err;

	};

public:
	static void init(const ref<window>& wnd);
	static void shutdown();
	static renderer_base* get();

	inline ID3D11Device* device() const { return m_device.Get(); }
	inline ID3D11DeviceContext* context() const { return m_imctx.Get(); }
	inline IDXGISwapChain* swap_chain() const { return m_swchain.Get(); }
	inline DXGI_FORMAT pixel_format() const { return m_output_format; }
	inline const iqvec& clear_color() const { return m_clear; }

private:
	renderer_base(const ref<window>& wnd);
	~renderer_base() = default;

private:
	ref<window> m_wnd;
	iqvec m_clear;
	DXGI_FORMAT m_output_format;

	Microsoft::WRL::ComPtr<ID3D11Device> m_device;
	Microsoft::WRL::ComPtr<ID3D11DeviceContext> m_imctx;	// immediate context, no calls to d3d from multiple threads yet
	Microsoft::WRL::ComPtr<IDXGISwapChain> m_swchain;
};
