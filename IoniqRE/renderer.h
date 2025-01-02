#pragma once

#include "ioniq_windows.h"
#include <d3d11.h>
#include <wrl.h>

class renderer
{
public:
	static void init(HWND hWnd);
	static void shutdown();
	static renderer* get();

	void begin_frame();
	void end_frame();

private:
	renderer(HWND hWnd);
	~renderer() = default;

private:
	Microsoft::WRL::ComPtr<ID3D11Device> m_device;
	Microsoft::WRL::ComPtr<ID3D11DeviceContext> m_imctx;	// immediate context, no calls to d3d from multiple threads yet
	Microsoft::WRL::ComPtr<IDXGISwapChain> m_swchain;
	Microsoft::WRL::ComPtr<ID3D11RenderTargetView> m_target;

};
