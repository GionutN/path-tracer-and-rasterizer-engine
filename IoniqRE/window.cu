#include "window.h"

window::window(HINSTANCE hInstance, UINT16 width, UINT16 height)
	:
	m_width(width),
	m_height(height)
{
	// fill out the window class struct
	WNDCLASSEX wndClass = {};
	wndClass.cbSize = sizeof(WNDCLASSEX);
	wndClass.style = CS_OWNDC;
	wndClass.lpfnWndProc = HandleMessageSetup;
	wndClass.cbClsExtra = 0;
	wndClass.cbWndExtra = 0;
	wndClass.hInstance = hInstance;
	wndClass.hIcon = nullptr;
	wndClass.hCursor = nullptr;
	wndClass.hbrBackground = nullptr;
	wndClass.lpszMenuName = nullptr;
	wndClass.lpszClassName = window_class_name;
	wndClass.hIconSm = nullptr;
	RegisterClassEx(&wndClass);

	// create the window
	RECT client = {};
	client.left = 1000;
	client.top = 200;
	client.right = client.left + width;
	client.bottom = client.top + height;
	AdjustWindowRect(&client, WS_CAPTION | WS_MINIMIZEBOX | WS_SYSMENU, FALSE);
	m_hWnd = CreateWindow(window_class_name, "Ioniq Rendering Engine", WS_CAPTION | WS_MINIMIZEBOX | WS_SYSMENU, CW_USEDEFAULT, CW_USEDEFAULT,
		client.right - client.left, client.bottom - client.top, nullptr, nullptr, hInstance, this);
	UpdateWindow(m_hWnd);
	ShowWindow(m_hWnd, SW_SHOWDEFAULT);
}

window::~window()
{
	DestroyWindow(m_hWnd);
	UnregisterClass(window_class_name, GetModuleHandle(nullptr));
}

bool window::should_close() const
{
	MSG msg;
	while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
		if (msg.message == WM_QUIT)
			return true;
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}

	return false;
}

LRESULT window::HandleMessageSetup(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	// setup win32 so that it calls our handle message method, which is not static
	if (msg == WM_NCCREATE) {
		CREATESTRUCTA* pCreate = reinterpret_cast<CREATESTRUCTA*>(lParam);
		window* wnd = reinterpret_cast<window*>(pCreate->lpCreateParams);
		SetWindowLongPtr(hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(wnd));
		SetWindowLongPtr(hWnd, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(&HandleMessageDummy));
	}

	return DefWindowProc(hWnd, msg, wParam, lParam);
}

LRESULT window::HandleMessageDummy(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	// the dummy function that calls our message handling method
	window* wnd = reinterpret_cast<window*>(GetWindowLongPtr(hWnd, GWLP_USERDATA));
	return wnd->HandleMessage(hWnd, msg, wParam, lParam);
}

LRESULT window::HandleMessage(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	switch (msg) {
	case WM_QUIT:
	case WM_CLOSE:
		PostQuitMessage(0);
		break;
	}

	return DefWindowProc(hWnd, msg, wParam, lParam);
}
