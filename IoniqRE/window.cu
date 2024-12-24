#include "window.h"

#include <sstream>

window::window(HINSTANCE hInstance, UINT16 width, UINT16 height)
	:
	m_width(width),
	m_height(height)
{
	HRESULT hr;
	BOOL ok;

	// fill out the window class struct
	WNDCLASSEX wnd_class = {};
	wnd_class.cbSize = sizeof(WNDCLASSEX);
	wnd_class.style = CS_OWNDC;
	wnd_class.lpfnWndProc = HandleMessageSetup;
	wnd_class.cbClsExtra = 0;
	wnd_class.cbWndExtra = 0;
	wnd_class.hInstance = hInstance;
	wnd_class.hIcon = nullptr;
	wnd_class.hCursor = nullptr;
	wnd_class.hbrBackground = nullptr;
	wnd_class.lpszMenuName = nullptr;
	wnd_class.lpszClassName = window_class_name;
	wnd_class.hIconSm = nullptr;
	hr = (HRESULT)RegisterClassEx(&wnd_class);
	if (hr == 0) {
		throw IONIQWNDEXCEPT_LAST();
	}

	// create the window
	RECT client = {};
	client.left = 1000;
	client.top = 200;
	client.right = client.left + width;
	client.bottom = client.top + height;
	ok = AdjustWindowRect(&client, WS_CAPTION | WS_MINIMIZEBOX | WS_SYSMENU, FALSE);
	if (ok == 0) {
		throw IONIQWNDEXCEPT_LAST();
	}
	m_hWnd = CreateWindow(window_class_name, "Ioniq Rendering Engine", WS_CAPTION | WS_MINIMIZEBOX | WS_SYSMENU, CW_USEDEFAULT, CW_USEDEFAULT,
		client.right - client.left, client.bottom - client.top, nullptr, nullptr, hInstance, this);
	if (m_hWnd == nullptr) {
		throw IONIQWNDEXCEPT_LAST();
	}
	ok = UpdateWindow(m_hWnd);
	if (ok == 0) {
		throw IONIQWNDEXCEPT_LAST();
	}
	ShowWindow(m_hWnd, SW_SHOWDEFAULT);
}

window::~window()
{
	DestroyWindow(m_hWnd);
	UnregisterClass(window_class_name, GetModuleHandle(nullptr));
}

LRESULT window::HandleMessageSetup(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	// setup win32 so that it calls our handle message method, which is not static
	if (msg == WM_NCCREATE) {
		CREATESTRUCTA* create = reinterpret_cast<CREATESTRUCTA*>(lParam);
		window* wnd = reinterpret_cast<window*>(create->lpCreateParams);
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

window::exception::exception(int line, const std::string& file, HRESULT hr)
	:
	ioniq_exception(line, file),
	m_hr(hr)
{
}

const char* window::exception::what() const
{
	std::ostringstream oss;
	oss << get_type() << std::endl;
	oss << "[Error Code]: " << m_hr << std::endl;
	oss << "[Description]: " << get_description() << std::endl;
	oss << get_origin();
	m_what_buffer = oss.str();
	return m_what_buffer.c_str();
}

std::string window::exception::get_description() const
{
	char* msg = nullptr;
	DWORD len = FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
		nullptr, m_hr, 0, reinterpret_cast<LPSTR>(&msg), 0, nullptr);
	if (len == 0) {
		return "Unknown error code";
	}

	std::string out = msg;
	LocalFree(msg);
	return out;
}
