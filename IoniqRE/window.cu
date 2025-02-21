#include "window.h"

#include <sstream>

#include "core.h"
#include "keyboard.h"
#include "mouse.h"
#include "timer.h"
#include "renderer.h"

window::window(HINSTANCE hInstance, UINT16 width, UINT16 height)
	:
	width(width),
	height(height),
	m_title("Ioniq Rendering Engine"),
	m_hInstance(hInstance)
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
	if (!ok) {
		throw IONIQWNDEXCEPT_LAST();
	}
	m_hWnd = CreateWindow(window_class_name, m_title.c_str(), 
		WS_CAPTION | WS_MINIMIZEBOX | WS_SYSMENU, 
		CW_USEDEFAULT, CW_USEDEFAULT,
		client.right - client.left, client.bottom - client.top, 
		nullptr, nullptr, 
		hInstance, this);
	if (m_hWnd == nullptr) {
		throw IONIQWNDEXCEPT_LAST();
	}
	ok = UpdateWindow(m_hWnd);
	if (!ok) {
		throw IONIQWNDEXCEPT_LAST();
	}
	ShowWindow(m_hWnd, SW_SHOWDEFAULT);

	keyboard::init();
	mouse::init();
}

window::~window()
{
	mouse::shutdown();
	keyboard::shutdown();

	DestroyWindow(m_hWnd);
	UnregisterClass(window_class_name, m_hInstance);
}

void window::set_title(const std::string& title)
{
	m_title = title;
	BOOL ok = SetWindowText(m_hWnd, m_title.c_str());
	if (!ok) {
		throw IONIQWNDEXCEPT_LAST();
	}
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
	case WM_KILLFOCUS:
		// for some reason, killfocus is called after the window has been destroyed
		// so if an error occurs, keyboard will be null
		if (keyboard::get() != nullptr) {
			keyboard::get()->clear_states();
		}
		break;

	// keyboard messages
	case WM_SYSKEYDOWN:
	case WM_KEYDOWN:
		if (!(lParam & BIT(30))) {
			keyboard::get()->on_key_pressed((uint8_t)wParam);
		}
		break;
	case WM_SYSKEYUP:
	case WM_KEYUP:
		keyboard::get()->on_key_released((uint8_t)wParam);
		break;

	// mouse messages
	case WM_MOUSEMOVE:
	{
		const POINTS p = MAKEPOINTS(lParam);
		if (-1 < p.x && p.x < width && -1 < p.y && p.y < height) {
			mouse::get()->on_mouse_move(p.x, p.y);
			if (!mouse::get()->is_in_window()) {
				SetCapture(m_hWnd);
				mouse::get()->on_mouse_enter(p.x, p.y);
			}
		}
		else if (wParam & (MK_LBUTTON | MK_RBUTTON))
			mouse::get()->on_mouse_move(p.x, p.y);
		else {
			ReleaseCapture();
			mouse::get()->on_mouse_leave(p.x, p.y);
		}

		break;
	}
	case WM_LBUTTONDOWN:
	{
		const POINTS p = MAKEPOINTS(lParam);
		mouse::get()->on_button_pressed(mouse::button_codes::LEFT, p.x, p.y);
		break;
	}
	case WM_RBUTTONDOWN:
	{
		const POINTS p = MAKEPOINTS(lParam);
		mouse::get()->on_button_pressed(mouse::button_codes::RIGHT, p.x, p.y);
		break;
	}
	case WM_MBUTTONDOWN:
	{
		const POINTS p = MAKEPOINTS(lParam);
		mouse::get()->on_button_pressed(mouse::button_codes::MIDDLE, p.x, p.y);
		break;
	}

	case WM_LBUTTONUP:
	{
		const POINTS p = MAKEPOINTS(lParam);
		mouse::get()->on_button_released(mouse::button_codes::LEFT, p.x, p.y);
		break;
	}
	case WM_RBUTTONUP:
	{
		const POINTS p = MAKEPOINTS(lParam);
		mouse::get()->on_button_released(mouse::button_codes::RIGHT, p.x, p.y);
		break;
	}
	case WM_MBUTTONUP:
	{
		const POINTS p = MAKEPOINTS(lParam);
		mouse::get()->on_button_released(mouse::button_codes::MIDDLE, p.x, p.y);
		break;
	}

	case WM_MOUSEWHEEL:
	{
		const POINTS p = MAKEPOINTS(lParam);
		const short delta = GET_WHEEL_DELTA_WPARAM(wParam);
		mouse::get()->on_wheel_rotated(delta, p.x, p.y);
		break;
	}
	}


	return DefWindowProc(hWnd, msg, wParam, lParam);
}

window::exception::exception(int line, const std::string& file, HRESULT hr)
	:
	ioniq_exception(line, file),
	_hr(hr)
{
}

const char* window::exception::what() const
{
	std::ostringstream oss;
	oss << get_type() << std::endl;
	oss << "[Error Code]: " << _hr << std::endl;
	oss << "[Description]: " << get_description() << std::endl;
	oss << get_origin();
	m_what_buffer = oss.str();
	return m_what_buffer.c_str();
}

std::string window::exception::get_description() const
{
	char* msg = nullptr;
	DWORD len = FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
		nullptr, _hr, 0, reinterpret_cast<LPSTR>(&msg), 0, nullptr);
	if (len == 0) {
		return "Unknown error code";
	}

	std::string out = msg;
	LocalFree(msg);
	return out;
}
