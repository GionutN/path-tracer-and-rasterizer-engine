#pragma once

#include "ioniq_windows.h"
#include "ioniq_exception.h"

class window
{
public:
	class exception : public ioniq_exception
	{
	public:
		exception(int line, const std::string& file, HRESULT hr);

		const char* what() const override;
		inline const char* get_type() const override { return "Ioniq Window Exception"; }

	private:
		std::string get_description() const;

	private:
		HRESULT m_hr;
	};

public:
	window(HINSTANCE hInstance, UINT16 width, UINT16 height);
	window(const window&) = delete;
	window& operator=(const window&) = delete;
	~window();

	void set_title(const std::string& title);

private:
	static LRESULT CALLBACK HandleMessageSetup(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
	static LRESULT CALLBACK HandleMessageDummy(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
	LRESULT CALLBACK HandleMessage(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

public:
	static constexpr const char* window_class_name = "Ioniq Window Class";

private:
	HWND m_hWnd;
	HINSTANCE m_hInstance;

	const UINT16 m_width;
	const UINT16 m_height;
	std::string m_title;

};

#define IONIQWNDEXCEPT(hr) window::exception(__LINE__, __FILE__, hr)
#define IONIQWNDEXCEPT_LAST() window::exception(__LINE__, __FILE__, GetLastError())
