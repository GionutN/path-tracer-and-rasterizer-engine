#pragma once

#include "ioniq_windows.h"

class window
{
public:
	window(HINSTANCE hInstance, UINT16 width, UINT16 height);
	window(const window&) = delete;
	window& operator=(const window&) = delete;
	~window();

	bool should_close() const;

private:
	static LRESULT CALLBACK HandleMessageSetup(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
	static LRESULT CALLBACK HandleMessageDummy(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
	LRESULT CALLBACK HandleMessage(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

public:
	static constexpr const char* window_class_name = "Ioniq Window Class";

private:
	const UINT16 m_width;
	const UINT16 m_height;
	HWND m_hWnd;

};