#include <Windows.h>

LRESULT CALLBACK HandleMessage(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	switch (msg) {
	case WM_QUIT:
	case WM_CLOSE:
		PostQuitMessage(EXIT_SUCCESS);
		break;
	}

	return DefWindowProc(hWnd, msg, wParam, lParam);
}

bool should_close() {
	MSG msg;
	while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE) != 0) {
		if (msg.message == WM_QUIT)
			return true;
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}

	return false;
}

int WINAPI WinMain(HINSTANCE hInstance,
	HINSTANCE hPrevInstance,
	LPSTR lpCmdLine,
	int nCmdShow)
{
	HRESULT hr;
	BOOL b;

	// fill out the window class struct
	WNDCLASSEX wndClass = {};
	wndClass.cbSize = sizeof(WNDCLASSEX);
	wndClass.style = CS_OWNDC;
	wndClass.lpfnWndProc = HandleMessage;
	wndClass.cbClsExtra = 0;
	wndClass.cbWndExtra = 0;
	wndClass.hInstance = hInstance;
	wndClass.hIcon = nullptr;
	wndClass.hCursor = nullptr;
	wndClass.hbrBackground = nullptr;
	wndClass.lpszMenuName = nullptr;
	wndClass.lpszClassName = "Ioniq Window Class";
	wndClass.hIconSm = nullptr;
	hr = RegisterClassEx(&wndClass);
	if (hr == 0) {
		OutputDebugString("Could not register window class");
		return EXIT_FAILURE;
	}

	// create the window
	RECT client = {};
	client.left = 1000;
	client.top = 200;
	client.right = client.left + 1920;
	client.bottom = client.top + 1080;
	b = AdjustWindowRect(&client, WS_CAPTION | WS_MINIMIZEBOX | WS_SYSMENU, FALSE);
	HWND hWnd = CreateWindow("Ioniq Window Class", "Ioniq Rendering Engine", WS_CAPTION | WS_MINIMIZEBOX | WS_SYSMENU, CW_USEDEFAULT, CW_USEDEFAULT,
		client.right - client.left, client.bottom - client.top, nullptr, nullptr, hInstance, nullptr);
	if (!hWnd) {
		OutputDebugString("Could not create window");
		return EXIT_FAILURE;
	}
	UpdateWindow(hWnd);
	ShowWindow(hWnd, SW_SHOWDEFAULT);

	while (!should_close()) {}

	return EXIT_SUCCESS;
}
