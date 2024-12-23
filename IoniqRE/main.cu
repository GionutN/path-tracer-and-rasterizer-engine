#include "ioniq_windows.h"

#include "window.h"

int WINAPI WinMain(HINSTANCE hInstance,
	HINSTANCE hPrevInstance,
	LPSTR lpCmdLine,
	int nCmdShow)
{
	try {
		window w(hInstance, 1920, 1080);
		while (!w.should_close());
		return EXIT_SUCCESS;
	}
	catch (const ioniq_exception & e) {
		MessageBox(nullptr, e.what(), e.get_type(), MB_OK | MB_ICONEXCLAMATION);
	}
	catch (const std::exception& e) {
		MessageBox(nullptr, e.what(), "Standard exception", MB_OK | MB_ICONEXCLAMATION);
	}
	catch (...) {
		MessageBox(nullptr, "No details available", "Unknown exception type", MB_OK | MB_ICONEXCLAMATION);
	}

	return EXIT_FAILURE;
}
