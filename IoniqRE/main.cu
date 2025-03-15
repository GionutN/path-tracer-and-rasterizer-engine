#include "ioniq_windows.h"

#include "core.h"
#include "window.h"
#include "application.h"
#include "renderer.h"

int WINAPI WinMain(HINSTANCE hInstance,
	HINSTANCE hPrevInstance,
	LPSTR lpCmdLine,
	int nCmdShow)
{
	try {
		// make a shared ptr to a window because a unique ptr can not be passed to the application constructor
		ref<window> w = std::make_shared<window>(hInstance);
		application app(w);

		while (app.process_message()) {
			app.run();
		}

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
