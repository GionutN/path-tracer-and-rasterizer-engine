#include "ioniq_windows.h"

#include "window.h"

int WINAPI WinMain(HINSTANCE hInstance,
	HINSTANCE hPrevInstance,
	LPSTR lpCmdLine,
	int nCmdShow)
{
	window w(hInstance, 1920, 1080);
	while (!w.should_close()) {}

	return 0;
}
