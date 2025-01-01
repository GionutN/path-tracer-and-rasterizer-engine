#include "application.h"

#include <sstream>

#include "keyboard.h"
#include "mouse.h"
#include "timer.h"

application::application(const ref<window>& wnd)
	:
	m_wnd(wnd)
{
}

bool application::process_message()
{
	MSG msg;
	while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
		if (msg.message == WM_QUIT)
			return false;
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}

	return true;
}

void application::run()
{
	update_frame();
	draw_frame();
}

void application::update_frame()
{
	real dt = timer::get()->get_delta();
	title_time += dt;
	if (title_time > 1.0) {
		std::ostringstream oss;
		oss << "FPS: " << 1 / dt;
		m_wnd->set_title(oss.str());
		title_time = 0.0;
	}
}

void application::draw_frame()
{
}
