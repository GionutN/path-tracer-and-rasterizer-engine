#pragma once

#include "core.h"
#include "window.h"

class application
{
public:
	application(window* wnd);
	application(const application&) = delete;
	application& operator=(const application&) = delete;

	bool process_message();
	void update_frame();
	void draw_frame();
	void run();

private:
	scope<window> m_wnd;

};
