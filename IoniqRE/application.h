#pragma once

#include "core.h"
#include "window.h"

class application
{
public:
	application(const ref<window>& wnd);
	application(const application&) = delete;
	application& operator=(const application&) = delete;

	bool process_message();
	void update_frame();
	void draw_frame();
	void run();

private:
	// use shared ptr instead of unique because it can not be instantiated from the ptr in main
	ref<window> m_wnd;
	int numd = 0, numu = 0;

};
