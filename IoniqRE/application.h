#pragma once

#include "core.h"
#include "window.h"

class application
{
public:
	application(const ref<window>& wnd);
	application(const application&) = delete;
	application& operator=(const application&) = delete;
	~application();

	bool process_message();
	void update_frame();
	void draw_frame();
	void run();

private:
	void get_fps(real dt);

private:
	// use shared ptr instead of unique because it can not be instantiated from the ptr in main
	ref<window> m_wnd;
	real title_time = 0.0;
	int frame = 0;

};
