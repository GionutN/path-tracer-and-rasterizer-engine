#include "application.h"

#include <sstream>

#include "keyboard.h"
#include "mouse.h"
#include "timer.h"
#include "renderer.h"
#include "random.h"

application::application(const ref<window>& wnd)
	:
	m_wnd(wnd)
{
	timer::init();
	random::init();
	renderer::init(wnd);

	meshes.add(tri());
	shaders.emplace_back(L"vertex_shader.cso", L"pixel_shader.cso");
}

application::~application()
{
	renderer::shutdown();
	random::shutdown();
	timer::shutdown();
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
	renderer::get()->begin_frame();
	update_frame();
	draw_frame();
	renderer::get()->end_frame();
}

void application::update_frame()
{
	dt = timer::get()->get_delta();
	get_fps(dt);

	keyboard::event e = keyboard::get()->get_event();
	switch (e.get_type()) {
	case keyboard::event::type::PRESS:
		if (e.get_key() == 'P') {
			renderer::get()->toggle_engine();
		}
	}
}

void application::draw_frame()
{
	renderer::get()->draw_scene(meshes, shaders, dt);
}

void application::get_fps(real dt)
{
	frame++;
	title_time += dt;
	if (title_time > 1.0f) {
		std::ostringstream oss;
		oss << "FPS: " << frame << ' ';
		oss << '(' << 1000.0f / frame << "ms)";
		m_wnd->set_title(oss.str());
		title_time = 0.0f;
		frame = 0;
	}
}
