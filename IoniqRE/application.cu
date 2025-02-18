#include "application.h"

#include <sstream>

#include "keyboard.h"
#include "mouse.h"
#include "timer.h"
#include "renderer.h"

application::application(const ref<window>& wnd)
	:
	m_wnd(wnd)
{
	timer::init();
	renderer::init(wnd);

	meshes.emplace_back(reg_polygon(3));
	shaders.emplace_back(L"vertex_shader.cso", L"pixel_shader.cso");
}

application::~application()
{
	renderer::shutdown();
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
	real dt = timer::get()->get_delta();
	get_fps(dt);

	switch (mouse::get()->get_event().get_type()) {
	case mouse::event::type::WHEELUP:
		verts++;
		meshes[0] = reg_polygon(verts);
		break;
	case mouse::event::type::WHEELDOWN:
		verts--;
		meshes[0] = reg_polygon(verts);
		break;
	}
	if (verts < 3) verts = 3;
	if (verts > 100) verts = 100;
}

void application::draw_frame()
{
	renderer::get()->draw_scene(meshes, shaders);
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
