#include "application.h"

#include <sstream>

#include "keyboard.h"
#include "mouse.h"
#include "timer.h"
#include "renderer.h"
#include "random.h"

application::application(const ref<window>& wnd)
	:
	m_wnd(wnd),
	cam(wnd->width, wnd->height)
{
	timer::init();
	random::init();
	renderer::init(wnd);

	shaders.emplace_back(L"vertex_shader.cso", L"pixel_shader.cso");

	scn.add_mesh("default", tri());
	scn.add_mesh("cube", cube());
	scn.add_mesh("sphere", uv_sphere(false, 256, 128));

	scn.add_model("main", model("cube"));
	//scn.get_model("main").set_transforms(1.0f, iqvec(0.0f, 0.0f, 0.0f, 0.0f), iqvec(0.0f, 0.0f, 10.0f, 0.0f));

	// TODO:
	// add the depth stencil buffer
	// get the camera working for the path tracer
	// add phong lighting

	shaders[0].update_view_proj(cam.get_view(), cam.get_projection());
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
			//renderer::get()->toggle_engine();
		}
	}

	if (mouse::get()->button_is_pressed(mouse::button_codes::RIGHT)) {
		renderer::get()->reset();
	}

	radians += dt;
	radians = std::fmodf(radians, tau);
	scn.get_model("main").set_rotation(iqvec(radians, 0.0f, radians, 0.0f));
}

void application::draw_frame()
{
	renderer::get()->draw_scene(scn, shaders, dt);
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
