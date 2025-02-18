#pragma once

#include <vector>

#include "core.h"
#include "window.h"
#include "mesh.h"
#include "shader.h"

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
	real title_time = 0.0f;
	float dt;
	int frame = 0;

	std::vector<mesh> meshes;
	std::vector<shader> shaders;
	UINT verts = 3;
};
