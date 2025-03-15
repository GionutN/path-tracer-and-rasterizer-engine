#pragma once

#include "scene.h"
#include "shader.h"

class renderer_template
{
public:
	virtual void begin_frame() = 0;
	virtual void end_frame() = 0;
	virtual void draw_scene(const scene& scene, const std::vector<shader>& shaders, float dt) = 0;
};
