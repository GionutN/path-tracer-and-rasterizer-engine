#pragma once

#include "ioniq_windows.h"
#include <d3d11.h>
#include <wrl.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "renderer_base.h"
#include "renderer_template.h"

class renderer : public renderer_template
{
public:
	enum class engine
	{
		INVALID = -1,
		RASTERIZER,
		PATHTRACER,
		NUMENGINES
	};

public:
	static void init(const ref<window>& wnd);
	static void shutdown();
	static renderer* get();

	void begin_frame() override;
	void end_frame() override ;
	void draw_scene(const scene& scene, const std::vector<shader>& shaders, float dt) override;

	void toggle_engine() { m_new_engine = (engine)(((int)m_new_engine + 1) % 2); }

private:
	renderer();
	~renderer() = default;

private:
	engine m_new_engine, m_engine_idx;
	renderer_template* m_engines[2];

};
