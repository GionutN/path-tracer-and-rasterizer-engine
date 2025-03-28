#include "renderer.h"

#include <sstream>
#include <d3dcompiler.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "mesh.h"
#include "shader.h"
#include "iqmath.h"

#include "rasterizer.h"
#include "path_tracer.h"

static renderer* g_renderer = nullptr;

namespace wrl = Microsoft::WRL;

void renderer::init(const ref<window>& wnd)
{
	if (!g_renderer) {
		renderer_base::init(wnd);
		rasterizer::init();
		path_tracer::init();
		g_renderer = new renderer();
	}
}

void renderer::shutdown()
{
	if (g_renderer) {
		path_tracer::shutdown();
		rasterizer::shutdown();
		renderer_base::shutdown();
		delete g_renderer;
		g_renderer = nullptr;
	}
}

renderer* renderer::get()
{
	return g_renderer;
}

void renderer::begin_frame()
{
	// make sure the engine is not changing mid-frame
	if (m_new_engine != m_engine_idx) {
		m_engine_idx = m_new_engine;
	}
	
	m_engines[(size_t)m_engine_idx]->begin_frame();
}

void renderer::end_frame()
{
	m_engines[(size_t)m_engine_idx]->end_frame();
}

void renderer::draw_scene(const scene& scene, const std::vector<shader>& shaders, float dt)
{
	m_engines[(size_t)m_engine_idx]->draw_scene(scene, shaders, dt);
}

renderer::renderer()
	:
	m_engine_idx(engine::PATHTRACER)
{
	m_new_engine = m_engine_idx;
	m_engines[0] = rasterizer::get();
	m_engines[1] = path_tracer::get();
}
