#include "mouse.h"

static mouse* g_mouse;

mouse::mouse()
{
	m_states = std::bitset<2>();
	m_coords = { NULL, NULL };
	m_in_window = true;
	m_delta = NULL;
}

void mouse::init()
{
	if (!g_mouse) {
		g_mouse = new mouse();
	}
}

void mouse::shutdown()
{
	if (g_mouse) {
		delete g_mouse;
		g_mouse = nullptr;
	}
}

mouse* mouse::get()
{
	return g_mouse;
}
