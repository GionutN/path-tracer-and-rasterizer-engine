#include "keyboard.h"

static keyboard* g_kbd;

keyboard::keyboard()
{
	m_states = std::bitset<num_keys>();
}

void keyboard::init()
{
	if (!g_kbd) {
		g_kbd = new keyboard();
	}
}

void keyboard::shutdown()
{
	if (g_kbd) {
		delete g_kbd;
		g_kbd = nullptr;
	}
}

keyboard* keyboard::get()
{
	return g_kbd;
}
