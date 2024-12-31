#include "keyboard.h"

static keyboard* g_kbd;

keyboard::keyboard()
{
	m_states = std::bitset<num_keys>();
	m_eventq = std::queue<event>();
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

keyboard::event keyboard::get_event()
{
	if (m_eventq.empty()) {
		return event(event::type::INVALID, NULL);
	}

	event e = m_eventq.front();
	m_eventq.pop();
	return e;
}
keyboard::event keyboard::peek_event() const
{
	if (m_eventq.empty()) {
		return event(event::type::INVALID, NULL);
	}

	return m_eventq.front();
}

void keyboard::on_key_pressed(uint8_t key)
{
	m_states[key] = true;
	m_eventq.emplace(event::type::PRESS, key);
	trim_queue();
}

void keyboard::on_key_released(uint8_t key)
{
	m_states[key] = false;
	m_eventq.emplace(event::type::RELEASE, key);
	trim_queue();
}

void keyboard::trim_queue()
{
	while (m_eventq.size() > num_events) {
		m_eventq.pop();
	}
}
