#include "mouse.h"

static mouse* g_mouse;

mouse::mouse()
{
	m_states = std::bitset<3>();
	m_coords = { NULL, NULL };
	m_in_window = true;
	m_eventq = std::queue<event>();
	total_delta = 0;
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

bool mouse::button_is_pressed(button_codes btn) const
{
	int code = (int)btn;
	if (code < 0 || code >= (int)button_codes::NUMCODES) {
		return false;
	}

	return m_states[code];
}

mouse::event mouse::get_event()
{
	if (m_eventq.empty()) {
		return event(event::type::INVALID, button_codes::INVALID, NULL, NULL);
	}

	event result = m_eventq.front();
	m_eventq.pop();
	return result;
}

mouse::event mouse::peek_event() const
{
	if (m_eventq.empty()) {
		return event(event::type::INVALID, button_codes::INVALID, NULL, NULL);
	}

	return m_eventq.front();
}

void mouse::on_mouse_move(int16_t x, int16_t y)
{
	m_coords = { x, y };
	m_eventq.emplace(event::type::MOVE, button_codes::INVALID, x, y);
	trim_queue();
}

void mouse::on_mouse_enter(int16_t x, int16_t y)
{
	m_in_window = true;
	m_eventq.emplace(event::type::ENTER, button_codes::INVALID, x, y);
	trim_queue();
}

void mouse::on_mouse_leave(int16_t x, int16_t y)
{
	m_in_window = false;
	m_eventq.emplace(event::type::LEAVE, button_codes::INVALID, x, y);
	trim_queue();
}

void mouse::on_button_pressed(button_codes btn, int16_t x, int16_t y)
{
	m_states[(int)btn] = true;
	m_eventq.emplace(event::type::PRESS, btn, x, y);
	trim_queue();
}

void mouse::on_button_released(button_codes btn, int16_t x, int16_t y)
{
	m_states[(int)btn] = false;
	m_eventq.emplace(event::type::RELEASE, btn, x, y);
	trim_queue();
}

void mouse::on_wheel_rotated(short delta, int16_t x, int16_t y)
{
	const short wheel_delta = 120;

	total_delta += delta;
	while (total_delta >= wheel_delta) {
		m_eventq.emplace(event::type::WHEELUP, button_codes::INVALID, x, y);
		trim_queue();
		total_delta -= wheel_delta;
	}
	while (total_delta <= -wheel_delta) {
		m_eventq.emplace(event::type::WHEELDOWN, button_codes::INVALID, x, y);
		trim_queue();
		total_delta += wheel_delta;
	}
}

void mouse::trim_queue()
{
	while (m_eventq.size() > num_events) {
		m_eventq.pop();
	}
}
