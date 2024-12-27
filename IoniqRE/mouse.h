#pragma once

#include <bitset>

// TODO: add a Event class and queue for both mouse and keyboard
class mouse
{
	friend class window;
public:
	static void init();
	static void shutdown();
	static mouse* get();

public:
	inline bool button_is_pressed(uint8_t btn) const { return m_states[btn]; }
	inline bool is_in_window() const { return m_in_window; }
	inline const std::pair<int16_t, int16_t>& get_position() const { return m_coords; }
	inline short is_wheel_up() const { return m_delta > 0; }
	inline short is_wheel_down() const { return m_delta < 0; }

private:
	inline void on_mouse_move(int16_t x, int16_t y) { m_coords = { x, y }; }
	inline void on_mouse_enter() { m_in_window = true; }
	inline void on_mouse_leave() { m_in_window = false; }
	inline void on_button_pressed (uint8_t btn) { m_states[btn] = true; }
	inline void on_button_released(uint8_t btn) { m_states[btn] = false; }
	inline void on_wheel_rotated(short delta) { m_delta = delta; }
	inline void clear_states() { m_states.reset(); }

private:
	mouse();
	~mouse() = default;

private:
	std::bitset<2> m_states;
	bool m_in_window;
	std::pair<int16_t, int16_t> m_coords;
	short m_delta;

};