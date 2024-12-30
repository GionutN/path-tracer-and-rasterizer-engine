#pragma once

#include <bitset>
#include <queue>

class mouse
{
	friend class window;

public:
	// used as indices in the states array
	enum class button_codes
	{
		INVALID = -1,
		LEFT,
		RIGHT,
		MIDDLE,
		NUMCODES
	};

	class event
	{
	public:
		enum class type
		{
			INVALID = -1,
			PRESS,
			RELEASE,
			MOVE,
			ENTER,
			LEAVE,
			WHEELDOWN,
			WHEELUP,
			NUMTYPES
		};

	public:
		event(type t, button_codes c, int16_t x, int16_t y)
			:
			_c(c),
			_t(t),
			_coords({ x, y })
		{}

		inline type get_type() const { return _t; }
		inline int16_t get_x() const { return _coords.first; }
		inline int16_t get_y() const { return _coords.second; }
		inline const std::pair<int16_t, int16_t>& get_position() const { return _coords; }

	private:
		type _t;
		button_codes _c;
		std::pair<int16_t, int16_t> _coords;

	};

public:
	static void init();
	static void shutdown();
	static mouse* get();

public:
	inline bool button_is_pressed(int8_t btn) const { return m_states[btn]; }
	inline bool is_in_window() const { return m_in_window; }
	inline int16_t get_x() const { return m_coords.first; }
	inline int16_t get_y() const { return m_coords.second; }
	inline const std::pair<int16_t, int16_t>& get_position() const { return m_coords; }
	event get_event();
	event peek_event();

public:
	static constexpr size_t num_events = 16;

private:
	mouse();
	~mouse() = default;

	void on_mouse_move(int16_t x, int16_t y);
	void on_mouse_enter(int16_t x, int16_t y);
	void on_mouse_leave(int16_t x, int16_t y);
	void on_button_pressed(button_codes btn, int16_t x, int16_t y);
	void on_button_released(button_codes btn, int16_t x, int16_t y);
	void on_wheel_rotated(short delta, int16_t x, int16_t y);	// sets multiple events for high deltas
	void trim_queue();
	inline void clear_states() { m_states.reset(); }

private:
	std::bitset<3> m_states;
	std::pair<int16_t, int16_t> m_coords;
	std::queue<event> m_eventq;
	bool m_in_window;
	short total_delta;
};