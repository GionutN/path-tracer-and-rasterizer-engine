#pragma once

#include <bitset>
#include <queue>

class keyboard
{
	friend class window;

public:
	class event
	{
	public:
		enum class type
		{
			INVALID = -1,
			PRESS,
			RELEASE,
			NUMTYPES
		};

	public:
		event(type t, uint8_t key)
			:
			_t(t),
			_key(key)
		{}

		inline type get_type() const { return _t; }
		inline uint8_t get_key() const { return _key; }

	private:
		type _t;
		uint8_t _key;

	};

public:
	static void init();
	static void shutdown();
	static keyboard* get();

public:
	inline bool key_is_pressed(uint8_t key) const { return m_states[key]; }
	event get_event();
	event peek_event() const;

public:
	static constexpr size_t num_events = 16;

private:
	void on_key_pressed(uint8_t key);
	void on_key_released(uint8_t key);
	void trim_queue();
	inline void clear_states() { m_states.reset(); }

public:
	static constexpr size_t num_keys = 256;

private:
	keyboard();
	~keyboard() = default;

private:
	std::bitset<num_keys> m_states;
	std::queue<event> m_eventq;

};
