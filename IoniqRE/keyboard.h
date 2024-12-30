#pragma once

#include <bitset>

class keyboard
{
	friend class window;
public:
	static void init();
	static void shutdown();
	static keyboard* get();

public:
	inline bool key_is_pressed(uint8_t key) const { return m_states[key]; }

private:
	inline void on_key_pressed (uint8_t key) { m_states[key] = true; }
	inline void on_key_released(uint8_t key) { m_states[key] = false; }
	inline void clear_states() { m_states.reset(); }

public:
	static constexpr size_t num_keys = 256;

private:
	keyboard();
	~keyboard() = default;

private:
	std::bitset<num_keys> m_states;

};
