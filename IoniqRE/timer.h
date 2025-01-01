#pragma once

#include <chrono>

#include "core.h"

class timer
{
	using duration_seconds = std::chrono::duration<real>;

public:
	static void init();
	static void shutdown();
	static timer* get();

	inline real get_total_time() const { return (std::chrono::high_resolution_clock::now() - m_start).count(); }
	real get_delta();

private:
	timer();
	~timer() = default;

private:
	std::chrono::high_resolution_clock::time_point m_last;
	const std::chrono::high_resolution_clock::time_point m_start;

};