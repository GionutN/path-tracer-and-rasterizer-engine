#include "timer.h"

static timer* g_timer;

using namespace std::chrono;
using namespace std::chrono_literals;

void timer::init()
{
	if (!g_timer) {
		g_timer = new timer();
	}
}

void timer::shutdown()
{
	if (g_timer) {
		delete g_timer;
		g_timer = nullptr;
	}
}

timer* timer::get()
{
	return g_timer;
}

timer::timer()
	:
	m_start(high_resolution_clock::now()),
	m_last(high_resolution_clock::now())
{}

real timer::get_delta()
{
	const auto old = m_last;
	m_last = high_resolution_clock::now();
	const duration_seconds dur = m_last - old;
	return dur.count();
}
