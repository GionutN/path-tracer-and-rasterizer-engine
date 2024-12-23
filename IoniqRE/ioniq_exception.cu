#include "ioniq_exception.h"

#include <sstream>

ioniq_exception::ioniq_exception(int line, const std::string& file)
	:
	m_line(line),
	m_file(file)
{
}

const char* ioniq_exception::what() const
{
	std::ostringstream oss;
	oss << "[Type]: " << get_type() << std::endl;
	oss << get_origin() << std::endl;
	m_what_buffer = oss.str();
	return m_what_buffer.c_str();
}

std::string ioniq_exception::get_origin() const
{
	std::ostringstream oss;
	oss << "[File]: " << m_file << std::endl;
	oss << "[Line]: " << m_line << std::endl;
	return oss.str();
}
