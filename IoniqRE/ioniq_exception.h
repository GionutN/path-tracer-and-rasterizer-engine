#pragma once

#include <exception>
#include <string>

class ioniq_exception : public std::exception
{
public:
	ioniq_exception(int line, const std::string& file);

	const char* what() const override;
	virtual inline const char* get_type() const { return "Ioniq Base Exception"; }
	std::string get_origin() const;

protected:
	mutable std::string m_what_buffer;

private:
	int m_line;
	std::string m_file;

};
