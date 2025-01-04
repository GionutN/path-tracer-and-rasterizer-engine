#pragma once

#include <memory>

#define BIT(x) (1 << (x))
#define VOIDPP reinterpret_cast<void**>

template<typename T>
using scope = std::unique_ptr<T>;

template<typename T>
using ref = std::shared_ptr<T>;

typedef float real;
