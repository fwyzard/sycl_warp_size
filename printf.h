#pragma once

#include <cstdio>
#include <sycl/sycl.hpp>

#ifdef __SYCL_DEVICE_ONLY__
#define __DEVICE_CONSTANT__ [[clang::opencl_constant]]
#else
#define __DEVICE_CONSTANT__
#endif

// if printf is defined as a macro, undefine it
#ifdef printf
#undef printf
#endif

#define printf(FORMAT, ...)                                                     \
  do {                                                                          \
    static const char* __DEVICE_CONSTANT__ format = FORMAT;                     \
    sycl::ext::oneapi::experimental::printf(format, ##__VA_ARGS__);             \
  } while(false)
