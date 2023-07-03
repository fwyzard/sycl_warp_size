#include <sycl/sycl.hpp>

#ifdef __AMDGCN__

// does not work on the AMD backend
#define printf(...)

#else

#ifdef __SYCL_DEVICE_ONLY__
#define CONSTANT __attribute__((opencl_constant))
#else
#define CONSTANT
#endif  //__SYCL_DEVICE_ONLY__

#define printf(FORMAT, ...)                                         \
  do {                                                              \
    static const CONSTANT char format[] = FORMAT;                   \
    sycl::ext::oneapi::experimental::printf(format, ##__VA_ARGS__); \
  } while (false)

#endif  //__AMDGCN__

template <std::size_t S>
void do_some_work(sycl::nd_item<1> item) {
  printf("        the expected subgroup size is %d\n", S);
  printf("        the actual subgroup size is %d\n", item.get_sub_group().get_max_local_range()[0]);
}
