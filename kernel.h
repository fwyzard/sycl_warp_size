#include "sycl_subgroup_size.h"

#include <sycl/sycl.hpp>


template<std::size_t S>
struct do_some_work
{
    void operator()(sycl::nd_item<1> item, std::size_t* expected, std::size_t* actual) const
    {
        *expected = S;
        *actual = item.get_sub_group().get_max_local_range()[0];
    }
};


template<int S, int D, typename F, typename... Args>
sycl::event launch_kernel(sycl::queue queue, sycl::nd_range<D> range, F&& f, Args&&... args)
{
  // Note: S will be determined as a trait of F

  if constexpr(S == 0) {
    // no explicit subgroup size requirement
    return queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            range,
            [f, args...](sycl::nd_item<D> item) {
                        f(item, args...);
            });
    });
  } else {
#if (SYCL_SUBGROUP_SIZE == 0)
    // no explicit SYCL target, assume JIT compilation
    return queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            range,
            [f, args...](sycl::nd_item<D> item) [[intel::reqd_sub_group_size(S)]] {
                        f(item, args...);
            });
    });
#else
    // check if the kernel should be launched with a subgroup size of 4
    if constexpr(S == 4) {
#  if (SYCL_SUBGROUP_SIZE & 4)
      // launch the kernel with a subgroup size of 4
      return queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(
              range,
              [f, args...](sycl::nd_item<D> item) [[intel::reqd_sub_group_size(S)]] {
                          f(item, args...);
              });
      });
#  else
      // this subgroup size is not support, raise an exception
      throw sycl::errc::kernel_not_supported;
      // empty kernel, required to keep SYCL happy
      return queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(
              range,
              [f, args...](sycl::nd_item<D> item) {
              });
      });
#  endif
    }

    // check if the kernel should be launched with a subgroup size of 8
    if constexpr(S == 8) {
#  if (SYCL_SUBGROUP_SIZE & 8)
      // launch the kernel with a subgroup size of 8
      return queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(
              range,
              [f, args...](sycl::nd_item<D> item) [[intel::reqd_sub_group_size(S)]] {
                          f(item, args...);
              });
      });
#  else
      // this subgroup size is not support, raise an exception
      throw sycl::errc::kernel_not_supported;
      // empty kernel, required to keep SYCL happy
      return queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(
              range,
              [f, args...](sycl::nd_item<D> item) {
              });
      });
#  endif
    }

    // check if the kernel should be launched with a subgroup size of 16
    if constexpr(S == 16) {
#  if (SYCL_SUBGROUP_SIZE & 16)
      // launch the kernel with a subgroup size of 16
      return queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(
              range,
              [f, args...](sycl::nd_item<D> item) [[intel::reqd_sub_group_size(S)]] {
                          f(item, args...);
              });
      });
#  else
      // this subgroup size is not support, raise an exception
      throw sycl::errc::kernel_not_supported;
      // empty kernel, required to keep SYCL happy
      return queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(
              range,
              [f, args...](sycl::nd_item<D> item) {
              });
      });
#  endif
    }

    // check if the kernel should be launched with a subgroup size of 32
    if constexpr(S == 32) {
#  if (SYCL_SUBGROUP_SIZE & 32)
      // launch the kernel with a subgroup size of 32
      return queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(
              range,
              [f, args...](sycl::nd_item<D> item) [[intel::reqd_sub_group_size(S)]] {
                          f(item, args...);
              });
      });
#  else
      // this subgroup size is not support, raise an exception
      throw sycl::errc::kernel_not_supported;
      // empty kernel, required to keep SYCL happy
      return queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(
              range,
              [f, args...](sycl::nd_item<D> item) {
              });
      });
#  endif
    }

    // check if the kernel should be launched with a subgroup size of 64
    if constexpr(S == 64) {
#  if (SYCL_SUBGROUP_SIZE & 64)
      // launch the kernel with a subgroup size of 64
      return queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(
              range,
              [f, args...](sycl::nd_item<D> item) [[intel::reqd_sub_group_size(S)]] {
                          f(item, args...);
              });
      });
#  else
      // this subgroup size is not support, raise an exception
      throw sycl::errc::kernel_not_supported;
      // empty kernel, required to keep SYCL happy
      return queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(
              range,
              [f, args...](sycl::nd_item<D> item) {
              });
      });
#  endif
    }
#endif

    // this subgroup size is not support, raise an exception
    throw sycl::errc::kernel_not_supported;
  }
}
