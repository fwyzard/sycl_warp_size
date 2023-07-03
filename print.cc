#include <cstdio>

#include <sycl/sycl.hpp>

#include "warpsize.h"
#include "kernel.h"

int main() {
  auto platforms = sycl::platform::get_platforms();

  for (auto const& platform : platforms) {
    std::cout << "SYCL platform: " << platform.get_info<sycl::info::platform::name>() << '\n';
    auto devices = platform.get_devices();

    for (auto const& device : devices) {
      sycl::queue queue{device};

      std::cout << "  SYCL device: " << device.get_info<sycl::info::device::name>() << '\n';
      bool* supported = sycl::malloc_host<bool>(1, queue);
      *supported = false;
      try {
        queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(sycl::nd_range<1>(1, 1), [=](sycl::nd_item<1> item) {
            bool first = true;
            printf("%s", "  sub-group sizes supported by the compiler: ");
#if defined SYCL_HAS_WARP_SIZE_4
            if (not first)
              printf("%s", ", ");
            printf("%d", 4);
            first = false;
#endif
#if defined SYCL_HAS_WARP_SIZE_8
            if (not first)
              printf("%s", ", ");
            printf("%d", 8);
            first = false;
#endif
#if defined SYCL_HAS_WARP_SIZE_16
            if (not first)
              printf("%s", ", ");
            printf("%d", 16);
            first = false;
#endif
#if defined SYCL_HAS_WARP_SIZE_32
            if (not first)
              printf("%s", ", ");
            printf("%d", 32);
            first = false;
#endif
#if defined SYCL_HAS_WARP_SIZE_64
            if (not first)
              printf("%s", ", ");
            printf("%d", 64);
            first = false;
#endif
            printf("%s", "\n");
            *supported = true;
          });
        }).wait();
      } catch(...) {}
      if (not *supported) {
        std::cout << "  this device is not supported by this binary.\n\n";
        continue;
      }

      auto sizes = device.get_info<sycl::info::device::sub_group_sizes>();
      std::cout << "  sub-group sizes supported by the device: " << sizes[0];
      for (int i = 1; i < sizes.size(); ++i) {
        std::cout << ", " << sizes[i];
      }
      std::cout << '\n';

      for (int size : sizes) {
        std::cout << "\n      test sub-group of " << size << " elements\n";
        int threads = 1;
        int blocks = 1;
        queue.submit([&](sycl::handler& cgh) {
#if defined SYCL_HAS_WARP_SIZE_4
          if (size == 4) {
            std::cout << "      sub-group size of 4 is being tested\n";
            cgh.parallel_for(sycl::nd_range<1>(blocks * threads, threads), [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(4)]] { do_some_work<4>(item); });
          } else
#endif
#if defined SYCL_HAS_WARP_SIZE_8
              if (size == 8) {
            std::cout << "      sub-group size of 8 is being tested\n";
            cgh.parallel_for(sycl::nd_range<1>(blocks * threads, threads), [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(8)]] { do_some_work<8>(item); });
          } else
#endif
#if defined SYCL_HAS_WARP_SIZE_16
              if (size == 16) {
            std::cout << "      sub-group size of 16 is being tested\n";
            cgh.parallel_for(sycl::nd_range<1>(blocks * threads, threads), [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(16)]] { do_some_work<16>(item); });
          } else
#endif
#if defined SYCL_HAS_WARP_SIZE_32
              if (size == 32) {
            std::cout << "      sub-group size of 32 is being tested\n";
            cgh.parallel_for(sycl::nd_range<1>(blocks * threads, threads), [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(32)]] { do_some_work<32>(item); });
          } else
#endif
#if defined SYCL_HAS_WARP_SIZE_64
              if (size == 64) {
            std::cout << "      sub-group size of 64 is being tested\n";
            cgh.parallel_for(sycl::nd_range<1>(blocks * threads, threads), [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(64)]] { do_some_work<64>(item); });
          } else
#endif
            std::cout << "      unsupported sub-group size\n";
        }).wait();
      }
    }
    std::cout << '\n';
  }
  std::cout << '\n';
}
