#include "kernel.h"
#include "sycl_subgroup_size.h"

#include <sycl/sycl.hpp>

#include <cstdio>

int main()
{
    auto platforms = sycl::platform::get_platforms();

    for(auto const& platform : platforms)
    {
        std::cout << "SYCL platform: " << platform.get_info<sycl::info::platform::name>() << '\n';
        auto devices = platform.get_devices();

        for(auto const& device : devices)
        {
            sycl::queue queue{device};

            std::cout << "  SYCL device: " << device.get_info<sycl::info::device::name>() << '\n';
            bool* supported = sycl::malloc_host<bool>(1, queue);
            bool* subgroups = sycl::malloc_host<bool>(65, queue);
            *supported = false;
            for(int i = 0; i < 65; ++i)
                subgroups[i] = false;
            try
            {
                queue
                    .submit(
                        [&](sycl::handler& cgh)
                        {
                            cgh.parallel_for(
                                sycl::nd_range<1>(1, 1),
                                [supported, subgroups](sycl::nd_item<1> item)
                                {
                                    *supported = true;
#if(SYCL_SUBGROUP_SIZE & 4)
                                    subgroups[4] = true;
#endif
#if(SYCL_SUBGROUP_SIZE & 8)
                                    subgroups[8] = true;
#endif
#if(SYCL_SUBGROUP_SIZE & 16)
                                    subgroups[16] = true;
#endif
#if(SYCL_SUBGROUP_SIZE & 32)
                                    subgroups[32] = true;
#endif
#if(SYCL_SUBGROUP_SIZE & 64)
                                    subgroups[64] = true;
#endif
                                });
                        })
                    .wait();
            }
            catch(std::exception const& e)
            {
                std::cerr << "  " << e.what() << '\n';
            }
            if(not *supported)
            {
                std::cout << "  this device is not supported by this binary.\n";
                continue;
            }
            else
            {
                std::cout << "  sub-group sizes supported by the compiler: ";
                bool first = true;
                for(int i = 0; i < 65; ++i)
                    if(subgroups[i])
                    {
                        if(not first)
                            std::cout << ", ";
                        std::cout << i;
                        first = false;
                    }
                if(first)
                {
                    std::cout << "(undefined)";
                }
                std::cout << '\n';
            }

            sycl::free(supported, queue);
            sycl::free(subgroups, queue);

            auto sizes = device.get_info<sycl::info::device::sub_group_sizes>();
            std::cout << "  sub-group sizes supported by the device: " << sizes[0];
            for(int i = 1; i < sizes.size(); ++i)
            {
                std::cout << ", " << sizes[i];
            }
            std::cout << '\n';

            std::size_t* expected = sycl::malloc_host<std::size_t>(1, queue);
            std::size_t* actual = sycl::malloc_host<std::size_t>(1, queue);

            std::cout << "\n    test automatic sub-group size\n";
            *expected = 0;
            *actual = 0;
            launch_kernel<0>(queue, sycl::nd_range<1>(1, 1), do_some_work<0>{}, expected, actual).wait();
            std::cout << "    the automatic sub-group size is " << *actual << '\n';

            for(int size : sizes)
            {
                std::cout << "\n    test sub-group of " << size << " elements\n";
                *expected = 0;
                *actual = 0;
                if(size == 4)
                {
                    launch_kernel<4>(queue, sycl::nd_range<1>(1, 1), do_some_work<4>{}, expected, actual).wait();
                }
                if(size == 8)
                {
                    launch_kernel<8>(queue, sycl::nd_range<1>(1, 1), do_some_work<8>{}, expected, actual).wait();
                }
                if(size == 16)
                {
                    launch_kernel<16>(queue, sycl::nd_range<1>(1, 1), do_some_work<16>{}, expected, actual).wait();
                }
                if(size == 32)
                {
                    launch_kernel<32>(queue, sycl::nd_range<1>(1, 1), do_some_work<32>{}, expected, actual).wait();
                }
                if(size == 64)
                {
                    launch_kernel<64>(queue, sycl::nd_range<1>(1, 1), do_some_work<64>{}, expected, actual).wait();
                }

                if(*actual)
                {
                    std::cout << "    the expected sub-group size is " << *expected << '\n';
                    std::cout << "    the actual sub-group size is " << *actual << '\n';
                }
                else
                {
                    std::cout << "    unsupported sub-group size\n";
                }
            }

            sycl::free(expected, queue);
            sycl::free(actual, queue);
        }
        std::cout << '\n';
    }
}
