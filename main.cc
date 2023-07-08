#include "kernel.h"
#include "launch.h"
#include "sycl_subgroup_size.h"

#include <sycl/sycl.hpp>

#include <iostream>

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
            uint32_t* subgroups = sycl::malloc_host<uint32_t>(1, queue);
            *supported = false;
            *subgroups = 0;
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
#ifdef SYCL_SUBGROUP_SIZE
                                    *subgroups = SYCL_SUBGROUP_SIZE;
#else
                                    *subgroups = 0;
#endif
                                    *supported = true;
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
                if (*subgroups == 0U) {
                    std::cout << "(undefined)";
                } else {
                    bool first = true;
                    for(uint32_t size = 1; size; size <<= 1)
                        if(*subgroups & size)
                        {
                            if(not first)
                                std::cout << ", ";
                            std::cout << size;
                            first = false;
                        }
                }
                std::cout << '\n';
            }

            auto sizes = device.get_info<sycl::info::device::sub_group_sizes>();
            std::cout << "  sub-group sizes supported by the device: " << sizes[0];
            for(int i = 1; i < sizes.size(); ++i)
            {
                std::cout << ", " << sizes[i];
            }
            std::cout << '\n';

            std::cout << "\n    test automatic sub-group size:\n";
            launch(queue, sycl::nd_range<1>(1, 1), do_some_work<0>{}, supported).wait();

            for(int size : sizes)
            {
                *supported = false;

                std::cout << "\n    test sub-group of " << size << " elements:\n";
                if(size == 4)
                {
                    launch(queue, sycl::nd_range<1>(1, 1), do_some_work<4>{}, supported).wait();
                }
                if(size == 8)
                {
                    launch(queue, sycl::nd_range<1>(1, 1), do_some_work<8>{}, supported).wait();
                }
                if(size == 16)
                {
                    launch(queue, sycl::nd_range<1>(1, 1), do_some_work<16>{}, supported).wait();
                }
                if(size == 32)
                {
                    launch(queue, sycl::nd_range<1>(1, 1), do_some_work<32>{}, supported).wait();
                }
                if(size == 64)
                {
                    launch(queue, sycl::nd_range<1>(1, 1), do_some_work<64>{}, supported).wait();
                }

                if(not *supported)
                {
                    std::cout << "    unsupported sub-group size\n";
                }
            }

            sycl::free(supported, queue);
            sycl::free(subgroups, queue);
        }
        std::cout << '\n';
    }
}
