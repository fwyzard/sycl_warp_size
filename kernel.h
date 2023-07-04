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


#if(SYCL_SUBGROUP_SIZE & 4)
#    define SYCL_REQUIRED_SUBGROUP_SIZE_4 [[intel::reqd_sub_group_size(4)]]
#else
#    define SYCL_REQUIRED_SUBGROUP_SIZE_4
#endif

#if(SYCL_SUBGROUP_SIZE & 8)
#    define SYCL_REQUIRED_SUBGROUP_SIZE_8 [[intel::reqd_sub_group_size(8)]]
#else
#    define SYCL_REQUIRED_SUBGROUP_SIZE_8
#endif

#if(SYCL_SUBGROUP_SIZE & 16)
#    define SYCL_REQUIRED_SUBGROUP_SIZE_16 [[intel::reqd_sub_group_size(16)]]
#else
#    define SYCL_REQUIRED_SUBGROUP_SIZE_16
#endif

#if(SYCL_SUBGROUP_SIZE & 32)
#    define SYCL_REQUIRED_SUBGROUP_SIZE_32 [[intel::reqd_sub_group_size(32)]]
#else
#    define SYCL_REQUIRED_SUBGROUP_SIZE_32
#endif

#if(SYCL_SUBGROUP_SIZE & 64)
#    define SYCL_REQUIRED_SUBGROUP_SIZE_64 [[intel::reqd_sub_group_size(64)]]
#else
#    define SYCL_REQUIRED_SUBGROUP_SIZE_64
#endif

#define SYCL_REQUIRED_SUBGROUP_SIZE_0


#define LAUNCH_SYCL_KERNEL_IF_SUBGROUP_SIZE_IS(VALUE, SIZE)                                                           \
    if constexpr(VALUE == SIZE)                                                                                       \
    {                                                                                                                 \
        queue                                                                                                         \
            .submit(                                                                                                  \
                [&](sycl::handler& cgh)                                                                               \
                {                                                                                                     \
                    cgh.parallel_for(                                                                                 \
                        sycl::nd_range<1>(blocks * threads, threads),                                                 \
                        [f, args...](sycl::nd_item<1> item)                                                           \
                            SYCL_REQUIRED_SUBGROUP_SIZE_##SIZE { f(item, args...); });                                \
                })                                                                                                    \
            .wait();                                                                                                  \
    }

template<std::size_t S, typename F, typename... Args>
void launch_kernel(sycl::queue queue, F&& f, Args&&... args)
{
    constexpr const auto size = S;
    int threads = 1;
    int blocks = 1;

    LAUNCH_SYCL_KERNEL_IF_SUBGROUP_SIZE_IS(size, 4)
      else
    LAUNCH_SYCL_KERNEL_IF_SUBGROUP_SIZE_IS(size, 8)
      else
    LAUNCH_SYCL_KERNEL_IF_SUBGROUP_SIZE_IS(size, 16)
      else
    LAUNCH_SYCL_KERNEL_IF_SUBGROUP_SIZE_IS(size, 32)
      else
    LAUNCH_SYCL_KERNEL_IF_SUBGROUP_SIZE_IS(size, 64)
      else
    LAUNCH_SYCL_KERNEL_IF_SUBGROUP_SIZE_IS(size, 0)  // does not require any subgroup size
}

#undef LAUNCH_SYCL_KERNEL_IF_SUBGROUP_SIZE_IS
