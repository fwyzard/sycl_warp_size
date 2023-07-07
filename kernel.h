#include "sycl_subgroup_size.h"

#include <sycl/sycl.hpp>

#include "printf.h"


template<std::size_t S>
struct do_some_work
{
    void operator()(sycl::nd_item<1> item, bool* supported) const
    {
        printf("    the expected sub-group size is %d\n", S);
        printf("    the actual sub-group size is %d\n", item.get_sub_group().get_max_local_range()[0]);
        *supported = true;
    }
};

namespace traits {

template <typename T>
struct RequiredSubGroupSize {
  static constexpr std::size_t value = 0;
};

template <typename T>
constexpr std::size_t required_sub_group_size = RequiredSubGroupSize<T>::value;

}

template<std::size_t S>
struct traits::RequiredSubGroupSize<do_some_work<S>> {
  static constexpr std::size_t value = S;
};



template<int D, typename F, typename... Args>
sycl::event launch_kernel(sycl::queue queue, sycl::nd_range<D> range, F&& f, Args&&... args)
{
    constexpr std::size_t sub_group_size = traits::required_sub_group_size<F>;

    if constexpr(sub_group_size == 0)
    {
        // no explicit subgroup size requirement
        return queue.submit([&](sycl::handler& cgh)
                            { cgh.parallel_for(range, [f, args...](sycl::nd_item<D> item) { f(item, args...); }); });
    }
    else
    {
#if(SYCL_SUBGROUP_SIZE == 0)
        // no explicit SYCL target, assume JIT compilation
        return queue.submit(
            [&](sycl::handler& cgh)
            {
                cgh.parallel_for(
                    range,
                    [f, args...](sycl::nd_item<D> item) [[intel::reqd_sub_group_size(sub_group_size)]] { f(item, args...); });
            });
#else
        // check if the kernel should be launched with a subgroup size of 4
        if constexpr(sub_group_size == 4)
        {
#    if(SYCL_SUBGROUP_SIZE & 4)
            // launch the kernel with a subgroup size of 4
            return queue.submit(
                [&](sycl::handler& cgh)
                {
                    cgh.parallel_for(
                        range,
                        [f, args...](sycl::nd_item<D> item) [[intel::reqd_sub_group_size(sub_group_size)]] { f(item, args...); });
                });
#    else
            // this subgroup size is not support, raise an exception
            throw sycl::errc::kernel_not_supported;
            // empty kernel, required to keep SYCL happy
            return queue.submit([&](sycl::handler& cgh)
                                { cgh.parallel_for(range, [f, args...](sycl::nd_item<D> item) {}); });
#    endif
        }

        // check if the kernel should be launched with a subgroup size of 8
        if constexpr(sub_group_size == 8)
        {
#    if(SYCL_SUBGROUP_SIZE & 8)
            // launch the kernel with a subgroup size of 8
            return queue.submit(
                [&](sycl::handler& cgh)
                {
                    cgh.parallel_for(
                        range,
                        [f, args...](sycl::nd_item<D> item) [[intel::reqd_sub_group_size(sub_group_size)]] { f(item, args...); });
                });
#    else
            // this subgroup size is not support, raise an exception
            throw sycl::errc::kernel_not_supported;
            // empty kernel, required to keep SYCL happy
            return queue.submit([&](sycl::handler& cgh)
                                { cgh.parallel_for(range, [f, args...](sycl::nd_item<D> item) {}); });
#    endif
        }

        // check if the kernel should be launched with a subgroup size of 16
        if constexpr(sub_group_size == 16)
        {
#    if(SYCL_SUBGROUP_SIZE & 16)
            // launch the kernel with a subgroup size of 16
            return queue.submit(
                [&](sycl::handler& cgh)
                {
                    cgh.parallel_for(
                        range,
                        [f, args...](sycl::nd_item<D> item) [[intel::reqd_sub_group_size(sub_group_size)]] { f(item, args...); });
                });
#    else
            // this subgroup size is not support, raise an exception
            throw sycl::errc::kernel_not_supported;
            // empty kernel, required to keep SYCL happy
            return queue.submit([&](sycl::handler& cgh)
                                { cgh.parallel_for(range, [f, args...](sycl::nd_item<D> item) {}); });
#    endif
        }

        // check if the kernel should be launched with a subgroup size of 32
        if constexpr(sub_group_size == 32)
        {
#    if(SYCL_SUBGROUP_SIZE & 32)
            // launch the kernel with a subgroup size of 32
            return queue.submit(
                [&](sycl::handler& cgh)
                {
                    cgh.parallel_for(
                        range,
                        [f, args...](sycl::nd_item<D> item) [[intel::reqd_sub_group_size(sub_group_size)]] { f(item, args...); });
                });
#    else
            // this subgroup size is not support, raise an exception
            throw sycl::errc::kernel_not_supported;
            // empty kernel, required to keep SYCL happy
            return queue.submit([&](sycl::handler& cgh)
                                { cgh.parallel_for(range, [f, args...](sycl::nd_item<D> item) {}); });
#    endif
        }

        // check if the kernel should be launched with a subgroup size of 64
        if constexpr(sub_group_size == 64)
        {
#    if(SYCL_SUBGROUP_SIZE & 64)
            // launch the kernel with a subgroup size of 64
            return queue.submit(
                [&](sycl::handler& cgh)
                {
                    cgh.parallel_for(
                        range,
                        [f, args...](sycl::nd_item<D> item) [[intel::reqd_sub_group_size(sub_group_size)]] { f(item, args...); });
                });
#    else
            // this subgroup size is not support, raise an exception
            throw sycl::errc::kernel_not_supported;
            // empty kernel, required to keep SYCL happy
            return queue.submit([&](sycl::handler& cgh)
                                { cgh.parallel_for(range, [f, args...](sycl::nd_item<D> item) {}); });
#    endif
        }
#endif

        // this subgroup size is not support, raise an exception
        throw sycl::errc::kernel_not_supported;
    }
}
