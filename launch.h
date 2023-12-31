#pragma once

#include "sycl_subgroup_size.h"

#include <sycl/sycl.hpp>

namespace traits
{

    template<typename T>
    struct RequiredSubGroupSize
    {
        static constexpr uint32_t value = 0;
    };

    template<typename T>
    constexpr uint32_t required_sub_group_size = RequiredSubGroupSize<T>::value;

} // namespace traits

template<int D, typename F, typename... Args>
sycl::event launch(sycl::queue queue, sycl::nd_range<D> range, F&& f, Args&&... args)
{
    constexpr uint32_t sub_group_size = traits::required_sub_group_size<F>;

    if constexpr(sub_group_size == 0)
    {
        // no explicit subgroup size requirement
        return queue.submit(
            [&](sycl::handler& cgh)
            {
                cgh.parallel_for(
                    range,
                    [f, args...](sycl::nd_item<D> item)
                    {
                        // call the user kernel function with the SYCL item and the user arguments
                        f(item, args...);
                    });
            });
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
                    [f, args...](sycl::nd_item<D> item) [[intel::reqd_sub_group_size(sub_group_size)]]
                    {
                        // call the user kernel function with the SYCL item and the user arguments
                        f(item, args...);
                    });
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
                        [f, args...](sycl::nd_item<D> item) [[intel::reqd_sub_group_size(sub_group_size)]]
                        {
                            // call the user kernel function with the SYCL item and the user arguments
                            f(item, args...);
                        });
                });
#    else
            // this subgroup size is not supported, raise an exception
            throw sycl::errc::kernel_not_supported;
            // empty kernel, required to keep SYCL happy
            return queue.submit(
                [&](sycl::handler& cgh)
                {
                    cgh.parallel_for(
                        range,
                        [f, args...](sycl::nd_item<D> item)
                        {
                            // [[maybe_unused]] is not allowed in a lambda capture
                            (void) f;
                            (void) (args, ...);
                        });
                });
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
                        [f, args...](sycl::nd_item<D> item) [[intel::reqd_sub_group_size(sub_group_size)]]
                        {
                            // call the user kernel function with the SYCL item and the user arguments
                            f(item, args...);
                        });
                });
#    else
            // this subgroup size is not supported, raise an exception
            throw sycl::errc::kernel_not_supported;
            // empty kernel, required to keep SYCL happy
            return queue.submit(
                [&](sycl::handler& cgh)
                {
                    cgh.parallel_for(
                        range,
                        [f, args...](sycl::nd_item<D> item)
                        {
                            // [[maybe_unused]] is not allowed in a lambda capture
                            (void) f;
                            (void) (args, ...);
                        });
                });
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
                        [f, args...](sycl::nd_item<D> item) [[intel::reqd_sub_group_size(sub_group_size)]]
                        {
                            // call the user kernel function with the SYCL item and the user arguments
                            f(item, args...);
                        });
                });
#    else
            // this subgroup size is not supported, raise an exception
            throw sycl::errc::kernel_not_supported;
            // empty kernel, required to keep SYCL happy
            return queue.submit(
                [&](sycl::handler& cgh)
                {
                    cgh.parallel_for(
                        range,
                        [f, args...](sycl::nd_item<D> item)
                        {
                            // [[maybe_unused]] is not allowed in a lambda capture
                            (void) f;
                            (void) (args, ...);
                        });
                });
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
                        [f, args...](sycl::nd_item<D> item) [[intel::reqd_sub_group_size(sub_group_size)]]
                        {
                            // call the user kernel function with the SYCL item and the user arguments
                            f(item, args...);
                        });
                });
#    else
            // this subgroup size is not supported, raise an exception
            throw sycl::errc::kernel_not_supported;
            // empty kernel, required to keep SYCL happy
            return queue.submit(
                [&](sycl::handler& cgh)
                {
                    cgh.parallel_for(
                        range,
                        [f, args...](sycl::nd_item<D> item)
                        {
                            // [[maybe_unused]] is not allowed in a lambda capture
                            (void) f;
                            (void) (args, ...);
                        });
                });
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
                        [f, args...](sycl::nd_item<D> item) [[intel::reqd_sub_group_size(sub_group_size)]]
                        {
                            // call the user kernel function with the SYCL item and the user arguments
                            f(item, args...);
                        });
                });
#    else
            // this subgroup size is not supported, raise an exception
            throw sycl::errc::kernel_not_supported;
            // empty kernel, required to keep SYCL happy
            return queue.submit(
                [&](sycl::handler& cgh)
                {
                    cgh.parallel_for(
                        range,
                        [f, args...](sycl::nd_item<D> item)
                        {
                            // [[maybe_unused]] is not allowed in a lambda capture
                            (void) f;
                            (void) (args, ...);
                        });
                });
#    endif
        }
#endif

        // this subgroup size is not supported, raise an exception
        throw sycl::errc::kernel_not_supported;
    }
}
