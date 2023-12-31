#pragma once

#include "launch.h"
#include "printf.h"

#include <sycl/sycl.hpp>

template<uint32_t S>
struct do_some_work
{
    void operator()(sycl::nd_item<1> item, bool* supported) const
    {
        if constexpr(S == 0)
        {
            printf("      the automatic sub-group size is %d\n", item.get_sub_group().get_max_local_range()[0]);
        }
        else
        {
            printf("      the expected sub-group size is %d\n", S);
            printf("      the actual sub-group size is %d\n", item.get_sub_group().get_max_local_range()[0]);
        }
        *supported = true;
    }
};

template<uint32_t S>
struct traits::RequiredSubGroupSize<do_some_work<S>>
{
    static constexpr uint32_t value = S;
};
