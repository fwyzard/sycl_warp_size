#include <sycl/sycl.hpp>

template <std::size_t S>
void do_some_work(sycl::nd_item<1> item, std::size_t* expected, std::size_t* actual) {
  *expected = S;
  *actual = item.get_sub_group().get_max_local_range()[0];
}
