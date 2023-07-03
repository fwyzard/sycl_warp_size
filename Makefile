.PHONY: all clean

TARGETS := test test.cpu test.gpu test.nv

all: $(TARGETS)

clean:
	rm -f $(TARGETS)

test: main.cc kernel.h sycl_subgroup_size.h
	icpx -std=c++17 -O2 -g -fsycl -fsycl-targets=spir64_x86_64,intel_gpu_tgllp,intel_gpu_acm_g10,intel_gpu_pvc,nvidia_gpu_sm_75,nvidia_gpu_sm_86 -Wno-unknown-cuda-version $< -o $@

test.cpu: main.cc kernel.h sycl_subgroup_size.h
	icpx -std=c++17 -O2 -g -fsycl -fsycl-targets=spir64_x86_64 $< -o $@

test.gpu: main.cc kernel.h sycl_subgroup_size.h
	icpx -std=c++17 -O2 -g -fsycl -fsycl-targets=intel_gpu_tgllp,intel_gpu_acm_g10,intel_gpu_pvc $< -o $@

test.nv: main.cc kernel.h sycl_subgroup_size.h
	icpx -std=c++17 -O2 -g -fsycl -fsycl-targets=nvidia_gpu_sm_75,nvidia_gpu_sm_86 -Wno-unknown-cuda-version $< -o $@
