.PHONY: all clean

TARGETS := test test.jit test.aot test.cpu test.gpu test.nv

all: $(TARGETS)

clean:
	rm -f $(TARGETS)

test: main.cc kernel.h sycl_subgroup_size.h
	icpx -std=c++17 -O2 -g -Wno-unknown-cuda-version -fsycl -fsycl-device-code-split=per_kernel -fsycl-targets=spir64,nvptx64-nvidia-cuda $< -o $@

test.jit: main.cc kernel.h sycl_subgroup_size.h
	icpx -std=c++17 -O2 -g -Wno-unknown-cuda-version -fsycl -fsycl-device-code-split=per_kernel -fsycl-targets=spir64 $< -o $@

test.aot: main.cc kernel.h sycl_subgroup_size.h
	icpx -std=c++17 -O2 -g -Wno-unknown-cuda-version -fsycl -fsycl-targets=spir64_x86_64,intel_gpu_tgllp,intel_gpu_acm_g10,intel_gpu_pvc,nvidia_gpu_sm_75,nvidia_gpu_sm_86 $< -o $@

test.cpu: main.cc kernel.h sycl_subgroup_size.h
	icpx -std=c++17 -O2 -g -Wno-unknown-cuda-version -fsycl -fsycl-targets=spir64_x86_64 $< -o $@

test.gpu: main.cc kernel.h sycl_subgroup_size.h
	icpx -std=c++17 -O2 -g -Wno-unknown-cuda-version -fsycl -fsycl-targets=intel_gpu_tgllp,intel_gpu_acm_g10,intel_gpu_pvc $< -o $@

test.pvc: main.cc kernel.h sycl_subgroup_size.h
	icpx -std=c++17 -O2 -g -Wno-unknown-cuda-version -fsycl -fsycl-targets=intel_gpu_pvc $< -o $@

test.nv: main.cc kernel.h sycl_subgroup_size.h
	icpx -std=c++17 -O2 -g -Wno-unknown-cuda-version -fsycl -fsycl-targets=nvidia_gpu_sm_75,nvidia_gpu_sm_86 $< -o $@
