.PHONY: all clean

TARGETS := test test.x86_64 test.acm_g10

all: $(TARGETS)

clean:
	rm -f $(TARGETS)

test: main.cc kernel.h warpsize.h
	icpx -std=c++17 -O2 -g -fsycl -fsycl-targets=spir64_x86_64,intel_gpu_acm_g10 $< -Wno-unknown-cuda-version -o $@

test.x86_64: main.cc kernel.h warpsize.h
	icpx -std=c++17 -O2 -g -fsycl -fsycl-targets=spir64_x86_64 $< -Wno-unknown-cuda-version -o $@

test.acm_g10: main.cc kernel.h warpsize.h
	icpx -std=c++17 -O2 -g -fsycl -fsycl-targets=intel_gpu_acm_g10 $< -Wno-unknown-cuda-version -o $@
