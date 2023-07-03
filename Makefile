.PHONY: all clean

TARGETS := print print.x86_64 print.acm_g10 print.gen

all: $(TARGETS)

clean:
	rm -f $(TARGETS)

print: print.cc kernel.h warpsize.h
	icpx -std=c++17 -O2 -g -fsycl -fsycl-targets=spir64_x86_64,intel_gpu_acm_g10 $< -Wno-unknown-cuda-version -o $@

print.x86_64: print.cc kernel.h warpsize.h
	icpx -std=c++17 -O2 -g -fsycl -fsycl-targets=spir64_x86_64 $< -Wno-unknown-cuda-version -o $@

print.acm_g10: print.cc kernel.h warpsize.h
	icpx -std=c++17 -O2 -g -fsycl -fsycl-targets=intel_gpu_acm_g10 $< -Wno-unknown-cuda-version -o $@
