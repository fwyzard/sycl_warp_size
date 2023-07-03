.PHONY: all clean

TARGETS := print print.x86 print.tgl print.nv

all: $(TARGETS)

clean:
	rm -f $(TARGETS)

print: print.cc kernel.h warpsize.h
	icpx -std=c++17 -O2 -g -fsycl -fsycl-targets=spir64_x86_64,intel_gpu_tgllp,nvidia_gpu_sm_86 $< -Wno-unknown-cuda-version -o $@

print.x86: print.cc kernel.h warpsize.h
	icpx -std=c++17 -O2 -g -fsycl -fsycl-targets=spir64_x86_64 $< -Wno-unknown-cuda-version -o $@

print.tgl: print.cc kernel.h warpsize.h
	icpx -std=c++17 -O2 -g -fsycl -fsycl-targets=intel_gpu_tgllp $< -Wno-unknown-cuda-version -o $@

print.nv: print.cc kernel.h warpsize.h
	icpx -std=c++17 -O2 -g -fsycl -fsycl-targets=nvidia_gpu_sm_86 $< -Wno-unknown-cuda-version -o $@
