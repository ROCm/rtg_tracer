###########################################################################
## Copyright (c) 2022 Advanced Micro Devices, Inc.
###########################################################################

dlopen_test: dlopen_test.c
	$(CC) -o $@ $< -ldl

nccl-example: nccl-example.cpp
	mpicxx -o $@ $< -L/opt/rocm/lib -lamdhip64 -lrccl -I/opt/rocm/include -D__HIP_PLATFORM_AMD__

buffer_overflow: buffer_overflow.cpp
	hipcc -o $@ $< -L/opt/rocm/lib -lamdhip64 -I/opt/rocm/include -D__HIP_PLATFORM_AMD__

clean:
	rm -f nccl-example dlopen_test
