###########################################################################
## Copyright (c) 2022 Advanced Micro Devices, Inc.
###########################################################################

all:
	$(MAKE) -C rtg_tracer

clean:
	$(MAKE) -C rtg_tracer clean

.PHONY: all clean
