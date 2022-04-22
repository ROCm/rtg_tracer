###########################################################################
## Copyright (c) 2022 Advanced Micro Devices, Inc.
###########################################################################

all:
	$(MAKE) -C src

clean:
	$(MAKE) -C src clean

.PHONY: all clean
