all:
	$(MAKE) -C get_system_ticks
	$(MAKE) -C hsa_trace

clean:
	$(MAKE) -C get_system_ticks clean
	$(MAKE) -C hsa_trace clean

.PHONY: all clean
