all: get_system_ticks

get_system_ticks: get_system_ticks.cpp
	/opt/rocm/bin/hcc `/opt/rocm/bin/hcc-config --cxxflags --ldflags` $< -o $@ -lstdc++

clean:
	rm -f get_system_ticks
