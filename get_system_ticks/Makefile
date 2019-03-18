all: get_system_ticks

get_system_ticks: get_system_ticks.cpp
	hcc `hcc-config --cxxflags --ldflags` $< -o $@

clean:
	rm -f get_system_ticks
