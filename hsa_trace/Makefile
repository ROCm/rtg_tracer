RECKLESS=$(HOME)/reckless

rtg_tracer.so: rtg_tracer.o
	$(CXX) -o $@ $< -shared -rdynamic -std=c++11 -L/opt/rocm/lib -lhsa-runtime64 -g -L$(RECKLESS)/build -lreckless -lpthread

rtg_tracer.o: rtg_tracer.cpp
	$(CXX) -o $@ -c $< -I/opt/rocm/hsa/include -DAMD_INTERNAL_BUILD -std=c++11 -fPIC -g -I$(RECKLESS)/reckless/include

clean:
	rm -f rtg_tracer.o rtg_tracer.so
