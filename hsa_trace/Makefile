rtg_tracer.so: rtg_tracer.o
	$(CXX) -o $@ $< -shared -rdynamic -std=c++11

rtg_tracer.o: rtg_tracer.cpp
	$(CXX) -o $@ -c $< -I/opt/rocm/hsa/include -DAMD_INTERNAL_BUILD -std=c++11 -fPIC

clean:
	rm -f rtg_tracer.o rtg_tracer.so
