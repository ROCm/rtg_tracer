rtg_tracer.so: rtg_tracer.o
	$(CXX) -o $@ $< -shared -rdynamic -std=c++14 -L/opt/rocm/lib -lamdhip64 -lhsa-runtime64 -g

rtg_tracer.o: rtg_tracer.cpp flags.h ToStringDefinitions.h ctpl_stl.h flags.h missing_ostream_definitions.h
	$(CXX) -o $@ -c $< -I/opt/rocm/hip/include -I/opt/rocm/include -I/opt/rocm/include/roctracer -I/opt/rocm/hsa/include -D__HIP_PLATFORM_AMD__ -DAMD_INTERNAL_BUILD -std=c++14 -fPIC -g

clean:
	rm -f rtg_tracer.o rtg_tracer.so
