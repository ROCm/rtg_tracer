HIP_PATH = /opt/rocm/hip

HIPCC=$(HIP_PATH)/bin/hipcc

ROC_LIBS = -lroctx64 -lroctracer64 -lamdhip64 -lhsa-runtime64 -lsqlite3
ROC_INCLUDE = -I/opt/rocm/include -I/opt/rocm/hip/include -I/opt/rocm/roctracer/include -I/opt/rocm/hsa/include/hsa

MAIN = rtg_tracer.so

SRCS =
SRCS += rtg_tracer.cpp
SRCS += rtg_out_printf.cpp
SRCS += rtg_out_rpd.cpp
SRCS += Table.cpp
SRCS += OpTable.cpp
SRCS += ApiTable.cpp
SRCS += StringTable.cpp
SRCS += MetadataTable.cpp
SRCS += ApiIdList.cpp

OBJS = $(SRCS:.cpp=.o)

$(MAIN): $(OBJS)
	$(CXX) -o $@ $(OBJS) -shared -rdynamic -std=c++14 -L/opt/rocm/lib $(ROC_LIBS) -g

.cpp.o:
	$(CXX) -o $@ -c $< $(ROC_INCLUDE) -D__HIP_PLATFORM_AMD__ -DAMD_INTERNAL_BUILD -std=c++14 -fPIC -g -O3

clean:
	rm -f *.o *.so
