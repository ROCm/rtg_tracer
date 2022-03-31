HIP_PATH = /opt/rocm/hip

HIPCC=$(HIP_PATH)/bin/hipcc

CXXFLAGS = -O3
#CXXFLAGS = -g -O0

ROC_LIBS = -lroctx64 -lroctracer64 -lamdhip64 -lhsa-runtime64 -lsqlite3
ROC_INCLUDE = -I/opt/rocm/include -I/opt/rocm/hip/include -I/opt/rocm/roctracer/include -I/opt/rocm/hsa/include/hsa

RTG_MAIN = rtg_tracer.so
RPD_MAIN = rpd_tracer.so

all: $(RTG_MAIN) $(RPD_MAIN)

RTG_SRCS =
RTG_SRCS += rtg_tracer.cpp
RTG_SRCS += rtg_out_printf.cpp
RTG_SRCS += rtg_out_printf_lockless.cpp
RTG_SRCS += missing_ostream_definitions.cpp

RTG_OBJS = $(RTG_SRCS:.cpp=.o)

$(RTG_MAIN): $(RTG_OBJS)
	$(CXX) -o $@ $(RTG_OBJS) -shared -rdynamic -std=c++14 -L/opt/rocm/lib $(ROC_LIBS) $(CXXFLAGS) -Wl,-rpath,/opt/rocm/lib

RPD_SRCS =
RPD_SRCS += rtg_out_printf.cpp
RPD_SRCS += rtg_out_printf_lockless.cpp
RPD_SRCS += rtg_out_rpd.cpp
RPD_SRCS += missing_ostream_definitions.cpp
RPD_SRCS += Table.cpp
RPD_SRCS += OpTable.cpp
RPD_SRCS += ApiTable.cpp
RPD_SRCS += StringTable.cpp
RPD_SRCS += MetadataTable.cpp
RPD_SRCS += ApiIdList.cpp
RPD_SRCS += CopyApiTable.cpp
RPD_SRCS += KernelApiTable.cpp

RPD_OBJS = $(RPD_SRCS:.cpp=.o)
RPD_OBJS += rpd_tracer.o

$(RPD_MAIN): $(RPD_OBJS)
	$(CXX) -o $@ $(RPD_OBJS) -shared -rdynamic -std=c++14 -L/opt/rocm/lib $(ROC_LIBS) $(CXXFLAGS) -Wl,-rpath,/opt/rocm/lib

.cpp.o:
	$(CXX) -o $@ -c $< $(ROC_INCLUDE) -D__HIP_PLATFORM_AMD__ -DAMD_INTERNAL_BUILD -std=c++14 -fPIC $(CXXFLAGS)

rpd_tracer.o: rtg_tracer.cpp
	$(CXX) -o $@ -c $< $(ROC_INCLUDE) -D__HIP_PLATFORM_AMD__ -DAMD_INTERNAL_BUILD -std=c++14 -fPIC $(CXXFLAGS) -DRPD_TRACER


rtg_out_printf.o: rtg_out_printf.h rtg_out.h
rtg_out_printf_lockless.o: rtg_out_printf_lockless.h rtg_out.h
rtg_out_rpd.o: rtg_out_rpd.h rtg_out.h
missing_ostream_definitions.o: missing_ostream_definitions.h
Table.o: hsa_rsrc_factory.h Table.h
OpTable.o: hsa_rsrc_factory.h Table.h
ApiTable.o: hsa_rsrc_factory.h Table.h
StringTable.o: hsa_rsrc_factory.h Table.h
MetadataTable.o: hsa_rsrc_factory.h Table.h
KernelApiTable.o: hsa_rsrc_factory.h Table.h
CopyApiTable.o: hsa_rsrc_factory.h Table.h
ApiIdList.o: ApiIdList.h

clean:
	rm -f *.o *.so
