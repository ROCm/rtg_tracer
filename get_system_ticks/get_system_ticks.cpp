#include <hc.hpp>
#include <sys/time.h>
#include <iostream>

suseconds_t get_host_timestamp() {
       struct timeval tv;
       gettimeofday(&tv, NULL);
       return tv.tv_sec * 1e6 + tv.tv_usec;
}

int main(int argc, char **argv)
{
    uint64_t gpu_time = hc::get_system_ticks();
    suseconds_t host_time = get_host_timestamp();
    std::cout << "unix_ts " << host_time << " microseconds" << std::endl;
    std::cout << "gpu_ts " << gpu_time << " nanoseconds" << std::endl;
    std::cout << "offset " << host_time - (gpu_time/1000) << " microseconds" << std::endl;
    return 0;
}
