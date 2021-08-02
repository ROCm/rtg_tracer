/**********************************************************************
Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:

<95>    Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer.
<95>    Redistributions in binary form must reproduce the above copyright notice, this list of
conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#ifndef SRC_UTIL_HSA_RSRC_FACTORY_H_
#define SRC_UTIL_HSA_RSRC_FACTORY_H_

#include <hsa/hsa.h>

#include <vector>

#define CHECK_STATUS(msg, status) do {                                                             \
  if ((status) != HSA_STATUS_SUCCESS) {                                                            \
    const char* emsg = 0;                                                                          \
    hsa_status_string(status, &emsg);                                                              \
    printf("%s: %s\n", msg, emsg ? emsg : "<unknown error>");                                      \
    abort();                                                                                       \
  }                                                                                                \
} while (0)

#define CHECK_ITER_STATUS(msg, status) do {                                                        \
  if ((status) != HSA_STATUS_INFO_BREAK) {                                                         \
    const char* emsg = 0;                                                                          \
    hsa_status_string(status, &emsg);                                                              \
    printf("%s: %s\n", msg, emsg ? emsg : "<unknown error>");                                      \
    abort();                                                                                       \
  }                                                                                                \
} while (0)

namespace util {

// HSA timer class
// Provides current HSA timestampa and system-clock/ns conversion API
class HsaTimer {
 public:
  typedef uint64_t timestamp_t;
  static const timestamp_t TIMESTAMP_MAX = UINT64_MAX;
  typedef long double freq_t;

  enum time_id_t {
    TIME_ID_CLOCK_REALTIME = 0,
    TIME_ID_CLOCK_MONOTONIC = 1,
    TIME_ID_NUMBER
  };

  HsaTimer() {
    timestamp_t sysclock_hz = 0;
    hsa_status_t status = hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &sysclock_hz);
    CHECK_STATUS("hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY)", status);
    sysclock_factor_ = (freq_t)1000000000 / (freq_t)sysclock_hz;
  }

  // Methods for system-clock/ns conversion
  timestamp_t sysclock_to_ns(const timestamp_t& sysclock) const {
    return timestamp_t((freq_t)sysclock * sysclock_factor_);
  }
  timestamp_t ns_to_sysclock(const timestamp_t& time) const {
    return timestamp_t((freq_t)time / sysclock_factor_);
  }

  // Method for timespec/ns conversion
  static timestamp_t timespec_to_ns(const timespec& time) {
    return ((timestamp_t)time.tv_sec * 1000000000) + time.tv_nsec;
  }

  // Return timestamp in 'ns'
  timestamp_t timestamp_ns() const {
    timestamp_t sysclock;
    hsa_status_t status = hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP, &sysclock);
    CHECK_STATUS("hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP)", status);
    return sysclock_to_ns(sysclock);
  }

  // Return time in 'ns'
  static timestamp_t clocktime_ns(clockid_t clock_id) {
    timespec time;
    clock_gettime(clock_id, &time);
    return timespec_to_ns(time);
  }

  // Return pair of correlated values of profiling timestamp and time with
  // correlation error for a given time ID and number of iterations
  void correlated_pair_ns(time_id_t time_id, uint32_t iters,
                          timestamp_t* timestamp_v, timestamp_t* time_v, timestamp_t* error_v) const {
    clockid_t clock_id = 0;
    switch (clock_id) {
      case TIME_ID_CLOCK_REALTIME:
        clock_id = CLOCK_REALTIME;
        break;
      case TIME_ID_CLOCK_MONOTONIC:
        clock_id = CLOCK_MONOTONIC;
        break;
      default:
        CHECK_STATUS("internal error: invalid time_id", HSA_STATUS_ERROR);
    }

    std::vector<timestamp_t> ts_vec(iters);
    std::vector<timespec> tm_vec(iters);
    const uint32_t steps = iters - 1;

    for (uint32_t i = 0; i < iters; ++i) {
      hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP, &ts_vec[i]);
      clock_gettime(clock_id, &tm_vec[i]);
    }

    const timestamp_t ts_base = sysclock_to_ns(ts_vec.front());
    const timestamp_t tm_base = timespec_to_ns(tm_vec.front());
    const timestamp_t error = (ts_vec.back() - ts_vec.front()) / (2 * steps);

    timestamp_t ts_accum = 0;
    timestamp_t tm_accum = 0;
    for (uint32_t i = 0; i < iters; ++i) {
      ts_accum += (ts_vec[i] - ts_base);
      tm_accum += (timespec_to_ns(tm_vec[i]) - tm_base);
    }

    *timestamp_v = (ts_accum / iters) + ts_base + error;
    *time_v = (tm_accum / iters) + tm_base;
    *error_v = error;
  }

 private:
  // Timestamp frequency factor
  freq_t sysclock_factor_;
};

}  // namespace util

#endif  // SRC_UTIL_HSA_RSRC_FACTORY_H_
