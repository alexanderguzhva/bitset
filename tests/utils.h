#pragma once

#include <chrono>

struct StopWatch {
    using time_type = std::chrono::time_point<std::chrono::high_resolution_clock>;
    time_type start;

    StopWatch() {
        start = now();
    }

    inline double elapsed() {
        auto current = now();
        return std::chrono::duration<double>(current - start).count();
    }

    static inline time_type now() {
        return std::chrono::high_resolution_clock::now();
    }
};