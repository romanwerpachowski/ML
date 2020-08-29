/* (C) 2020 Roman Werpachowski. */
#include <iostream>
#include <benchmark/benchmark.h>

int main(int argc, char** argv)
{
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    try {
        ::benchmark::RunSpecifiedBenchmarks();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what();
        return -1;
    }
}