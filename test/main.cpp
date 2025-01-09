#include "simple_gtest.h"

#include <iostream>

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: " << std::endl;
        std::cout << "Run all test cases: exe all" << std::endl;
        std::cout << "Run one test case: exe matmul_test_case" << std::endl;
    }

    std::string run_case_str = argv[1];

    if ("all" == run_case_str)
    {
        SimpleGtest::RunAllTestCases();
    }
    else
    {
        SimpleGtest::RunSingleTestCase(run_case_str);
    }

    return 0;
}
