#pragma once

#include <string>
#include <vector>
#include <functional>
#include <iostream>

class SimpleGtest
{
public:
    // Used to register the test cases
    struct TestCase
    {
        std::string            name;
        std::function<void()> func;
    };

    // Run all test cases
    static void RunAllTestCases()
    {
        int pass_num = 0, failed_num = 0;

        for (const auto& test : GetTests())
        {
            std::cout << "[ RUN      ] " << test.name << std::endl;

            try
            {
                test.func();

                ++pass_num;
                std::cout << "[       OK ] " << test.name << std::endl;
            }
            catch (const std::exception& e)
            {
                ++failed_num;
                std::cout << "[  FAILED  ] " << test.name << ": " << e.what() << std::endl;
            }
            catch (...)
            {
                ++failed_num;
                std::cout << "[  FAILED  ] " << test.name << ": unknown error" << std::endl;
            }
        }

        std::cout << "\n[==========] " << pass_num + failed_num << " tests run.\n";
        std::cout << "[  PASSED  ] " << pass_num << " tests.\n";
        if (failed_num > 0)
        {
            std::cout << "[  FAILED  ] " << failed_num << " tests.\n";
        }
    }

    // Run single test case
    static void RunSingleTestCase(const std::string& case_name)
    {
        for (const auto& test : GetTests())
        {
            if (case_name == test.name)
            {
                std::cout << "[ RUN SINGLE FOUND    ]: " << test.name << std::endl;

                try
                {
                    test.func();

                    std::cout << "[       OK ] " << test.name << std::endl;
                }
                catch (const std::exception& e)
                {
                    std::cout << "[  FAILED  ] " << test.name << ": " << e.what() << std::endl;
                }
                catch (...)
                {
                    std::cout << "[  FAILED  ] " << test.name << ": unknown error" << std::endl;
                }

                return;
            }
        }

        std::cout << "[ RUN SINGLE NOT FOUND]: " << case_name << std::endl;
    }

private:
    // Save all registered test cases
    static std::vector<TestCase>& GetTests()
    {
        static std::vector<TestCase> vec_tests;
        return vec_tests;
    }
};

// Define new test case
#define SIMPLE_TEST(test_name)                                              \
    void test_name();                                                       \
    static bool test_name##_registered = []() {                             \
        SimpleGTest::RegisterTest(#test_name, test_name);                   \
        return true;                                                        \
    }();                                                                    \
    void test_name()
