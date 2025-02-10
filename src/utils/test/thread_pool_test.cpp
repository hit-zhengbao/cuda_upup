#include "thread_pool.h"
#include "log.h"
#include "simple_gtest.h"
#include <atomic>

SIMPLE_TEST(thread_pool_test)
{
    ThreadPool &thread_pool = ThreadPool::GetTheadPool();

    std::atomic<int32_t> count = 0;

    auto ThreadPrint = [&count] () {
        std::ostringstream oss;
        oss << std::this_thread::get_id();

        count.fetch_add(1, std::memory_order_relaxed);

        LOG_INFO("Thread id: %s, count: %d\n", oss.str().c_str(), count.load(std::memory_order_relaxed));

        return (int32_t)0;
    };

    std::vector<std::future<int32_t>> futures;

    for (int32_t i = 0; i < std::thread::hardware_concurrency(); ++i)
    {
        futures.push_back(thread_pool.AddTask(ThreadPrint));
    }

    for (auto &future : futures)
    {
        future.get();
    }

    LOG_INFO("num of threads: %d, final count: %d\n", std::thread::hardware_concurrency(), count.load(std::memory_order_relaxed));
}