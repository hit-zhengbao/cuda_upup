#pragma once

#include <thread>
#include <vector>
#include <queue>
#include <atomic>
#include <functional>
#include <mutex>
#include <future>

class ThreadPool
{
public:
    static ThreadPool& GetTheadPool()
    {
        static ThreadPool thread_pool(std::thread::hardware_concurrency());
        return thread_pool;
    }

    ThreadPool(int32_t num_threads)
    {
        for (int32_t i = 0; i < num_threads; ++i)
        {
            m_threads.emplace_back(std::bind(&ThreadPool::ThreadFunc, this));
        }
    }

    ~ThreadPool()
    {
        m_stop = true;
        m_cond.notify_all();

        for (auto &thread : m_threads)
        {
            if (thread.joinable())
            {
                thread.join();
            }
        }
    }

    template<typename FuncType, typename ...ArgcType>
    auto AddTask(FuncType &&func, ArgcType &&...argc) -> std::future<typename std::result_of<FuncType(ArgcType...)>::type>
    {
        using RetType = typename std::result_of<FuncType(ArgcType...)>::type;

        auto task = std::make_shared<std::packaged_task<RetType()>>(std::bind(std::forward<FuncType>(func), std::forward<ArgcType>(argc)...));

        std::future<RetType> ret = task->get_future();

        if (!m_stop)
        {
            {
                std::unique_lock lock(m_mutex);

                m_tasks.emplace([task]() {(*task)();});
            }

            m_cond.notify_one();
        }

        return ret;
    }

private:
    void ThreadFunc()
    {
        while (1)
        {
            std::function<void()> task;

            {
                std::unique_lock lock(m_mutex);

                m_cond.wait(lock, [this]() { return m_stop || !m_tasks.empty(); });

                if (m_stop && m_tasks.empty())
                {
                    return;
                }

                task = std::move(m_tasks.front());
                m_tasks.pop();
            }

            task();
        }
    }

    std::vector<std::thread>            m_threads;
    std::queue<std::function<void()>>   m_tasks;
    std::atomic_bool                    m_stop;
    std::mutex                          m_mutex;
    std::condition_variable             m_cond;
};