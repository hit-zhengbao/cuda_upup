#pragma once

#include "log.h"

#include <chrono>
#include <iostream>

namespace cudaup
{
/**
 * @brief The class of counting time.
*/
class TimeTracker
{
public:
    using ClockType = std::chrono::steady_clock;
    using TimePoint = ClockType::time_point;

    TimeTracker(const std::string &tag_name): m_tag_name(tag_name)
    {
        m_start_time = ClockType::now();
        m_timepoint_time = m_start_time;

        // LOG_DEBUG("***** New tag: %s *****", m_tag_name.c_str());

    }

    /**
     * @brief count the time between the last timepoint_name and now.
     * 
     * @return int64_t return the time cost in microseconds.
    */
    int64_t AddTimePoint(const std::string &timepoint_name)
    {
        auto cur_time = ClockType::now();

        auto time_cost = std::chrono::duration_cast<std::chrono::microseconds>(cur_time - m_timepoint_time).count();

        // LOG_DEBUG("***** Time for sub tag %s cost %ldus", timepoint_name.c_str(), time_cost);

        m_timepoint_time = cur_time;

        return time_cost;
    }

    /**
     * @brief print the time cost for tag_name from the constructor.
    */
    ~TimeTracker()
    {
        auto cur_time = ClockType::now();
        // LOG_DEBUG("***** Time for tag %s all cost %ldus\n", m_tag_name.c_str(),
        //          std::chrono::duration_cast<std::chrono::microseconds>(cur_time - m_start_time).count());
    }

private:
    std::string m_tag_name;
    TimePoint   m_start_time;
    TimePoint   m_timepoint_time;
};

/**
 * @brief Analyze the time cost of a function.
 * 
 * @param iteration_num the number of iteration.
 * @param skip_num the number of iteration to skip.
 * @param func the function to be analyzed.
 * @param args the arguments of the function.
*/
template<typename FuncType, typename ...Args>
int32_t AnalyzeFuncTime(const int32_t iteration_num, const int32_t skip_num, const FuncType &func, Args&&...args)
{
    if (iteration_num <= 0 || skip_num < 0 || skip_num > iteration_num)
    {
        LOG_ERROR("AnalyzeFuncTime: invalid input: iter(%d), skip(%d)\n!", iteration_num, skip_num);
        return RET_ERR;
    }

    int64_t total_time = 0;

    for (int32_t i = 0; i < iteration_num; i++)
    {
        TimeTracker time_tracker("time_each_iter");

        int32_t ret = func(std::forward<Args>(args)...);

        total_time += (i >= skip_num) ? time_tracker.AddTimePoint("time_each_iter_another") : 0;

        if (ret != RET_OK)
        {
            LOG_ERROR("AnalyzeFuncTime: func return error!\n");
            return ret;
        }
    }

    LOG_INFO("*****Average Time %.3fus", (float)total_time / (iteration_num - skip_num));

    return RET_OK;
}
}