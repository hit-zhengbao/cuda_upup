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

        LOG_INFO("***** New tag: %s *****", m_tag_name.c_str());

    }

    /**
     * @brief count the time between the last timepoint_name and now.
    */
    void AddTimePoint(const std::string &timepoint_name)
    {
        auto cur_time = ClockType::now();

        LOG_INFO("***** Time for sub tag %s cost %ldms", timepoint_name.c_str(),
                 std::chrono::duration_cast<std::chrono::milliseconds>(cur_time - m_timepoint_time).count());

        m_timepoint_time = cur_time;
    }

    /**
     * @brief print the time cost for tag_name from the constructor.
    */
    ~TimeTracker()
    {
        auto cur_time = ClockType::now();
        LOG_INFO("***** Time for tag %s all cost %ldus\n", m_tag_name.c_str(),
                 std::chrono::duration_cast<std::chrono::microseconds>(cur_time - m_start_time).count());
    }

private:
    std::string m_tag_name;
    TimePoint   m_start_time;
    TimePoint   m_timepoint_time;
};
}