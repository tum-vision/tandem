/**
 *  This file is part of dvo.
 *
 *  Copyright 2012 Christian Kerl <christian.kerl@in.tum.de> (Technical University of Munich)
 *  For more information see <http://vision.in.tum.de/data/software/dvo>.
 *
 *  dvo is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  dvo is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with dvo.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef STOPWATCH_H_
#define STOPWATCH_H_

#include <iostream>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>

#include <opencv2/opencv.hpp>

namespace dvo
{
namespace util
{

struct stopwatch
{
private:
  std::string name;
  int64_t begin;
  int count;
  int interval;
  boost::accumulators::accumulator_set<double, boost::accumulators::stats<boost::accumulators::tag::mean> > acc;

public:
  stopwatch(std::string name, int interval = 500) : name(name + ": "), count(0), interval(interval)
  {
  }

  inline void start()
  {
    begin = cv::getTickCount();
  }

  inline void stop()
  {
    int64_t duration = cv::getTickCount() - begin;

    acc(double(duration) / cv::getTickFrequency());

    count++;
  }

  inline void print()
  {
    if(count == interval)
    {
      double m = boost::accumulators::mean(acc);
      std::cerr  << name << m << std::endl;

      // reset
      acc = boost::accumulators::accumulator_set<double, boost::accumulators::stats<boost::accumulators::tag::mean> >();
      count = 0;
    }
  }

  inline void stopAndPrint()
  {
    stop();
    print();
  }
};

struct stopwatch_collection
{
public:
  stopwatch_collection(const size_t num, std::string base_name, int interval = 500) : num(num)
  {
    std::stringstream name;

    for(size_t idx = 0; idx < num; ++idx)
    {
      name.str("");
      name.clear();

      name << base_name << idx;
      watches.push_back(new stopwatch(name.str(), interval));
    }
  }

  ~stopwatch_collection()
  {
    for(size_t idx = 0; idx < num; ++idx)
      delete watches[idx];
  }

  stopwatch& operator[](int idx)
  {
    return *watches[idx];
  }
private:
  size_t num;
  std::vector<stopwatch*> watches;
};

} /* namespace util */
} /* namespace dvo */
#endif /* STOPWATCH_H_ */
