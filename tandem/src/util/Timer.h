// Copyright (c) 2021 Lukas Koestler, Nan Yang. All rights reserved.

#ifndef DR_DSO_TIMER_H
#define DR_DSO_TIMER_H

#include <chrono>
#include <string>
#include <map>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdio.h>

template <class Rep, class Period>
constexpr double cast_to_ms(const std::chrono::duration<Rep,Period>& d)
{
  return std::chrono::duration<double>(d).count() * 1e3;
}

class Timer {
  using hrc = std::chrono::high_resolution_clock;
  using time_point = hrc::time_point;
  using duration = hrc::duration;

  using string = std::string;

  class Instance {
  public:
    time_point t;
    duration dt;

    Instance(time_point const &t_in, duration const &dt_in) : t(t_in), dt(dt_in) {};

    double dt2ms() const{return cast_to_ms(dt);};
    double t2ms(time_point const& start) const{return cast_to_ms(t - start );};
  };

  using map = std::map<std::string, std::vector<Instance>>;

public:
    static time_point start(){return hrc::now();};
    static double end_ms(time_point const& start) {return cast_to_ms(hrc::now() - start); };
public:
  Timer(){
    global_start = hrc::now();
  };

  int start_timing(string const &key) {
    starts[key][counter] = hrc::now();
    counter++;
    return (counter - 1);
  };

  void end_timing(string const& key, int id, bool print=false){
    auto const& start = starts[key][id];
    instances[key].emplace_back(start, hrc::now()-start);
    starts[key].erase(id);
    if (print)
      printf(("DRMVSNET:   "+key+" %6.2f ms\n").c_str(), instances[key].back().dt2ms());
  };

  double mean_timing(string const& key){
    double sum = 0;
    int count = 0;

    for (auto const& inst: instances[key]){
      sum += inst.dt2ms();
      count += 1;
    }

    if (count == 0)
      return 0.0;
    return sum / (double)count;
  };

  void write_to_file(string const& filename){
    std::ofstream myfile;
    myfile.open (filename);
    for (auto const& key_vec : instances){
      for (auto const& inst: key_vec.second){
        myfile << key_vec.first;
        myfile << " "<< std::setprecision (15) << inst.t2ms(global_start);
        myfile << " "<< std::setprecision (15) << inst.dt2ms() << "\n";
      }
    }
    myfile.close();
  }

private:
  int counter = 0;
  std::map<std::string, std::map<int, time_point>> starts;
  std::map<std::string, std::map<int, duration>> cum_times;
  std::map<std::string, std::vector<Instance>> instances;
  time_point global_start;
};


#endif //DR_DSO_TIMER_H
