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

#ifndef ID_GENERATOR_H_
#define ID_GENERATOR_H_

#include <vector>
#include <string>
#include <sstream>

namespace dvo
{
namespace util
{

class IdGenerator
{
public:
  IdGenerator(const std::string prefix) :
    prefix_(prefix),
    var_(0)
  {
  }

  const std::vector<std::string>& all()
  {
    return generated_;
  }

  void next(std::string& id)
  {
    id = next();
  }

  std::string next()
  {
    std::stringstream ss;
    ss << prefix_ << var_;

    var_ += 1;
    generated_.push_back(ss.str());

    return ss.str();
  }

  void reset()
  {
    var_ = 0;
    generated_.clear();
  }
private:
  std::string prefix_;
  std::vector<std::string> generated_;
  int var_;
};

} /* namespace util */
} /* namespace dvo */
#endif /* ID_GENERATOR_H_ */
