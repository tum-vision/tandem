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

#ifndef HISTOGRAM_H_
#define HISTOGRAM_H_

#include <opencv2/opencv.hpp>

namespace dvo
{
namespace util
{

/**
 * Computes the number of bins for a histogram with values in the range of [min max] and binWidth values in each bin.
 */
int getNumberOfBins(float min, float max, float binWidth);

/**
 * Computes the one dimensional histogram of the given data. The values have to be in the range [min max].
 *
 * See: cv::calcHist(...)
 */
void compute1DHistogram(const cv::Mat& data, cv::Mat& histogram, float min, float max, float binWidth);

float computeMedianFromHistogram(const cv::Mat& histogram, float min, float max);

float computeEntropyFromHistogram(const cv::Mat& histogram);

int countElementsInHistogram(const cv::Mat& histogram);

} /* namespace util */
} /* namespace dvo */
#endif /* HISTOGRAM_H_ */
