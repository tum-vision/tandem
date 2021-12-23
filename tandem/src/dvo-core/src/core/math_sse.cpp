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

#include <dvo/core/math_sse.h>

#ifdef __CDT_PARSER__
  #define __SSE3__
#endif

#include <immintrin.h>
#include <pmmintrin.h>

#include <iostream>

namespace dvo
{
namespace core
{

static inline void dump(const char* prefix, __m128 v)
{
  EIGEN_ALIGN16 float data[4];

  _mm_store_ps(data, v);

  std::cerr << prefix << " " << data[0] << " " << data[1] << " " << data[2] << " " << data[3] << std::endl;
}

OptimizedSelfAdjointMatrix6x6f::OptimizedSelfAdjointMatrix6x6f()
{
}

void OptimizedSelfAdjointMatrix6x6f::setZero()
{
  for(size_t idx = 0; idx < Size; idx++)
    data[idx] = 0.0f;
}

void OptimizedSelfAdjointMatrix6x6f::rankUpdate(const Eigen::Matrix<float, 6, 1>& u, const float& alpha)
{
  __m128 s = _mm_set1_ps(alpha);
  __m128 v1234 = _mm_loadu_ps(u.data());
  __m128 v56xx = _mm_loadu_ps(u.data() + 4);

  __m128 v1212 = _mm_movelh_ps(v1234, v1234);
  __m128 v3434 = _mm_movehl_ps(v1234, v1234);
  __m128 v5656 = _mm_movelh_ps(v56xx, v56xx);

  __m128 v1122 = _mm_mul_ps(s, _mm_unpacklo_ps(v1212, v1212));

  _mm_store_ps(data + 0, _mm_add_ps(_mm_load_ps(data + 0), _mm_mul_ps(v1122, v1212)));
  _mm_store_ps(data + 4, _mm_add_ps(_mm_load_ps(data + 4), _mm_mul_ps(v1122, v3434)));
  _mm_store_ps(data + 8, _mm_add_ps(_mm_load_ps(data + 8), _mm_mul_ps(v1122, v5656)));

  __m128 v3344 = _mm_mul_ps(s, _mm_unpacklo_ps(v3434, v3434));

  _mm_store_ps(data + 12, _mm_add_ps(_mm_load_ps(data + 12), _mm_mul_ps(v3344, v3434)));
  _mm_store_ps(data + 16, _mm_add_ps(_mm_load_ps(data + 16), _mm_mul_ps(v3344, v5656)));

  __m128 v5566 = _mm_mul_ps(s, _mm_unpacklo_ps(v5656, v5656));

  _mm_store_ps(data + 20, _mm_add_ps(_mm_load_ps(data + 20), _mm_mul_ps(v5566, v5656)));
}

void OptimizedSelfAdjointMatrix6x6f::rankUpdate(const Eigen::Matrix<float, 2, 6>& v, const Eigen::Matrix2f& alpha)
{
  /**
   * layout of alpha:
   *
   *   1 2
   *   3 4
   */
  __m128 alpha1324 = _mm_load_ps(alpha.data());           // load first two columns from column major data
  __m128 alpha1313 = _mm_movelh_ps(alpha1324, alpha1324); // first column 2x
  __m128 alpha2424 = _mm_movehl_ps(alpha1324, alpha1324); // second column 2x

  /**
   * layout of v:
   *
   *   1a 2a 3a 4a 5a 6a
   *   1b 2b 3b 4b 5b 6b
   */

  /**
   * layout of u = v * alpha:
   *
   *   1a 2a 3a 4a 5a 6a
   *   1b 2b 3b 4b 5b 6b
   */
  __m128 v1a1b2a2b = _mm_load_ps(v.data() + 0); // load first and second column

  __m128 u1a2a1b2b = _mm_hadd_ps(
      _mm_mul_ps(v1a1b2a2b, alpha1313),
      _mm_mul_ps(v1a1b2a2b, alpha2424)
  );

  __m128 u1a1b1a1b = _mm_shuffle_ps(u1a2a1b2b, u1a2a1b2b, _MM_SHUFFLE(2, 0, 2, 0));
  __m128 u2a2b2a2b = _mm_shuffle_ps(u1a2a1b2b, u1a2a1b2b, _MM_SHUFFLE(3, 1, 3, 1));

  // upper left 2x2 block of A matrix in row major format
  __m128 b11 = _mm_hadd_ps(
      _mm_mul_ps(u1a1b1a1b, v1a1b2a2b),
      _mm_mul_ps(u2a2b2a2b, v1a1b2a2b)
  );
  _mm_store_ps(data + 0, _mm_add_ps(_mm_load_ps(data + 0), b11));

  __m128 v3a3b4a4b = _mm_load_ps(v.data() + 4); // load third and fourth column

  // upper center 2x2 block of A matrix in row major format
  __m128 b12 = _mm_hadd_ps(
      _mm_mul_ps(u1a1b1a1b, v3a3b4a4b),
      _mm_mul_ps(u2a2b2a2b, v3a3b4a4b)
  );
  _mm_store_ps(data + 4, _mm_add_ps(_mm_load_ps(data + 4), b12));

  __m128 v5a5b6a6b = _mm_load_ps(v.data() + 8); // load fifth and sixth column

  // upper right 2x2 block of A matrix in row major format
  __m128 b13 = _mm_hadd_ps(
      _mm_mul_ps(u1a1b1a1b, v5a5b6a6b),
      _mm_mul_ps(u2a2b2a2b, v5a5b6a6b)
  );
  _mm_store_ps(data + 8, _mm_add_ps(_mm_load_ps(data + 8), b13));

  __m128 u3a4a3b4b = _mm_hadd_ps(
      _mm_mul_ps(v3a3b4a4b, alpha1313),
      _mm_mul_ps(v3a3b4a4b, alpha2424)
  );

  __m128 u3a3b3a3b = _mm_shuffle_ps(u3a4a3b4b, u3a4a3b4b, _MM_SHUFFLE(2, 0, 2, 0));
  __m128 u4a4b4a4b = _mm_shuffle_ps(u3a4a3b4b, u3a4a3b4b, _MM_SHUFFLE(3, 1, 3, 1));

  // center center 2x2 block of A matrix in row major format
  __m128 b22 = _mm_hadd_ps(
      _mm_mul_ps(u3a3b3a3b, v3a3b4a4b),
      _mm_mul_ps(u4a4b4a4b, v3a3b4a4b)
  );
  _mm_store_ps(data + 12, _mm_add_ps(_mm_load_ps(data + 12), b22));

  // center right 2x2 block of A matrix in row major format
  __m128 b23 = _mm_hadd_ps(
      _mm_mul_ps(u3a3b3a3b, v5a5b6a6b),
      _mm_mul_ps(u4a4b4a4b, v5a5b6a6b)
  );
  _mm_store_ps(data + 16, _mm_add_ps(_mm_load_ps(data + 16), b23));

  __m128 u5a6a5b6b = _mm_hadd_ps(
      _mm_mul_ps(v5a5b6a6b, alpha1313),
      _mm_mul_ps(v5a5b6a6b, alpha2424)
  );

  __m128 u5a5b5a5b = _mm_shuffle_ps(u5a6a5b6b, u5a6a5b6b, _MM_SHUFFLE(2, 0, 2, 0));
  __m128 u6a6b6a6b = _mm_shuffle_ps(u5a6a5b6b, u5a6a5b6b, _MM_SHUFFLE(3, 1, 3, 1));

  // bottom right 2x2 block of A matrix in row major format
  __m128 b33 = _mm_hadd_ps(
      _mm_mul_ps(u5a5b5a5b, v5a5b6a6b),
      _mm_mul_ps(u6a6b6a6b, v5a5b6a6b)
  );
  _mm_store_ps(data + 20, _mm_add_ps(_mm_load_ps(data + 20), b33));
}

void OptimizedSelfAdjointMatrix6x6f::operator +=(const OptimizedSelfAdjointMatrix6x6f& other)
{
  _mm_store_ps(data +  0, _mm_add_ps(_mm_load_ps(data +  0), _mm_load_ps(other.data +  0)));
  _mm_store_ps(data +  4, _mm_add_ps(_mm_load_ps(data +  4), _mm_load_ps(other.data +  4)));
  _mm_store_ps(data +  8, _mm_add_ps(_mm_load_ps(data +  8), _mm_load_ps(other.data +  8)));
  _mm_store_ps(data + 12, _mm_add_ps(_mm_load_ps(data + 12), _mm_load_ps(other.data + 12)));
  _mm_store_ps(data + 16, _mm_add_ps(_mm_load_ps(data + 16), _mm_load_ps(other.data + 16)));
  _mm_store_ps(data + 20, _mm_add_ps(_mm_load_ps(data + 20), _mm_load_ps(other.data + 20)));
}

void OptimizedSelfAdjointMatrix6x6f::toEigen(Eigen::Matrix<float, 6, 6>& m) const
{
  Eigen::Matrix<float, 6, 6> tmp;
  size_t idx = 0;

  for(size_t i = 0; i < 6; i += 2)
  {
    for(size_t j = i; j < 6; j += 2)
    {
      tmp(i  , j  ) = data[idx++];
      tmp(i  , j+1) = data[idx++];
      tmp(i+1, j  ) = data[idx++];
      tmp(i+1, j+1) = data[idx++];
    }
  }

  tmp.selfadjointView<Eigen::Upper>().evalTo(m);
}

template<>
void MathSse<Sse::Enabled, float>::addOuterProduct(Eigen::Matrix<float, 6, 6>& mat, const Eigen::Matrix<float, 6, 1>& vec, const float& scale)
{
  /**
   * idea:
   *
   * vec = [j0 j1 j2 j3 j4 j5]
   *
   * vec * vec.transpose() =
   * [ j0*j0 j0*j1 j0*j2 j0*j3 j0*j4 j0*j5
   *   j1*j0 j1*j1 j1*j2 j1*j3 j1*j4 j1*j5
   *   j2*j0 j2*j1 j2*j2 j2*j3 j2*j4 j2*j5
   *   j3*j0 j3*j1 j3*j2 j3*j3 j3*j4 j3*j5
   *   j4*j0 j4*j1 j4*j2 j4*j3 j4*j4 j4*j5
   *   j5*j0 j5*j1 j5*j2 j5*j3 j5*j4 j5*j5 ]
   *
   * row multiplicators:
   *
   * j0 * [j0 j1 j2 j3 j4 j5]
   * j1 * [j0 j1 j2 j3 j4 j5]
   * j2 * [j0 j1 j2 j3 j4 j5]
   * j3 * [j0 j1 j2 j3 j4 j5]
   * j4 * [j0 j1 j2 j3 j4 j5]
   * j5 * [j0 j1 j2 j3 j4 j5]
   *
   * substitute:
   *
   * v1 = [j0 j1 j2 j3], v2 = [j4 j5 j0 j1], v3 = [j2 j3 j4 j5]
   *
   * calculate:
   *
   * [j0 j0 j0 j0] * v1, [j0 j0 j1 j1] * v2, [j1 j1 j1 j1] * v3
   * [j2 j2 j2 j2] * v1, [j2 j2 j3 j3] * v2, [j3 j3 j3 j3] * v3
   * [j4 j4 j4 j4] * v1, [j4 j4 j5 j5] * v2, [j5 j5 j5 j5] * v3
   *
   * the vectors for multiplication can be obtained by shuffling v1 and v2 (they are prescaled with scale to v1s and v2s)
   *
   * after each multiplication add result to matrix
   */
  __m128 v1, v2, v3, fac, s, v1s, v2s;

  s = _mm_set1_ps(scale);
  float* target_ptr = mat.data();

  v1 = _mm_load_ps(vec.data() + 0); // [v0, v1, v2, v3]
  v2 = _mm_load_ps(vec.data() + 4); // [v4, v5,  ?,  ?]

  v2 = _mm_shuffle_ps(v2, v1, _MM_SHUFFLE(1, 0, 1, 0)); // [v4, v5, v0, v1]
  v3 = _mm_shuffle_ps(v1, v2, _MM_SHUFFLE(1, 0, 3, 2)); // [v2, v3, v4, v5]

  v1s = _mm_mul_ps(s, v1); // [s * v0, s * v1, s * v2, s * v3]

  // TODO: can we save some shuffles?
  fac = _mm_shuffle_ps(v1s, v1s, _MM_SHUFFLE(0, 0, 0, 0));
  _mm_store_ps(target_ptr, _mm_add_ps(_mm_load_ps(target_ptr), _mm_mul_ps(fac, v1))); target_ptr += 4;

  fac = _mm_shuffle_ps(v1s, v1s, _MM_SHUFFLE(1, 1, 0, 0));
  _mm_store_ps(target_ptr, _mm_add_ps(_mm_load_ps(target_ptr), _mm_mul_ps(fac, v2))); target_ptr += 4;

  fac = _mm_shuffle_ps(v1s, v1s, _MM_SHUFFLE(1, 1, 1, 1));
  _mm_store_ps(target_ptr, _mm_add_ps(_mm_load_ps(target_ptr), _mm_mul_ps(fac, v3))); target_ptr += 4;

  fac = _mm_shuffle_ps(v1s, v1s, _MM_SHUFFLE(2, 2, 2, 2));
  _mm_store_ps(target_ptr, _mm_add_ps(_mm_load_ps(target_ptr), _mm_mul_ps(fac, v1))); target_ptr += 4;

  fac = _mm_shuffle_ps(v1s, v1s, _MM_SHUFFLE(3, 3, 2, 2));
  _mm_store_ps(target_ptr, _mm_add_ps(_mm_load_ps(target_ptr), _mm_mul_ps(fac, v2))); target_ptr += 4;

  fac = _mm_shuffle_ps(v1s, v1s, _MM_SHUFFLE(3, 3, 3, 3));
  _mm_store_ps(target_ptr, _mm_add_ps(_mm_load_ps(target_ptr), _mm_mul_ps(fac, v3))); target_ptr += 4;

  v2s = _mm_mul_ps(s, v2);

  fac = _mm_shuffle_ps(v2s, v2s, _MM_SHUFFLE(0, 0, 0, 0));
  _mm_store_ps(target_ptr, _mm_add_ps(_mm_load_ps(target_ptr), _mm_mul_ps(fac, v1))); target_ptr += 4;

  fac = _mm_shuffle_ps(v2s, v2s, _MM_SHUFFLE(1, 1, 0, 0));
  _mm_store_ps(target_ptr, _mm_add_ps(_mm_load_ps(target_ptr), _mm_mul_ps(fac, v2))); target_ptr += 4;

  fac = _mm_shuffle_ps(v2s, v2s, _MM_SHUFFLE(1, 1, 1, 1));
  _mm_store_ps(target_ptr, _mm_add_ps(_mm_load_ps(target_ptr), _mm_mul_ps(fac, v3))); target_ptr += 4;
}

template<>
void MathSse<Sse::Enabled, double>::addOuterProduct( Eigen::Matrix<double, 6, 6>& mat, const Eigen::Matrix<double, 6, 1>& vec, const double& scale)
{
  __m128d v1, v2, v3; // divide vec in 3 blocks
  __m128d fac, target, s;

  s = _mm_set1_pd(scale);

  double* target_ptr = mat.data();

  v1 = _mm_load_pd(vec.data() + 0);
  v2 = _mm_load_pd(vec.data() + 2);
  v3 = _mm_load_pd(vec.data() + 4);

  // load 0
  fac = _mm_mul_pd(s, _mm_shuffle_pd(v1, v1, _MM_SHUFFLE2(0, 0)));

  _mm_store_pd(target_ptr, _mm_add_pd(_mm_load_pd(target_ptr), _mm_mul_pd(fac, v1))); target_ptr += 2;
  _mm_store_pd(target_ptr, _mm_add_pd(_mm_load_pd(target_ptr), _mm_mul_pd(fac, v2))); target_ptr += 2;
  _mm_store_pd(target_ptr, _mm_add_pd(_mm_load_pd(target_ptr), _mm_mul_pd(fac, v3))); target_ptr += 2;

  // load 1
  fac = _mm_mul_pd(s, _mm_shuffle_pd(v1, v1, _MM_SHUFFLE2(1, 1)));

  _mm_store_pd(target_ptr, _mm_add_pd(_mm_load_pd(target_ptr), _mm_mul_pd(fac, v1))); target_ptr += 2;
  _mm_store_pd(target_ptr, _mm_add_pd(_mm_load_pd(target_ptr), _mm_mul_pd(fac, v2))); target_ptr += 2;
  _mm_store_pd(target_ptr, _mm_add_pd(_mm_load_pd(target_ptr), _mm_mul_pd(fac, v3))); target_ptr += 2;

  // load 2
  fac = _mm_mul_pd(s, _mm_shuffle_pd(v2, v2, _MM_SHUFFLE2(0, 0)));

  _mm_store_pd(target_ptr, _mm_add_pd(_mm_load_pd(target_ptr), _mm_mul_pd(fac, v1))); target_ptr += 2;
  _mm_store_pd(target_ptr, _mm_add_pd(_mm_load_pd(target_ptr), _mm_mul_pd(fac, v2))); target_ptr += 2;
  _mm_store_pd(target_ptr, _mm_add_pd(_mm_load_pd(target_ptr), _mm_mul_pd(fac, v3))); target_ptr += 2;

  // load 3
  fac = _mm_mul_pd(s, _mm_shuffle_pd(v2, v2, _MM_SHUFFLE2(1, 1)));

  _mm_store_pd(target_ptr, _mm_add_pd(_mm_load_pd(target_ptr), _mm_mul_pd(fac, v1))); target_ptr += 2;
  _mm_store_pd(target_ptr, _mm_add_pd(_mm_load_pd(target_ptr), _mm_mul_pd(fac, v2))); target_ptr += 2;
  _mm_store_pd(target_ptr, _mm_add_pd(_mm_load_pd(target_ptr), _mm_mul_pd(fac, v3))); target_ptr += 2;

  // load 4
  fac = _mm_mul_pd(s, _mm_shuffle_pd(v3, v3, _MM_SHUFFLE2(0, 0)));

  _mm_store_pd(target_ptr, _mm_add_pd(_mm_load_pd(target_ptr), _mm_mul_pd(fac, v1))); target_ptr += 2;
  _mm_store_pd(target_ptr, _mm_add_pd(_mm_load_pd(target_ptr), _mm_mul_pd(fac, v2))); target_ptr += 2;
  _mm_store_pd(target_ptr, _mm_add_pd(_mm_load_pd(target_ptr), _mm_mul_pd(fac, v3))); target_ptr += 2;

  // load 5
  fac = _mm_mul_pd(s, _mm_shuffle_pd(v3, v3, _MM_SHUFFLE2(1, 1)));

  _mm_store_pd(target_ptr, _mm_add_pd(_mm_load_pd(target_ptr), _mm_mul_pd(fac, v1))); target_ptr += 2;
  _mm_store_pd(target_ptr, _mm_add_pd(_mm_load_pd(target_ptr), _mm_mul_pd(fac, v2))); target_ptr += 2;
  _mm_store_pd(target_ptr, _mm_add_pd(_mm_load_pd(target_ptr), _mm_mul_pd(fac, v3))); target_ptr += 2;

}

template<>
void MathSse<Sse::Enabled, float>::add(Eigen::Matrix<float, 6, 1>& vec, const Eigen::Matrix<float, 6, 1>& other, const float& scale)
{
  __m128 s = _mm_set1_ps(scale);
  __m128 v1 = _mm_load_ps(other.data());
  __m128 v2 = _mm_loadl_pi(s, (__m64*)(other.data() + 4));

  _mm_store_ps(
      vec.data(),
      _mm_add_ps(
          _mm_load_ps(vec.data()),
          _mm_mul_ps(v1, s)
      )
  );

  _mm_storel_pi(
      (__m64*)(vec.data() + 4),
      _mm_add_ps(
          _mm_loadl_pi(s, (__m64*)(vec.data() + 4)),
          _mm_mul_ps(v2, s)
      )
  );
}

template<>
void MathSse<Sse::Enabled, double>::add(Eigen::Matrix<double, 6, 1>& vec, const Eigen::Matrix<double, 6, 1>& other, const double& scale)
{
  __m128d s = _mm_set1_pd(scale);
  double* target_ptr = vec.data();

  _mm_store_pd(target_ptr, _mm_add_pd(_mm_load_pd(target_ptr), _mm_mul_pd(s, _mm_load_pd(other.data() + 0)))); target_ptr += 2;
  _mm_store_pd(target_ptr, _mm_add_pd(_mm_load_pd(target_ptr), _mm_mul_pd(s, _mm_load_pd(other.data() + 2)))); target_ptr += 2;
  _mm_store_pd(target_ptr, _mm_add_pd(_mm_load_pd(target_ptr), _mm_mul_pd(s, _mm_load_pd(other.data() + 4)))); target_ptr += 2;
}

} /* namespace core */
} /* namespace dvo */
