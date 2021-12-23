// Copyright 2019 Emanuele Palazzolo (emanuele.palazzolo@uni-bonn.de), Cyrill Stachniss, University of Bonn
#pragma once
#include <cuda_runtime.h>
#include <string>
#include <fstream>
#include <iostream>

namespace refusion {

namespace tsdfvh {

/**
 * @brief      Structure representing a 3D vertex
 */
struct Vertex {
  /** The 3D position of the vertex */
  float3 position;

  /** The color of the vertex */
  float3 color;
};

/**
 * @brief      Structure representing a triangle
 */
struct Triangle {
  /** The first vertex of the triangle */
  Vertex v0;

  /** The second vertex of the triangle */
  Vertex v1;

  /** The third vertex of the triangle */
  Vertex v2;
};

/**
 * @brief      Class representing a mesh.
 */
class Mesh {
 public:
  /**
   * @brief      Initializes the class.
   *
   * @param[in]  max_triangles  The maximum number of triangles in the mesh
   */
  void Init(unsigned int max_triangles);

  /**
   * @brief      Frees the memory allocated for the class.
   */
  void Free();

  /**
   * @brief      Appends a triangle to the mesh.
   *
   * @param[in]  t     The triangle
   */
  __device__ void AppendTriangle(Triangle t);

  /**
   * @brief      Saves the mesh to an obj file.
   *
   * @param[in]  filename  The filename
   */
  void SaveToFile(const std::string &filename, bool bgr=false);

  /** The triangles that compose the mesh */
  Triangle *triangles_;

  /** The number of triangles that compose the mesh */
  unsigned int num_triangles_;

  /** The maximum number of triangles that the mesh can have */
  unsigned int max_triangles_;
};

}  // namespace tsdfvh

}  // namespace refusion
