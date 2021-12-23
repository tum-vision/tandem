// Copyright (c) 2021 Lukas Koestler, Nan Yang. All rights reserved.

#include "cuda_coarse_tracker_private.h"
#include <cnpy.h>

#include <stdlib.h>
#include <thread>
#include <chrono>

int main(int argc, char *argv[]) {

  typedef float Accum;

  int loops = 1;
  if (argc == 2) loops = std::atoi(argv[1]);

  auto npy = cnpy::npy_load("cct_data/nfxfyaffLLb.npy");
  int n = (int) npy.data<float>()[0];

  int w = 640;
  int h = 480;

  auto npy2 = cnpy::npy_load("cct_data/setting_huberTH_maxEnergy_cutoffTH.npy");


  float *refToNew_Ki;
  cudaMalloc((void **) &refToNew_Ki, sizeof(float) * (16 + 9));
  float *dInew;
  cudaMalloc((void **) &dInew, sizeof(float) * 3 * w * h);
  float *pc_u;
  cudaMalloc((void **) &pc_u, sizeof(float) * n);
  float *pc_v;
  cudaMalloc((void **) &pc_v, sizeof(float) * n);
  float *pc_idepth;
  cudaMalloc((void **) &pc_idepth, sizeof(float) * n);
  float *pc_color;
  cudaMalloc((void **) &pc_color, sizeof(float) * n);
  float *warped_u;
  cudaMalloc((void **) &warped_u, sizeof(float) * n);
  float *warped_v;
  cudaMalloc((void **) &warped_v, sizeof(float) * n);
  float *warped_dx;
  cudaMalloc((void **) &warped_dx, sizeof(float) * n);
  float *warped_dy;
  cudaMalloc((void **) &warped_dy, sizeof(float) * n);
  float *warped_idepth;
  cudaMalloc((void **) &warped_idepth, sizeof(float) * n);
  float *warped_residual;
  cudaMalloc((void **) &warped_residual, sizeof(float) * n);
  float *warped_weight;
  cudaMalloc((void **) &warped_weight, sizeof(float) * n);
  {
    auto npy = cnpy::npy_load("cct_data/refToNew_Ki_host.npy");
    cudaMemcpy(refToNew_Ki, npy.data<float>(), sizeof(float) * (16 + 9), cudaMemcpyHostToDevice);
  }
  {
    auto npy = cnpy::npy_load("cct_data/dInew.npy");
    std::cout << npy.shape[0] << std::endl;
    cudaMemcpy(dInew, npy.data<float>(), sizeof(float) * 3 * w * h, cudaMemcpyHostToDevice);
  }
  {
    auto npy = cnpy::npy_load("cct_data/pc_u.npy");
    cudaMemcpy(pc_u, npy.data<float>(), sizeof(float) * n, cudaMemcpyHostToDevice);
  }
  {
    auto npy = cnpy::npy_load("cct_data/pc_v.npy");
    cudaMemcpy(pc_v, npy.data<float>(), sizeof(float) * n, cudaMemcpyHostToDevice);
  }
  {
    auto npy = cnpy::npy_load("cct_data/pc_idepth.npy");
    cudaMemcpy(pc_idepth, npy.data<float>(), sizeof(float) * n, cudaMemcpyHostToDevice);
  }
  {
    auto npy = cnpy::npy_load("cct_data/pc_color.npy");
    cudaMemcpy(pc_color, npy.data<float>(), sizeof(float) * n, cudaMemcpyHostToDevice);
  }
  {
    auto npy = cnpy::npy_load("cct_data/warped_u.npy");
    cudaMemcpy(warped_u, npy.data<float>(), sizeof(float) * n, cudaMemcpyHostToDevice);
  }
  {
    auto npy = cnpy::npy_load("cct_data/warped_v.npy");
    cudaMemcpy(warped_v, npy.data<float>(), sizeof(float) * n, cudaMemcpyHostToDevice);
  }
  {
    auto npy = cnpy::npy_load("cct_data/warped_dx.npy");
    cudaMemcpy(warped_dx, npy.data<float>(), sizeof(float) * n, cudaMemcpyHostToDevice);
  }
  {
    auto npy = cnpy::npy_load("cct_data/warped_dy.npy");
    cudaMemcpy(warped_dy, npy.data<float>(), sizeof(float) * n, cudaMemcpyHostToDevice);
  }
  {
    auto npy = cnpy::npy_load("cct_data/warped_idepth.npy");
    cudaMemcpy(warped_idepth, npy.data<float>(), sizeof(float) * n, cudaMemcpyHostToDevice);
  }
  {
    auto npy = cnpy::npy_load("cct_data/warped_residual.npy");
    cudaMemcpy(warped_residual, npy.data<float>(), sizeof(float) * n, cudaMemcpyHostToDevice);
  }
  {
    auto npy = cnpy::npy_load("cct_data/warped_weight.npy");
    cudaMemcpy(warped_weight, npy.data<float>(), sizeof(float) * n, cudaMemcpyHostToDevice);
  }

  Accum *outputs;
  cudaMalloc((Accum **) &outputs, sizeof(Accum) * 45);

  Accum *outputs_res;
  cudaMalloc((Accum **) &outputs_res, sizeof(Accum) * 7);

  int least_prio, greatest_prio;
  cudaStream_t stream;
  cudaDeviceGetStreamPriorityRange(&least_prio, &greatest_prio);
  std::cerr << "Loops: " << loops << std::endl;
  std::cerr << "Priority range: " << least_prio << " - " << greatest_prio << std::endl;
  cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatest_prio);

  float2 affLL;
  affLL.x = npy.data<float>()[3];
  affLL.y = npy.data<float>()[4];

  std::this_thread::sleep_for(std::chrono::milliseconds{1000});


  for (int rep = 0; rep < 10; rep++) {
    cudaMemsetAsync(outputs, 0, sizeof(Accum) * 45, stream);
    callCalcGKernel(128, stream,
                    npy.data<float>()[1],
                    npy.data<float>()[2],
                    affLL,
                    npy.data<float>()[5],
                    n, loops,
                    pc_color,
                    warped_u,
                    warped_v,
                    warped_dx,
                    warped_dy,
                    warped_idepth,
                    warped_residual,
                    warped_weight,
                    outputs);
  }

  for (int rep = 0; rep < 10; rep++) {
    cudaMemsetAsync(outputs_res, 0, sizeof(Accum) * 7, stream);
    callCalcResKernel(128, stream,
                      npy2.data<float>()[0],
                      w, h,
                      npy.data<float>()[1],
                      npy.data<float>()[2],
                      0.5 * w, 0.5 * h,

                      refToNew_Ki,
                      refToNew_Ki + 16,
                      affLL,
                      npy2.data<float>()[1],
                      npy2.data<float>()[2],
                      n,
                      pc_u,
                      pc_v,
                      pc_idepth,
                      pc_color,
                      dInew,
                      warped_u,
                      warped_v,
                      warped_dx,
                      warped_dy,
                      warped_idepth,
                      warped_residual,
                      warped_weight,
                      outputs_res
    );
  }

  cudaDeviceSynchronize();

  Accum *buf = (Accum *) malloc(sizeof(Accum) * 45);
//  float ref[45] = {8417579106304.00000, -1625169199104.00000, -1728827097088.00000, 487033143296.00000, 7605915222016.00000, -44643995648.00000, -13235101696.00000, -128642360.00000, 440491680.00000, 7978107797504.00000,
//                   -293547835392.00000, -8113838096384.00000, -374495805440.00000, 1513772417024.00000, 17350957056.00000, 121257512.00000, -16407692.00000, 1174363832320.00000, 400823812096.00000, -1293852606464.00000, -112537878528.00000,
//                   -102340456.00000, -9569832.00000, -31895036.00000, 10376914665472.00000, 220200157184.00000, -2010962329600.00000, -12037475328.00000, -57355648.00000, 33462538.00000, 11361430011904.00000, 1444491165696.00000,
//                   -19827128320.00000, -219718272.00000, 642264704.00000, 2473342009344.00000, -4100401152.00000, -54749400.00000, -6160792.50000, 34262847488.00000, 198524144.00000, -423276192.00000, 2228175.75000, 91339.85156,
//                   15358524.00000};

  Accum ref[45] = {841757999056.3202, -162516820703.6704, -172882233278.4789, 48703305894.3368, 760590549243.0692, -4464408848.9532, -1323512083.9209, -12864211.8993, 44048900.5819, 797813637793.8168, -29354793273.4161, -811379618811.9968,
                   -37449520793.1029, 151377780294.1273, 1735095081.1325, 12125757.9028, -1640768.5271, 117436242842.5514, 40082298930.2422, -129385627377.7247, -11253784553.4892, -10234039.7666, -956983.9401, -3189485.0642,
                   1037689919110.7374, 22019985214.8790, -201095314311.3658, -1203758437.9221, -5735566.8183, 3346251.3564, 1136143317826.7681, 144448785857.3471, -1982706002.2349, -21971835.8009, 64226754.9306, 247333016538.2271,
                   -410038091.5979, -5474938.0959, -616080.1733, 3426286103.9732, 19852437.9155, -42327684.8567, 222817.1944, 9134.0468, 1535855.3693
  };
  cudaMemcpy(buf, outputs, sizeof(Accum) * 45, cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  for (int i = 0; i < 45; i++) printf("%9.4f%s", buf[i], i == 44 ? "\n\n" : ", ");
  for (int i = 0; i < 45; i++) printf("%.10e%s", std::abs(buf[i] - ref[i]), i == 44 ? "\n" : " ");
  return EXIT_SUCCESS;
}
