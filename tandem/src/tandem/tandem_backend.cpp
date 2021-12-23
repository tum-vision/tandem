// Copyright (c) 2021 Lukas Koestler, Nan Yang. All rights reserved.

#include "tandem_backend.h"

class TandemBackendImpl {
public:
  explicit TandemBackendImpl(
      int width, int height, bool dense_tracking,
      DrMvsnet *mvsnet, DrFusion *fusion,
      float mvsnet_discard_percentage,
      Timer *dr_timer,
      std::vector<dso::IOWrap::Output3DWrapper *> const &outputWrapper,
      int mesh_extraction_freq) : \
      width(width), height(height), dense_tracking(dense_tracking), mvsnet(mvsnet), fusion(fusion), \
      mvsnet_discard_percentage(mvsnet_discard_percentage), dr_timer(dr_timer), outputWrapper(outputWrapper), \
      mesh_extraction_freq(mesh_extraction_freq), get_mesh(mesh_extraction_freq > 0) {
    dr_timing = dr_timer != nullptr;
    coarse_tracker_depth_map_A = new TandemCoarseTrackingDepthMap(width, height);
    coarse_tracker_depth_map_B = new TandemCoarseTrackingDepthMap(width, height);
    coarse_tracker_depth_map_valid = nullptr;
    coarse_tracker_depth_map_use_next = coarse_tracker_depth_map_A;
    worker_thread = boost::thread(&TandemBackendImpl::Loop, this);
  };

  ~TandemBackendImpl() {
    delete coarse_tracker_depth_map_B;
    delete coarse_tracker_depth_map_A;
  };


  // Blocking for last input. Non-blocking for this input.
  void CallAsync(
      int view_num_in,
      int index_offset_in,
      int corrected_ref_index_in,
      std::vector<cv::Mat> const &bgrs_in,
      cv::Mat const &intrinsic_matrix_in,
      std::vector<cv::Mat> const &cam_to_worlds_in,
      float depth_min_in,
      float depth_max_in,
      cv::Mat const &coarse_tracker_pose_in
  );

  // Non-blocking
  bool Ready();

  void Wait();

  boost::mutex &GetTrackingDepthMapMutex() { return mut_coarse_tracker; }

  TandemCoarseTrackingDepthMap const *GetTrackingDepthMap() { return coarse_tracker_depth_map_valid; };

private:
  void CallSequential();

  void Loop();

  const bool dense_tracking;

  DrMvsnet *mvsnet = nullptr;
  DrFusion *fusion = nullptr;

  // Will run Loop.
  boost::thread worker_thread;

  // Protects all below variables
  boost::mutex mut;
  bool running = true;
  bool unprocessed_data = false;

  boost::condition_variable newInputSignal;
  boost::condition_variable dataProcessedSignal;

  // Internal
  bool setting_debugout_runquiet = true;
  bool dr_timing;
  int call_num = 0;
  const bool get_mesh = true;

  float mesh_lower_corner[3] = {-5, -5, -5};
  float mesh_upper_corner[3] = {5, 5, 5};
  const int mesh_extraction_freq;

  const int width;
  const int height;
  float mvsnet_discard_percentage;
  Timer *dr_timer;

  std::vector<dso::IOWrap::Output3DWrapper *> const &outputWrapper;

  //
  boost::mutex mut_coarse_tracker;
  TandemCoarseTrackingDepthMap *coarse_tracker_depth_map_valid = nullptr;
  TandemCoarseTrackingDepthMap *coarse_tracker_depth_map_use_next = nullptr;
  TandemCoarseTrackingDepthMap *coarse_tracker_depth_map_A;
  TandemCoarseTrackingDepthMap *coarse_tracker_depth_map_B;

  // data from call
  bool has_to_wait_current = false;
  cv::Mat intrinsic_matrix_current;
  int index_offset_current;
  int view_num_current;
  int corrected_ref_index_current;
  std::vector<cv::Mat> bgrs_current;
  std::vector<cv::Mat> cam_to_worlds_current;
  float depth_min_current;
  float depth_max_current;

  DrMvsnetOutput *output_previous;


  // data from last call
  bool has_to_wait_previous = false;
  cv::Mat intrinsic_matrix_previous;
  int index_offset_previous;
  int view_num_previous;
  int corrected_ref_index_previous;
  std::vector<cv::Mat> bgrs_previous;
  std::vector<cv::Mat> cam_to_worlds_previous;
  float depth_min_previous;
  float depth_max_previous;

};

void TandemBackendImpl::Loop() {
  boost::unique_lock<boost::mutex> lock(mut);
  while (running) {
    if (unprocessed_data) {
      CallSequential();
      unprocessed_data = false;
      dataProcessedSignal.notify_all();
    }
    newInputSignal.wait(lock);
  }
}

void TandemBackendImpl::CallSequential() {
  int id_time;
  call_num++;

  /* --- 3.5 CURRENT: Call MVSNet Async --- */
  std::vector<unsigned char *> bgrs_current_ptr;
  for (auto const &e : bgrs_current) bgrs_current_ptr.push_back(e.data);
  std::vector<float *> cam_to_worlds_current_ptr;
  for (auto const &e : cam_to_worlds_current) cam_to_worlds_current_ptr.push_back((float *) e.data);

  mvsnet->CallAsync(
      height,
      width,
      view_num_current,
      corrected_ref_index_current,
      bgrs_current_ptr.data() + index_offset_current,
      (float *) intrinsic_matrix_current.data,
      cam_to_worlds_current_ptr.data() + index_offset_current,
      depth_min_current,
      depth_max_current,
      mvsnet_discard_percentage,
      false
  );
  has_to_wait_current = true;

  // Here we have the lock (the loop function has it)
  if (has_to_wait_previous && fusion != nullptr) {
    /* --- 3. PREVIOUS: Integrate into Fusion --- */
    if (dr_timing) id_time = dr_timer->start_timing("IntegrateScanAsync");
    fusion->IntegrateScanAsync(bgrs_previous[corrected_ref_index_previous].data, output_previous->depth, (float *) cam_to_worlds_previous[corrected_ref_index_previous].data);
    if (dr_timing) dr_timer->end_timing("IntegrateScanAsync", id_time, !setting_debugout_runquiet);

    /* --- 4. PREVIOUS: RenderAsync for coarse tracker --- */
    std::vector<float const *> poses_to_render_previous;
    if (dense_tracking) poses_to_render_previous.push_back(coarse_tracker_depth_map_use_next->cam_to_world);
    fusion->RenderAsync(poses_to_render_previous);

    /* --- 5. PREVIOUS: Get render result --- */
    std::vector<unsigned char *> bgr_rendered_previous;
    std::vector<float *> depth_rendered_previous;
    fusion->GetRenderResult(bgr_rendered_previous, depth_rendered_previous);
    if (dense_tracking) {
      memcpy(coarse_tracker_depth_map_use_next->depth, depth_rendered_previous[0], sizeof(float) * width * height);
      coarse_tracker_depth_map_use_next->is_valid = true; // atomic

      /* --- 5.5 PREVIOUS: Set Coarse Tracker --- */
      {
        boost::unique_lock<boost::mutex> lock_coarse_tracker(mut_coarse_tracker);

        // Ternary will only be false on first iter
        TandemCoarseTrackingDepthMap *tmp = coarse_tracker_depth_map_valid ? coarse_tracker_depth_map_valid : coarse_tracker_depth_map_B;
        coarse_tracker_depth_map_valid = coarse_tracker_depth_map_use_next;
        coarse_tracker_depth_map_use_next = tmp;
      }
    }

    /* --- 6. PREVIOUS: Get mesh and push to output_previous wrapper --- */
    if (get_mesh && (call_num % mesh_extraction_freq) == 0) {
      if (dr_timing) id_time = dr_timer->start_timing("fusion-mesh");
      fusion->ExtractMeshAsync(mesh_lower_corner, mesh_upper_corner);
      fusion->GetMeshSync();
      if (dr_timing) dr_timer->end_timing("fusion-mesh", id_time, !setting_debugout_runquiet);

      for (dso::IOWrap::Output3DWrapper *ow : outputWrapper) ow->pushDrMesh(fusion->dr_mesh_num, fusion->dr_mesh_vert, fusion->dr_mesh_cols);

      has_to_wait_previous = false;
    }
  }

  // Now we swap *_previous <- *_current
  view_num_previous = view_num_current;
  index_offset_previous = index_offset_current;
  corrected_ref_index_previous = corrected_ref_index_current;
  bgrs_previous = bgrs_current;
  intrinsic_matrix_previous = intrinsic_matrix_current;
  cam_to_worlds_previous = cam_to_worlds_current;
  depth_min_previous = depth_min_current;
  depth_max_previous = depth_max_current;
  has_to_wait_previous = has_to_wait_current;
  unprocessed_data = false;
}


void TandemBackendImpl::CallAsync(
    int view_num_in,
    int index_offset_in,
    int corrected_ref_index_in,
    std::vector<cv::Mat> const &bgrs_in,
    cv::Mat const &intrinsic_matrix_in,
    std::vector<cv::Mat> const &cam_to_worlds_in,
    float depth_min_in,
    float depth_max_in,
    cv::Mat const &coarse_tracker_pose_in
) {
  using std::cerr;
  using std::cout;
  using std::endl;

  if (unprocessed_data) {
    cerr << "Wrong Call Order in TANDEM Backend. Will just return." << endl;
    return;
  }

  {
    boost::unique_lock<boost::mutex> lock(mut);
    {
      // Now we have the lock
      // We will process the MVSNet result for the *_previous data
      // The Loop will finish the processing of the *_previous data
      // The end of Loop will switch *_previous <- *_current

      /* --- 0. Copy input Data --- */
      view_num_current = view_num_in;
      index_offset_current = index_offset_in;
      corrected_ref_index_current = corrected_ref_index_in;
      bgrs_current = bgrs_in;
      intrinsic_matrix_current = intrinsic_matrix_in;
      cam_to_worlds_current = cam_to_worlds_in;
      depth_min_current = depth_min_in;
      depth_max_current = depth_max_in;
      if (dense_tracking) {
        coarse_tracker_depth_map_use_next->is_valid = false; // atomic
        memcpy(coarse_tracker_depth_map_use_next->cam_to_world, coarse_tracker_pose_in.data, sizeof(float) * 16);
      }

      if (has_to_wait_previous) {
        /* --- 1. PREVIOUS: Get MVSNet result --- */
        if (!mvsnet->Ready()) {
          std::cerr << "MVSNET IS NOT READY!!! WHY" << std::endl;
          exit(EXIT_FAILURE);
        }
        output_previous = mvsnet->GetResult();

        /* --- 2. PREVIOUS: Push depth map to output_previous wrapper --- */
        const float depth_max_value_previous = *std::max_element(output_previous->depth, output_previous->depth + width * height);
        for (dso::IOWrap::Output3DWrapper *ow : outputWrapper) {
          ow->pushDrKfImage(bgrs_previous[corrected_ref_index_previous].data);
          ow->pushDrKfDepth(output_previous->depth_dense, depth_min_previous, depth_max_value_previous);
        }
      }

    }

    unprocessed_data = true;
    newInputSignal.notify_all();
  }
}

bool TandemBackendImpl::Ready() {
  return !unprocessed_data && mvsnet->Ready();
}

TandemCoarseTrackingDepthMap::TandemCoarseTrackingDepthMap(int width, int height) {
  depth = (float *) malloc(sizeof(float) * width * height);
}

TandemCoarseTrackingDepthMap::~TandemCoarseTrackingDepthMap() {
  free(depth);
}

TandemBackend::TandemBackend(int width, int height, bool dense_tracking,
                             DrMvsnet *mvsnet, DrFusion *fusion,
                             float mvsnet_discard_percentage,
                             Timer *dr_timer,
                             std::vector<dso::IOWrap::Output3DWrapper *> const &outputWrapper,
                             int mesh_extraction_freq) {
  impl = new TandemBackendImpl(width, height, dense_tracking, mvsnet, fusion, mvsnet_discard_percentage, dr_timer, outputWrapper, mesh_extraction_freq);
}

TandemBackend::~TandemBackend() {
  delete impl;
}

bool TandemBackend::Ready() {
  return impl->Ready();
}

void TandemBackend::CallAsync(int view_num_in, int index_offset_in, int corrected_ref_index_in, const std::vector<cv::Mat> &bgrs_in, const cv::Mat &intrinsic_matrix_in, const std::vector<cv::Mat> &cam_to_worlds_in, float depth_min_in,
                              float depth_max_in, const cv::Mat &coarse_tracker_pose_in) {
  impl->CallAsync(
      view_num_in,
      index_offset_in,
      corrected_ref_index_in,
      bgrs_in,
      intrinsic_matrix_in,
      cam_to_worlds_in,
      depth_min_in,
      depth_max_in,
      coarse_tracker_pose_in
  );

}

boost::mutex &TandemBackend::GetTrackingDepthMapMutex() {
  return impl->GetTrackingDepthMapMutex();
}

TandemCoarseTrackingDepthMap const *TandemBackend::GetTrackingDepthMap() {
  return impl->GetTrackingDepthMap();
}

void TandemBackend::Wait() {
  impl->Wait();
}

void TandemBackendImpl::Wait() {
  boost::unique_lock<boost::mutex> lock(mut);
  while (unprocessed_data) {
    dataProcessedSignal.wait(lock);
  }
  mvsnet->Wait();
  if (!Ready()) {
    std::cerr << "TandemBackendImpl must be Ready() after Wait(). Something went wrong." << std::endl;
    exit(EXIT_FAILURE);
  }
}

float get_idepth_quantile(int n, const float *const idepth, float fraction) {
  std::vector<float> idepth_sorted(idepth, idepth + n);
  const int quantile_n = (int) (fraction * (float) n);
  auto m = idepth_sorted.begin() + quantile_n;
  std::nth_element(idepth_sorted.begin(), m, idepth_sorted.end());
  const float idepth_quantile = idepth_sorted[quantile_n];
  return 1.0f / idepth_quantile;
}
