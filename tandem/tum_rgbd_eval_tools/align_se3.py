import sys
import numpy as np
import argparse
import associate



def align_sim3(gt_pos, est_pos, align_scale=True, eval_rmse=False):
    """
    Align two trajs with sim3
    :param gt_pos: ground truth pose, shape [N, 3]
    :param est_pos: estimated pose, shape [N, 3]
    :param align_scale: whether to calculate the scale (sim3). Set to 1.0 if False
    :param eval_rmse: whether to evalute rmse. Set to -1 if False
    :return: R, t, scale, rmse
    """
    centroid_est = est_pos.mean(0)
    centroid_gt = gt_pos.mean(0)
    eval_zerocentered = est_pos - centroid_est
    gt_zerocentered = gt_pos - centroid_gt

    H = np.dot(np.transpose(eval_zerocentered),
               gt_zerocentered) / gt_pos.shape[0]

    U, D, Vh = np.linalg.svd(H)
    S = np.array(np.identity(3))
    if (np.linalg.det(U) * np.linalg.det(Vh) < 0):
        S[2, 2] = -1

    R_inv = np.dot(U, np.dot(S, Vh))
    R = np.transpose(R_inv)

    rot_centroid_est = np.dot(R, np.transpose(centroid_est))
    rot_zerocentered_est = np.dot(est_pos, R_inv) - rot_centroid_est

    scale = np.trace(np.dot(np.diag(D), S)) / np.mean((np.sum(eval_zerocentered **
                                                       2, 1))) if align_scale else 1.

    t = np.transpose(centroid_gt) - scale * rot_centroid_est

    rmse = -1.
    if eval_rmse:
        diff = (scale * rot_zerocentered_est - gt_zerocentered)
        size = np.shape(diff)
        rmse = np.sqrt(np.sum(np.multiply(diff, diff)) / size[0])

    return R, t, scale, rmse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Align two trajectories in SE(3)')
    parser.add_argument('first_file', help='ground truth trajectory (format: timestamp tx ty tz qx qy qz qw)')
    parser.add_argument('second_file', help='estimated trajectory (format: timestamp tx ty tz qx qy qz qw)')
    parser.add_argument('--offset', help='time offset added to the timestamps of the second file (default: 0.0)',default=0.0)
    parser.add_argument('--max_difference', help='maximally allowed time difference for matching entries (default: 0.02)',default=0.02)
    parser.add_argument('--verbose', help='print all evaluation data (otherwise, only the RMSE absolute translational error in meters after alignment will be printed)', action='store_true')
    args = parser.parse_args()

    first_list = associate.read_file_list(args.first_file)
    second_list = associate.read_file_list(args.second_file)

    matches = associate.associate(first_list, second_list,float(args.offset),float(args.max_difference))    
    if len(matches)<2:
        sys.exit("Couldn't find matching timestamp pairs between groundtruth and estimated trajectory! Did you choose the correct sequence?")


    first_xyz = np.array([[float(value) for value in first_list[a][0:3]] for a,b in matches])
    second_xyz = np.array([[float(value) for value in second_list[b][0:3]] for a,b in matches])

    R, t, scale, rmse = align_sim3(first_xyz, second_xyz, align_scale=True, eval_rmse=True)

    print("--scale " + str(scale))
    if args.verbose:
        print("RMSE:   " + str(rmse) + " m")