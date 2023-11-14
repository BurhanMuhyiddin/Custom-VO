import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_trajectory_data(trajectroy_file_path: str, sep: str=',', header: int=None):
    df = pd.read_csv(trajectroy_file_path, sep=sep, header=header)

    return df

def main():
    gt_trajectory_file_path = "../data/poses/00.txt"
    estimated_trajectory_file_path = "../data/trajectory/estimated_trajectory.txt"

    gt_traj_df = read_trajectory_data(gt_trajectory_file_path, sep=' ')
    est_traj_df = read_trajectory_data(estimated_trajectory_file_path)

    gt_traj = gt_traj_df.to_numpy()
    est_traj = est_traj_df.to_numpy()
    
    gt_traj_len = gt_traj.shape[0]
    est_traj_len = est_traj.shape[0]

    data_len = min(gt_traj_len, est_traj_len)

    trans_cols = [3, 7, 11]
    gt_traj_trans = gt_traj[:, trans_cols]
    est_traj_trans = est_traj[:, trans_cols]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot 3D GT trajectory
    gt_x = gt_traj_trans[0:data_len, 0]
    gt_y = gt_traj_trans[0:data_len, 1]
    gt_z = gt_traj_trans[0:data_len, 2]

    ax.plot(gt_x, gt_y, gt_z, c='b')

    # Plot 3D estimated trajectory
    est_x = est_traj_trans[0:data_len, 0]
    est_y = est_traj_trans[0:data_len, 1]
    est_z = est_traj_trans[0:data_len, 2]

    ax.plot(est_x, est_y, est_z, c='r')

    plt.show()

if __name__ == '__main__':
    main()