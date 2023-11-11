#ifndef HELPER_FUNCTIONS_HPP
#define HELPER_FUNCTIONS_HPP

#include <parse_params.hpp>

#include <chrono>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

cv::Mat compute_left_disparity_map(const cv::Mat &img_left, const cv::Mat &img_right, bool rgb=false, bool verbose=false);

void decompose_projection_matrix(const cv::Mat &projection_matrix, cv::Mat &camera_matrix, cv::Mat &rotation_matrix, cv::Mat &translation_vector);

cv::Mat calc_depth_map(const cv::Mat &disp_left, const cv::Mat &k_left, const cv::Mat &t_left, const cv::Mat &t_right, bool rectified=true);

cv::Mat stereo_2_depth(const cv::Mat &img_left, const cv::Mat &img_right, const cv::Mat &P0, const cv::Mat &P1,
                        bool rgb=false, bool verbose=false, bool rectified=true);

void extract_features(const cv::Mat &image, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors, const cv::Mat &mask=cv::Mat());

void match_features(const cv::Mat &des1, const cv::Mat &des2, std::vector<std::vector<cv::DMatch>> &matches, bool sort=true);

void filter_matches(const std::vector<std::vector<cv::DMatch>> &matches, std::vector<cv::DMatch> &filtered_matches);

void visualize_matches(const cv::Mat &img1, const std::vector<cv::KeyPoint> &kp1,
                       const cv::Mat &img2, const std::vector<cv::KeyPoint> &kp2,
                       const std::vector<cv::DMatch> &matches);

void plot_matrix(const cv::Mat &mat, const std::string &title, bool normalize=false);

#endif //HELPER_FUNCTIONS_HPP