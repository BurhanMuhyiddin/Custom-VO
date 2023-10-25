#ifndef HELPER_FUNCTIONS_HPP
#define HELPER_FUNCTIONS_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

cv::Mat compute_left_disparity_map(const cv::Mat &img_left, const cv::Mat &img_right, bool rgb=false, bool verbose=false);

#endif //HELPER_FUNCTIONS_HPP