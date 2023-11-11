#include <iostream>

#include <Eigen/Dense>

#include <opencv2/opencv.hpp>

#include <dataset_handler.hpp>
#include <parse_params.hpp>
#include <helper_functions.hpp>

int main() {
    const std::string img_folder = "../data";
    const std::string vo_parameters_file_path = "../params/vo_params.json";

    // Initialize parameter parser
    ParameterParser& parameter_parser = ParameterParser::GetInstance();
    parameter_parser.ReadJsonFile(vo_parameters_file_path);

    DatasetHandler dataset_handler(img_folder, "00", false, 0);

    auto left_image = *(*dataset_handler.left_image_loader_it_);
    auto right_image = *(*dataset_handler.right_image_loader_it_);

    auto depth = stereo_2_depth(left_image, right_image, dataset_handler.P0_, dataset_handler.P1_);  

    std::vector<cv::KeyPoint> kp0, kp1;
    cv::Mat des0, des1;
    extract_features(left_image, kp0, des0);
    extract_features(right_image, kp1, des1);

    std::vector<std::vector<cv::DMatch>> matches;
    match_features(des0, des1, matches);
    std::vector<cv::DMatch> good_matches;
    filter_matches(matches, good_matches);

    std::cout << "Number of matches: " << matches.size() << "\n";
    std::cout << "Number of matches after filtering: " << good_matches.size() << "\n";

    visualize_matches(left_image, kp0, right_image, kp1, good_matches);

    // plot_matrix(depth, "Normalized depth map", true);
    cv::waitKey(0);

    return 0;
}