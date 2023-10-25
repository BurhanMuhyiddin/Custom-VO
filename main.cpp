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

    auto disp = compute_left_disparity_map(left_image, right_image, true);

    cv::imshow("disparity_map", disp);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}