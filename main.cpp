#include <iostream>

#include <Eigen/Dense>

#include <opencv2/opencv.hpp>

#include <dataset_handler.hpp>
#include <parse_params.hpp>
#include <helper_functions.hpp>

int main() {
    const std::string img_folder = "../data";
    const std::string vo_parameters_file_path = "../params/vo_params.json";
    const std::string trajectory_file_path = "../data/trajectory/estimated_trajectory.txt";

    // Initialize parameter parser
    ParameterParser& parameter_parser = ParameterParser::GetInstance();
    parameter_parser.ReadJsonFile(vo_parameters_file_path);

    DatasetHandler dataset_handler(img_folder, "00", false, 0);
    
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<cv::Mat> trajectory;
    visual_odometry(dataset_handler, cv::Mat(), -1, &trajectory);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << "Time to perform odometry: " << duration/1000.0 << "ms" << "\n";

    std::cout << "Saving the trajectory into " << trajectory_file_path << " ...\n";
    save_trajectory(trajectory, trajectory_file_path);

    std::cout << "Calculating trajectory estimation error...\n";

    Error error = calculate_error(dataset_handler.gt_data_, trajectory);
    std::cout << "\tmae: " << error.GetMAE() << "\n";
    std::cout << "\tmse: " << error.GetMSE() << "\n";
    std::cout << "\trmse: " << error.GetRMSE() << "\n";

    return 0;
}