#include <helper_functions.hpp>
#include <parse_params.hpp>

#include <chrono>
#include <algorithm>

cv::Mat compute_left_disparity_map(const cv::Mat &img_left, const cv::Mat &img_right, bool rgb, bool verbose) {

    auto& parameter_server = ParameterParser::GetInstance();
    json parameters = parameter_server.GetParameters();

    auto matcher_name = parameters["matcher"]["type"].get<std::string>();
    auto sad_window = parameters["matcher"]["sad_window"].get<int>();
    auto num_disparities = sad_window * 16;
    auto block_size = parameters["matcher"]["block_size"].get<int>();
    auto min_disparity = parameters["matcher"]["min_disparity"].get<int>();
    auto disp_12_max_diff = parameters["matcher"]["disp_12_max_diff"].get<int>();
    auto pre_filter_cap = parameters["matcher"]["pre_filter_cap"].get<int>();
    auto uniqueness_ratio = parameters["matcher"]["uniqueness_ratio"].get<int>();
    auto speckle_window_size = parameters["matcher"]["speckle_window_size"].get<int>();
    auto speckle_range = parameters["matcher"]["speckle_range"].get<int>();
    auto mode = parameters["matcher"]["mode"].get<int>();
    auto P1 = 8 * 3 * sad_window * sad_window;
    auto P2 = 32 * 3 * sad_window * sad_window;

    cv::Ptr<cv::StereoBM> matcher_bm = cv::StereoBM::create();
    cv::Ptr<cv::StereoSGBM> matcher_sgbm = cv::StereoSGBM::create();

    cv::Mat left_image_to_be_processed;
    cv::Mat right_image_to_be_processed;
    if (rgb) {
        cv::cvtColor(img_left, left_image_to_be_processed, cv::COLOR_GRAY2BGR);
        cv::cvtColor(img_right, right_image_to_be_processed, cv::COLOR_GRAY2BGR);
    }
    left_image_to_be_processed = img_left.clone();
    right_image_to_be_processed = img_right.clone();

    cv::Mat disp_left_scaled, disp_left;
    if (matcher_name == "bm") {
        matcher_bm->setNumDisparities(num_disparities);
        matcher_bm->setBlockSize(block_size);

        auto start = std::chrono::high_resolution_clock::now();

        matcher_bm->compute(left_image_to_be_processed, right_image_to_be_processed, disp_left_scaled);
        disp_left_scaled.convertTo(disp_left, CV_32F, 1.0);
        disp_left /= 16.0;

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        std::transform(matcher_name.begin(), matcher_name.end(), matcher_name.begin(), ::toupper);
        std::cout << "Time taken to calculate disparity map using Stereo" << matcher_name << ": " << duration << " microseconds" << std::endl;
    } else if (matcher_name == "sgbm") {
        matcher_sgbm->setNumDisparities(num_disparities);
        matcher_sgbm->setMinDisparity(min_disparity);
        matcher_sgbm->setBlockSize(block_size);
        matcher_sgbm->setP1(P1);
        matcher_sgbm->setP2(P2);
        matcher_sgbm->setMode(mode);

        auto start = std::chrono::high_resolution_clock::now();

        matcher_sgbm->compute(left_image_to_be_processed, right_image_to_be_processed, disp_left_scaled);
        disp_left_scaled.convertTo(disp_left, CV_32F, 1.0);
        disp_left /= 16.0;

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        std::transform(matcher_name.begin(), matcher_name.end(), matcher_name.begin(), ::toupper);
        std::cout << "Time taken to calculate disparity map using Stereo" << matcher_name << ": " << duration << " microseconds" << std::endl;
    }

    return disp_left;
}