#include <helper_functions.hpp>

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

void decompose_projection_matrix(const cv::Mat &projection_matrix, cv::Mat &camera_matrix, cv::Mat &rotation_matrix, cv::Mat &translation_vector) {
    cv::Mat org_translation_vector;
    cv::decomposeProjectionMatrix(projection_matrix, camera_matrix, rotation_matrix, org_translation_vector);

    auto scaling_factor = org_translation_vector.at<double>(3, 0);
    org_translation_vector /= scaling_factor;

    translation_vector = org_translation_vector.rowRange(0, 3);
}

cv::Mat calc_depth_map(const cv::Mat &disp_left, const cv::Mat &k_left, const cv::Mat &t_left, const cv::Mat &t_right, bool rectified) {
    auto f = k_left.at<double>(0, 0);

    double b;
    if (rectified)
        b = t_right.at<double>(0, 0) - t_left.at<double>(0, 0);
    else
        b = t_left.at<double>(0, 0) - t_right.at<double>(0, 0);

    cv::Mat disp_left_copy = disp_left.clone();
    cv::Mat mask = (disp_left_copy == 0.0 | disp_left_copy == -1.0);
    disp_left_copy.setTo(0.1, mask);

    cv::Mat depth_map(disp_left_copy);
    for (int i = 0; i < depth_map.rows; i++) {
        for (int j = 0; j < depth_map.cols; j++) {
            depth_map.row(i).col(j) =  f * b / disp_left_copy.row(i).col(j);
        }
    }

    return depth_map;
}

cv::Mat stereo_2_depth(const cv::Mat &img_left, const cv::Mat &img_right, const cv::Mat &P0, const cv::Mat &P1, 
                        bool rgb, bool verbose, bool rectified) {
    auto disp = compute_left_disparity_map(img_left, img_right, rgb, verbose);

    cv::Mat camera_matrix_left, rotation_matrix_left, translation_vector_left;
    cv::Mat camera_matrix_right, rotation_matrix_right, translation_vector_right;
    decompose_projection_matrix(P0, camera_matrix_left, rotation_matrix_left, translation_vector_left);
    decompose_projection_matrix(P1, camera_matrix_right, rotation_matrix_right, translation_vector_right);

    cv::Mat depth = calc_depth_map(disp, camera_matrix_left, translation_vector_left, translation_vector_right);

    return depth;
}

void extract_features(const cv::Mat &image, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors, const cv::Mat &mask) {
    auto& parameter_server = ParameterParser::GetInstance();
    json parameters = parameter_server.GetParameters();

    auto detector_type = parameters["feature_extractor"]["type"].get<std::string>();

    if (detector_type == "sift") {
        auto detector = cv::SIFT::create();

        if (mask.empty())
            detector->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
        else
            detector->detectAndCompute(image, mask, keypoints, descriptors);
    } else if (detector_type == "orb") {
        auto detector = cv::ORB::create();

        if (mask.empty())
            detector->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
        else
            detector->detectAndCompute(image, mask, keypoints, descriptors);
    }
}

void match_features(const cv::Mat &des1, const cv::Mat &des2, std::vector<std::vector<cv::DMatch>> &matches, bool sort) {
    auto& parameter_server = ParameterParser::GetInstance();
    json parameters = parameter_server.GetParameters();

    auto feature_extractor_type = parameters["feature_extractor"]["type"].get<std::string>();
    auto feature_matcher_type = parameters["feature_matcher"]["type"].get<std::string>();
    auto k = parameters["feature_matcher"]["k"].get<int>();

    if (feature_matcher_type == "BF") {
        bool cross_check = parameters["feature_matcher"]["BF"]["crosscheck"].get<bool>();

        if (feature_extractor_type == "sift") {
            auto matcher = cv::BFMatcher::create(cv::NORM_L2, cross_check);
            matcher->knnMatch(des1, des2, matches, k);
        } else if (feature_extractor_type == "orb") {
            auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING, cross_check);
            matcher->knnMatch(des1, des2, matches, k);
        }
    } else if (feature_matcher_type == "FLANN") {
        int trees = parameters["feature_matcher"]["FLANN"]["trees"].get<int>();
        int checks = parameters["feature_matcher"]["FLANN"]["checks"].get<int>();

        cv::Ptr<cv::flann::IndexParams> index_params = cv::makePtr<cv::flann::KDTreeIndexParams>(trees);
        cv::Ptr<cv::flann::SearchParams> search_params = cv::makePtr<cv::flann::SearchParams>(checks);

        cv::FlannBasedMatcher matcher(index_params, search_params);
        matcher.knnMatch(des1, des2, matches, k);
    }

    if (sort) {
        std::sort(matches.begin(), matches.end(), [](const std::vector<cv::DMatch> &a, const std::vector<cv::DMatch> &b) {
            return a[0].distance < b[0].distance;
        });
    }
}

void filter_matches(const std::vector<std::vector<cv::DMatch>> &matches, std::vector<cv::DMatch> &filtered_matches) {
    auto& parameter_server = ParameterParser::GetInstance();
    json parameters = parameter_server.GetParameters();

    auto distance_threshold = parameters["feature_matcher"]["distance_threshold"].get<double>();
    
    for (const auto &match : matches) {
        if (match[0].distance < distance_threshold * match[1].distance) {
            filtered_matches.push_back(match[0]);
        }
    }
}

void visualize_matches(const cv::Mat &img1, const std::vector<cv::KeyPoint> &kp1,
                       const cv::Mat &img2, const std::vector<cv::KeyPoint> &kp2,
                       const std::vector<cv::DMatch> &matches) {
    cv::Mat image_matches;
    
    cv::drawMatches(img1, kp1, img2, kp2, matches, image_matches);

    cv::imshow("Matched images", image_matches);
}

void plot_matrix(const cv::Mat &mat, const std::string &title, bool normalize) {
    if (normalize) {
        cv::Mat normalized_mat;
        cv::normalize(mat, normalized_mat, 0, 255, cv::NORM_MINMAX);

        normalized_mat.convertTo(normalized_mat, CV_8U);

        cv::imshow(title, normalized_mat);
    } else {
        cv::imshow(title, mat);
    }
}