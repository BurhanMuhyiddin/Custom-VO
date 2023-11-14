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
        std::cout << "Time taken to calculate disparity map using Stereo" << matcher_name << ": " << duration/1000.0 << "ms" << std::endl;
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
        std::cout << "Time taken to calculate disparity map using Stereo" << matcher_name << ": " << duration/1000.0 << "ms" << std::endl;
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
        // std::cout << match[0].distance << ", " << match[1].distance << ", " << distance_threshold * match[1].distance << "\n";
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

void estimate_motion(const std::vector<cv::DMatch> &matches,
                     const std::vector<cv::KeyPoint> &kp1,
                     const std::vector<cv::KeyPoint> &kp2,
                     const cv::Mat &k,
                     cv::Mat &rmat,
                     cv::Mat &tvec,
                     std::vector<cv::Point2f> &img1_points,
                     std::vector<cv::Point2f> &img2_points,
                     const cv::Mat &depth1,
                     int max_depth) {
    std::vector<cv::Point2f> img1_points_org;
    std::vector<cv::Point2f> img2_points_org;
    for (const auto &match : matches) {
        img1_points_org.push_back(kp1[match.queryIdx].pt);
        img2_points_org.push_back(kp2[match.trainIdx].pt);
    }

    if (!depth1.empty()) {
        float cx = k.at<double>(0, 2);
        float cy = k.at<double>(1, 2);
        float fx = k.at<double>(0, 0);
        float fy = k.at<double>(1, 1);
        // std::cout << cx << "," << cy << "," << fx << "," << fy << "\n";

        std::vector<int> delete_buffer;
        std::vector<cv::Point3f> object_points;
        for (int i = 0; i < img1_points_org.size(); i++) {
            float u = img1_points_org[i].x;
            float v = img1_points_org[i].y;
            float z = depth1.at<float>(int(v), int(u));

            if (z > max_depth) {
                delete_buffer.push_back(i);
                continue;
            }

            float x = z * (u - cx) / fx;
            float y = z * (v - cy) / fy;

            object_points.push_back(cv::Point3f(x, y, z));
        }

        for (int i = 0; i < img1_points_org.size(); i++) {
            if (std::find(delete_buffer.begin(), delete_buffer.end(), i) == delete_buffer.end()) {
                img1_points.push_back(img1_points_org[i]);
                img2_points.push_back(img2_points_org[i]);
            }
        }

        // std::cout << object_points.size() << "," << img1_points.size() << "," << img2_points.size() << "\n";

        cv::Mat rvec, inliers;
        cv::solvePnPRansac(object_points, img2_points, k, cv::Mat(), rvec, tvec, inliers);

        // std::cout << rvec << "\n";

        cv::Rodrigues(rvec, rmat);
    } else {
        cv::Mat E;
        E = cv::findEssentialMat(img1_points, img2_points, k, cv::RANSAC);
        cv::recoverPose(E, img1_points, img2_points, k, rmat, tvec);
    }
}

void visual_odometry(DatasetHandler &handler, const cv::Mat &mask, int subset, std::vector<cv::Mat>* trajectory) {
    auto& parameter_server = ParameterParser::GetInstance();
    json parameters = parameter_server.GetParameters();

    auto feature_extractor_type = parameters["feature_extractor"]["type"].get<std::string>();
    auto feature_matcher_type = parameters["feature_matcher"]["type"].get<std::string>();
    auto stereo_matcher_type = parameters["matcher"]["type"].get<std::string>();
    auto depth_calculation_type = parameters["depth_calculation"]["type"].get<std::string>();
    auto do_plot = parameters["plotting"]["do_plot"].get<bool>();
    auto filter_match_distance = parameters["feature_matcher"]["distance_threshold"].get<float>();

    std::cout << "Generating disparities with Stereo" << to_uppercase(stereo_matcher_type) << "\n";
    std::cout << "Detecting features with " << to_uppercase(feature_extractor_type) << " and matrching with " << feature_matcher_type << "\n";
    std::cout << "Filtering feature matches at threshold of " << filter_match_distance << "*distance" << "\n";

    int num_frames = (subset > 0) ? std::min(subset, handler.num_frames_) : handler.num_frames_;

    cv::Mat T_tot = cv::Mat::eye(4, 4, CV_64F);
    trajectory->push_back(T_tot.rowRange(0, 3).clone());

    int imheight = handler.im_height_;
    int imwidth = handler.im_width_;

    cv::Mat k_left, r_left, t_left;
    decompose_projection_matrix(handler.P0_, k_left, r_left, t_left);

    auto img_left = (*(*handler.left_image_loader_it_)).clone();
    auto img_right = (*(*handler.right_image_loader_it_)).clone();
    for (int i = 0; i < num_frames-1; i++) {
        auto start = std::chrono::high_resolution_clock::now();

        cv::Mat depth;
        if (depth_calculation_type == "stereo") {
            depth = stereo_2_depth(img_left, img_right, handler.P0_, handler.P1_);
        }

        cv::Mat image_plus1 = (*(std::next(*handler.left_image_loader_it_))).clone();
        std::vector<cv::KeyPoint> kp0, kp1;
        cv::Mat des0, des1;
        extract_features(img_left, kp0, des0);
        extract_features(image_plus1, kp1, des1);

        // cv::imshow("img_left", img_left);
        // cv::imshow("image_plus1", image_plus1);
        // cv::waitKey(0);

        std::vector<std::vector<cv::DMatch>> matches_unfilt;
        match_features(des0, des1, matches_unfilt);

        // std::cout << matches_unfilt.size() << "\n";
        
        std::vector<cv::DMatch> matches;
        filter_matches(matches_unfilt, matches);

        if (matches.size() < 50) {
            std::cout << "Number of matches is less than 50: " << matches.size() << "\n";
            cv::imshow("img_left", img_left);
            cv::imshow("image_plus1", image_plus1);
            cv::waitKey(0);
            continue;
        }

        // std::cout << matches.size() << "\n";

        cv::Mat rmat, tvec;
        std::vector<cv::Point2f> img1_points, img2_points;
        estimate_motion(matches, kp0, kp1, k_left, rmat, tvec, img1_points, img2_points, depth);

        cv::Mat Tmat = cv::Mat::eye(4, 4, CV_64F);
        rmat.copyTo(Tmat(cv::Rect(0, 0, 3, 3)));
        tvec.copyTo(Tmat(cv::Rect(3, 0, 1, 3)));

        // std::cout << rmat << "\n";
        // std::cout << "----------------\n";
        // std::cout << tvec << "\n";
        // std::cout << "----------------\n";
        // std::cout << Tmat << "\n";

        T_tot *= Tmat.inv();

        trajectory->push_back(T_tot.rowRange(0, 3).clone());

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        std::cout << "Time to compute frame" << i << ": " << duration/1000.0 << "ms" << "\n";

        *handler.left_image_loader_it_ = std::next(*handler.left_image_loader_it_);
        *handler.right_image_loader_it_ = std::next(*handler.right_image_loader_it_);
        if ((*handler.left_image_loader_it_ != *handler.left_image_loader_end_it_) &&
           (*handler.right_image_loader_it_ != *handler.right_image_loader_end_it_)) {
            img_left = (*(*handler.left_image_loader_it_)).clone();
            img_right = (*(*handler.right_image_loader_it_)).clone();
        }
    }
    std::cout << "Finished Visual Odometry...\n";
}

Error calculate_error(const std::vector<cv::Mat> &ground_truth, const std::vector<cv::Mat> &estimated) {
    Error error;

    int nframes_est = estimated.size();

    double mse = get_mse(ground_truth, estimated);
    double mae = get_mae(ground_truth, estimated);

    error.SetMSE(mse);
    error.SetMAE(mae);
    error.SetRMSE(sqrt(mse));

    return error;
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