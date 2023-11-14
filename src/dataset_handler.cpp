#include <dataset_handler.hpp>

DatasetHandler::DatasetHandler(const std::string &data_path, const std::string &sequence, bool lidar, int imread_mode) {
    fs::path data_dir(data_path);

    seq_dir_ = data_dir / "sequences" / sequence;
    poses_dir_ = data_dir / "poses" / (sequence + ".txt");

    fs::path left_images_dir = seq_dir_ / "image_0";
    fs::path right_images_dir = seq_dir_ / "image_1";
    fs::path calib_file_dir = seq_dir_ / "calib.txt";
    fs::path times_file_dir = seq_dir_ / "times.txt";

    std::vector<std::string> left_images = GetImgPathInDirectory(left_images_dir);
    std::vector<std::string> right_images = GetImgPathInDirectory(right_images_dir);
    
    num_frames_ = left_images.size();

    left_image_loader_ = std::make_unique<ImageLoader>(left_images, imread_mode);
    right_image_loader_ = std::make_unique<ImageLoader>(right_images, imread_mode);
    left_image_loader_it_ = std::make_unique<ImageLoader::iterator>(left_image_loader_->begin());
    left_image_loader_end_it_ = std::make_unique<ImageLoader::iterator>(left_image_loader_->end());
    right_image_loader_it_ = std::make_unique<ImageLoader::iterator>(right_image_loader_->begin());
    right_image_loader_end_it_ = std::make_unique<ImageLoader::iterator>(right_image_loader_->end());

    auto calibration_matrices = ReadTxtToMat(calib_file_dir, 3, 4);
    if (calibration_matrices.size() == 5) {
        std::cout << "Calibration matrices read successfully." << std::endl;

        P0_ = calibration_matrices[0];
        P1_ = calibration_matrices[1];
        P2_ = calibration_matrices[2];
        P3_ = calibration_matrices[3];
        Tr_ = calibration_matrices[4];
    } else {
        std::cerr << "Calibration matrices couldn't be read." << std::endl;
    }

    auto time_sequence = ReadTxtToMat(times_file_dir, 1, 1);
    if (time_sequence.size() != 0) {
        std::cout << "Times sequence read successfully." << std::endl;

        time_sequence_ = cv::Mat::zeros(time_sequence.size(), 1, CV_64F);
        for (int i = 0; i < time_sequence.size(); i++) {
            time_sequence_.at<double>(i, 0) = time_sequence[i].at<double>(0, 0);
        }
    } else {
        std::cerr << "Times sequence couldn't be read." << std::endl;
    }

    gt_data_ = ReadTxtToMat(poses_dir_, 3, 4);
    if (gt_data_.size() != 0) {
        std::cout << "GT data read successfully." << std::endl;
    } else  {
        std::cout << "GT data couldn't be read." << std::endl;
    }
}

std::vector<std::string> DatasetHandler::GetImgPathInDirectory(const std::string& directory_path) const {
    std::vector<std::string> img_paths;
    try {
        for (const auto &entry : fs::directory_iterator(directory_path)) {
            if (entry.is_regular_file()) {
                img_paths.push_back(entry.path().string());
            }
        }
        std::sort(img_paths.begin(), img_paths.end(), std::greater<std::string>());
    } catch(const fs::filesystem_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    
    return img_paths;
}

std::vector<cv::Mat> DatasetHandler::ReadTxtToMat(const std::string &file_path, int num_rows, int num_cols) const {
    std::ifstream file(file_path);

    if (!file.is_open()) {
        std::cerr << "Failed to open file " << file_path << "." << std::endl;
    }

    std::string line;
    std::vector<cv::Mat> matrices;
    while (std::getline(file, line)) {
        auto trimmed_line = line.substr(line.find(':')+1, line.length());
        std::istringstream iss(trimmed_line);
        std::vector<double> values;
        double val;
        while(iss >> val) {
            values.push_back(val);
        }

        if (values.size() == (num_rows * num_cols)) {
            cv::Mat matrix = cv::Mat(num_rows, num_cols, CV_64F);
            for (int i = 0; i < num_rows; i++) {
                for (int j = 0; j < num_cols; j++) {
                    matrix.at<double>(i, j) = values[i * num_cols + j];
                }
            }

            matrices.push_back(matrix);
        } else {
            std::cerr << "Calibration matrix size should be (" << num_rows << "," << num_cols << ")." << std::endl;
        }
    }

    return matrices;
}