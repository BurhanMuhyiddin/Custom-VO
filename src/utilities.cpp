#include <utilities.hpp>

std::string to_uppercase(const std::string &inp_string) {
    std::string uppercase_string(inp_string.size(), ' ');
    std::transform(inp_string.begin(), inp_string.end(), uppercase_string.begin(), ::toupper);
    
    return uppercase_string;
}

double get_mae(const std::vector<cv::Mat> &ground_truth, const std::vector<cv::Mat> &estimated) {
    double ae = 0.0;
    for (int i = 0; i < estimated.size(); i++) {
        double x_err = ground_truth[i].at<double>(0, 3) - estimated[i].at<double>(0, 3);
        double y_err = ground_truth[i].at<double>(1, 3) - estimated[i].at<double>(1, 3);
        double z_err = ground_truth[i].at<double>(2, 3) - estimated[i].at<double>(2, 3);
        double dist = sqrt(x_err * x_err + y_err * y_err + z_err * z_err);
        ae += dist;
    }
    double mae = ae / estimated.size();

    return mae;
}

double get_mse(const std::vector<cv::Mat> &ground_truth, const std::vector<cv::Mat> &estimated) {
    double se = 0.0;
    for (int i = 0; i < estimated.size(); i++) {
        double x_err = ground_truth[i].at<double>(0, 3) - estimated[i].at<double>(0, 3);
        double y_err = ground_truth[i].at<double>(1, 3) - estimated[i].at<double>(1, 3);
        double z_err = ground_truth[i].at<double>(2, 3) - estimated[i].at<double>(2, 3);
        double dist = sqrt(x_err * x_err + y_err * y_err + z_err * z_err);
        se += dist * dist;
    }
    double mse = se / estimated.size();

    return mse;
}

void save_trajectory(const std::vector<cv::Mat> &trajectory, const std::string &file_path) {
    std::ofstream out_file(file_path);

    if (!out_file.is_open()) {
        std::cerr << "Error opening the file: " << file_path << std::endl;
        return;
    }

    for (const cv::Mat& matrix : trajectory) {
        // Ensure the matrix is of size 3x4
        if (matrix.rows == 3 && matrix.cols == 4) {
            // Reshape the matrix to a row vector and write to the file
            cv::Mat rowVector = matrix.reshape(1, 1);
            for (int i = 0; i < rowVector.cols; ++i) {
                out_file << rowVector.at<double>(0, i);

                // Add a comma after each element (except the last one)
                if (i < rowVector.cols - 1) {
                    out_file << ",";
                }
            }
            // Add a newline after each row
            out_file << "\n";
        } else {
            std::cerr << "Matrix size is not 3x4. Skipping.\n";
        }
    }

    // Close the file
    out_file.close();
}