#ifndef DATASET_HANDLER_HPP
#define DATASET_HANDLER_HPP

#include <string>
#include <filesystem>
#include <fstream>

#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

class ImageLoader {
public:
    ImageLoader(const std::vector<std::string> &image_paths, int imread_mode=0)
        : image_paths_(image_paths), imread_mode_(imread_mode) {}

    class iterator : public std::iterator<std::input_iterator_tag, cv::Mat> {
        public:
            iterator(std::vector<std::string>::iterator it, ImageLoader* image_loader_ptr)
                : it_(it), image_loader_(image_loader_ptr) {}
            
            bool operator!=(const iterator &other) const {
                return it_ != other.it_;
            }

            cv::Mat operator*() const {
                return cv::imread(*it_, image_loader_->imread_mode_);
            }

            iterator& operator++() {
                ++it_;
                return *this;
            }

        private:
            std::vector<std::string>::iterator it_;
            ImageLoader* image_loader_;
    };

    iterator begin() {
        return iterator(image_paths_.begin(), this);
    }

    iterator end() {
        return iterator(image_paths_.end(), this);
    }

private:
    int imread_mode_;
    std::vector<std::string> image_paths_;
};

class DatasetHandler {
public:
    DatasetHandler(const std::string &data_path, const std::string &sequence="00", bool lidar=false, int imread_mode=0);

public:
    fs::path seq_dir_;
    fs::path poses_dir_;

    int num_frames_;
    int im_height_;
    int im_width_;

    std::unique_ptr<ImageLoader> left_image_loader_;
    std::unique_ptr<ImageLoader> right_image_loader_;
    std::unique_ptr<ImageLoader::iterator> left_image_loader_it_;
    std::unique_ptr<ImageLoader::iterator> right_image_loader_it_;

    cv::Mat P0_;
    cv::Mat P1_;
    cv::Mat P2_;
    cv::Mat P3_;
    cv::Mat Tr_;
    cv::Mat time_sequence_;
    std::vector<cv::Mat> gt_data_;

private:
    std::vector<std::string> GetImgPathInDirectory(const std::string& directory_path) const;

    std::vector<cv::Mat> ReadTxtToMat(const std::string &file_path, int num_rows, int num_cols) const;
};

#endif // DATASET_HANDLER_HPP