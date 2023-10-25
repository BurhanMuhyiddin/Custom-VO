#ifndef PARSE_PARAMS_HPP
#define PARSE_PARAMS_HPP

#include <iostream>
#include <fstream>
#include <json.hpp>

using json = nlohmann::json;

class ParameterParser {
public:
    static ParameterParser& GetInstance() {
        static ParameterParser instance;
        return instance;
    }

    json GetParameters() const {
        return parameters_;
    }

    void ReadJsonFile(const std::string &param_file_path) {
        std::ifstream file_stream(param_file_path);
        if (file_stream.is_open()) {
            try {
                file_stream >> parameters_;
                file_stream.close();
                std::cout << "JSON file read successfully." << std::endl;
            } catch (const std::exception &e) {
                std::cerr << "Error parsing JSON: " << e.what() << std::endl;
            }
        } else {
            std::cerr << "Failed to open JSON file: " << param_file_path << std::endl;
        }
    }

private:
    ParameterParser() {}
    ParameterParser(ParameterParser const&) = delete;
    void operator=(ParameterParser const&) = delete;
private:
    json parameters_;
};

#endif