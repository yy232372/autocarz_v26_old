#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/float32.hpp>
#include <autocarz/msg/waypoint_info.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#define MAX_BUFF 200

namespace
{
std::string trim(const std::string& value)
{
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::vector<std::string> splitCsvLine(const std::string& line)
{
    std::vector<std::string> tokens;
    std::stringstream stream(line);
    std::string token;
    while (std::getline(stream, token, ',')) {
        tokens.push_back(trim(token));
    }
    return tokens;
}

int findColumn(const std::vector<std::string>& header, const std::string& name)
{
    for (std::size_t i = 0; i < header.size(); ++i) {
        if (header[i] == name) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

int wrapIndex(int index, int count)
{
    if (count <= 0) {
        return -1;
    }
    index %= count;
    if (index < 0) {
        index += count;
    }
    return index;
}

std::string resolvePackagePath(const std::string& package_share, const std::string& value)
{
    if (value.empty() || value.front() == '/') {
        return value;
    }
    return package_share + "/" + value;
}
}  // namespace

rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cluster_pub_in;
rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr max_roi_pub;

int globalNum = 0;
int roiNum = 0;
int roiNum_start = 0;
bool is_livox = false;
int interval = 2;
int way_indx[MAX_BUFF] = {0};
float htMat[12] = {};
int lane = 0;
float z_threshold = 0.1F;
float max_roi_x = 0.0F;
double radius_dim = 0.0;

std::vector<float> roi_x;
std::vector<float> roi_y;
std::vector<float> roi_radius;
std::vector<int> path_to_roi;

bool checkReadPath = false;
bool checkOdom = false;
bool checkCurWaypointIndx = false;

bool readDrivingPathFile(const std::string& file, int& waypoint_count)
{
    std::ifstream fin(file);
    if (!fin.is_open()) {
        RCLCPP_ERROR(rclcpp::get_logger("roi_in_track"), "Could not open driving Path file: %s", file.c_str());
        return false;
    }

    waypoint_count = 0;
    std::string line;
    std::size_t line_number = 0;

    while (std::getline(fin, line)) {
        ++line_number;
        line = trim(line);

        if (line.empty() || line.front() == '#') {
            continue;
        }

        std::replace(line.begin(), line.end(), ',', ' ');
        std::stringstream stream(line);

        double x = 0.0;
        double y = 0.0;

        // x,y로 시작하지 않는 CSV header 등은 건너뛴다.
        if (!(stream >> x >> y)) {
            continue;
        }

        if (!std::isfinite(x) || !std::isfinite(y)) {
            RCLCPP_ERROR(
                rclcpp::get_logger("roi_in_track"),
                "Invalid driving Path coordinate at line %zu: %s",
                line_number,
                file.c_str());
            return false;
        }

        ++waypoint_count;
    }

    if (waypoint_count < 2) {
        RCLCPP_ERROR(
            rclcpp::get_logger("roi_in_track"),
            "Driving Path contains fewer than 2 valid waypoints: %s",
            file.c_str());
        return false;
    }

    RCLCPP_INFO(
        rclcpp::get_logger("roi_in_track"),
        "[roi_in_track.cpp] loaded %d driving Path waypoints: %s",
        waypoint_count,
        file.c_str());

    return true;
}

bool readDenseRoiCsv(const std::string& file)
{
    std::ifstream fin(file);
    if (!fin.is_open()) {
        RCLCPP_ERROR(rclcpp::get_logger("roi_in_track"), "Could not open Dense ROI CSV: %s", file.c_str());
        return false;
    }

    std::string line;
    if (!std::getline(fin, line)) {
        RCLCPP_ERROR(rclcpp::get_logger("roi_in_track"), "Dense ROI CSV is empty: %s", file.c_str());
        return false;
    }

    const auto header = splitCsvLine(line);
    const int x_col = findColumn(header, "x");
    const int y_col = findColumn(header, "y");
    const int radius_col = findColumn(header, "roi_radius");

    if (x_col < 0 || y_col < 0 || radius_col < 0) {
        RCLCPP_ERROR(rclcpp::get_logger("roi_in_track"), "Dense ROI CSV requires x,y,roi_radius: %s", file.c_str());
        return false;
    }

    roi_x.clear();
    roi_y.clear();
    roi_radius.clear();

    const int required_col = std::max({x_col, y_col, radius_col});
    std::size_t line_number = 1;

    while (std::getline(fin, line)) {
        ++line_number;
        line = trim(line);
        if (line.empty()) {
            continue;
        }

        const auto tokens = splitCsvLine(line);
        if (static_cast<int>(tokens.size()) <= required_col) {
            RCLCPP_ERROR(rclcpp::get_logger("roi_in_track"), "Invalid Dense ROI row at line %zu", line_number);
            return false;
        }

        try {
            roi_x.push_back(std::stof(tokens[x_col]));
            roi_y.push_back(std::stof(tokens[y_col]));
            roi_radius.push_back(std::stof(tokens[radius_col]));
        } catch (const std::exception& exc) {
            RCLCPP_ERROR(rclcpp::get_logger("roi_in_track"), "Dense ROI parse error at line %zu: %s", line_number, exc.what());
            return false;
        }
    }

    if (roi_x.empty() || roi_x.size() != roi_y.size() || roi_x.size() != roi_radius.size()) {
        RCLCPP_ERROR(rclcpp::get_logger("roi_in_track"), "Dense ROI CSV contains no valid rows: %s", file.c_str());
        return false;
    }

    RCLCPP_INFO(rclcpp::get_logger("roi_in_track"), "[roi_in_track.cpp] loaded %zu Dense ROI points", roi_x.size());
    return true;
}

bool readMappingCsv(const std::string& file)
{
    std::ifstream fin(file);
    if (!fin.is_open()) {
        RCLCPP_ERROR(rclcpp::get_logger("roi_in_track"), "Could not open Path->ROI mapping CSV: %s", file.c_str());
        return false;
    }

    std::string line;
    if (!std::getline(fin, line)) {
        RCLCPP_ERROR(rclcpp::get_logger("roi_in_track"), "Mapping CSV is empty: %s", file.c_str());
        return false;
    }

    const auto header = splitCsvLine(line);
    const int path_col = findColumn(header, "path_index");
    const int roi_col = findColumn(header, "roi_index");

    if (path_col < 0 || roi_col < 0) {
        RCLCPP_ERROR(rclcpp::get_logger("roi_in_track"), "Mapping CSV requires path_index,roi_index: %s", file.c_str());
        return false;
    }

    std::vector<std::pair<int, int>> rows;
    int max_path_index = -1;
    const int required_col = std::max(path_col, roi_col);
    std::size_t line_number = 1;

    while (std::getline(fin, line)) {
        ++line_number;
        line = trim(line);
        if (line.empty()) {
            continue;
        }

        const auto tokens = splitCsvLine(line);
        if (static_cast<int>(tokens.size()) <= required_col) {
            RCLCPP_ERROR(rclcpp::get_logger("roi_in_track"), "Invalid mapping row at line %zu", line_number);
            return false;
        }

        try {
            const int path_index = std::stoi(tokens[path_col]);
            const int roi_index = std::stoi(tokens[roi_col]);

            if (path_index < 0) {
                throw std::runtime_error("negative path_index");
            }
            if (roi_index < 0 || roi_index >= static_cast<int>(roi_x.size())) {
                throw std::runtime_error("roi_index out of Dense ROI range");
            }

            rows.emplace_back(path_index, roi_index);
            max_path_index = std::max(max_path_index, path_index);
        } catch (const std::exception& exc) {
            RCLCPP_ERROR(rclcpp::get_logger("roi_in_track"), "Mapping parse/range error at line %zu: %s", line_number, exc.what());
            return false;
        }
    }

    if (max_path_index < 0) {
        RCLCPP_ERROR(rclcpp::get_logger("roi_in_track"), "Mapping CSV contains no valid rows: %s", file.c_str());
        return false;
    }

    path_to_roi.assign(static_cast<std::size_t>(max_path_index + 1), -1);

    for (const auto& row : rows) {
        if (path_to_roi[static_cast<std::size_t>(row.first)] != -1) {
            RCLCPP_ERROR(rclcpp::get_logger("roi_in_track"), "Duplicated path_index=%d in mapping", row.first);
            return false;
        }
        path_to_roi[static_cast<std::size_t>(row.first)] = row.second;
    }

    for (std::size_t i = 0; i < path_to_roi.size(); ++i) {
        if (path_to_roi[i] < 0) {
            RCLCPP_ERROR(rclcpp::get_logger("roi_in_track"), "Missing path_index=%zu in mapping", i);
            return false;
        }
    }

    globalNum = static_cast<int>(path_to_roi.size());
    RCLCPP_INFO(rclcpp::get_logger("roi_in_track"), "[roi_in_track.cpp] loaded %d Path->ROI mappings", globalNum);
    return true;
}

bool appendMappedIndex(int& output_count, int path_index)
{
    if (output_count >= MAX_BUFF - 1) {
        RCLCPP_WARN(rclcpp::get_logger("roi_in_track"), "ROI waypoint buffer reached MAX_BUFF=%d", MAX_BUFF);
        return false;
    }

    const int wrapped_path_index = wrapIndex(path_index, globalNum);
    if (wrapped_path_index < 0 || wrapped_path_index >= globalNum) {
        return false;
    }

    const int mapped_roi_index = path_to_roi[static_cast<std::size_t>(wrapped_path_index)];
    if (mapped_roi_index < 0 || mapped_roi_index >= static_cast<int>(roi_x.size())) {
        RCLCPP_ERROR(rclcpp::get_logger("roi_in_track"), "Invalid ROI mapping: path=%d roi=%d", wrapped_path_index, mapped_roi_index);
        return false;
    }

    way_indx[output_count++] = mapped_roi_index;
    return true;
}

void pc_cb(const sensor_msgs::msg::PointCloud2::SharedPtr input_pcl_raw)
{
    if (!(checkReadPath && checkCurWaypointIndx)) {
        return;
    }

    pcl::PCLPointCloud2 input_pcl;
    pcl::PCLPointCloud2 output_pcl;
    pcl::PointCloud<pcl::PointXYZI> xyz_array;
    pcl::PointCloud<pcl::PointXYZI> xyz_array_output;
    pcl::PointXYZI transformed;

    pcl_conversions::toPCL(*input_pcl_raw, input_pcl);
    pcl::fromPCLPointCloud2(input_pcl, xyz_array);

    for (const auto& point : xyz_array.points) {
        if (point.z >= z_threshold) {
            continue;
        }
        if (!(std::fabs(point.y) > 0.5 || std::fabs(point.x) > 0.7)) {
            continue;
        }

        transformed.x = htMat[0] * point.x + htMat[1] * point.y + htMat[2] * point.z + htMat[3];
        transformed.y = htMat[4] * point.x + htMat[5] * point.y + htMat[6] * point.z + htMat[7];
        transformed.z = htMat[8] * point.x + htMat[9] * point.y + htMat[10] * point.z + htMat[11];

        for (int l = 0; l < MAX_BUFF && way_indx[l] != -100; ++l) {
            const int roi_index = way_indx[l];
            if (roi_index < 0 || roi_index >= static_cast<int>(roi_x.size())) {
                continue;
            }

            const double effective_radius = roi_radius[roi_index] - radius_dim;
            const double rsquare = effective_radius * effective_radius;
            const double dx = roi_x[roi_index] - transformed.x;
            const double dy = roi_y[roi_index] - transformed.y;

            if (dx * dx + dy * dy <= rsquare) {
                xyz_array_output.push_back(point);
                max_roi_x = std::max(max_roi_x, point.x);
                break;
            }
        }
    }

    sensor_msgs::msg::PointCloud2 output_pcl_complete;
    pcl::toPCLPointCloud2(xyz_array_output, output_pcl);
    pcl_conversions::fromPCL(output_pcl, output_pcl_complete);
    output_pcl_complete.header.frame_id = "velodyne";
    cluster_pub_in->publish(output_pcl_complete);

    if (is_livox) {
        std_msgs::msg::Float32 max_roi_dis;
        max_roi_dis.data = max_roi_x;
        max_roi_pub->publish(max_roi_dis);
    }

    checkOdom = false;
    checkCurWaypointIndx = false;
    max_roi_x = 0.0F;
}

void odom_cb(const nav_msgs::msg::Odometry::SharedPtr odometry)
{
    if (!checkReadPath) {
        return;
    }

    const float qx = odometry->pose.pose.orientation.x;
    const float qy = odometry->pose.pose.orientation.y;
    const float qz = odometry->pose.pose.orientation.z;
    const float qw = odometry->pose.pose.orientation.w;

    htMat[0] = 1.0F - 2.0F * qy * qy - 2.0F * qz * qz;
    htMat[1] = 2.0F * qx * qy - 2.0F * qw * qz;
    htMat[2] = 2.0F * qx * qz + 2.0F * qw * qy;
    htMat[3] = odometry->pose.pose.position.x;
    htMat[4] = 2.0F * qx * qy + 2.0F * qw * qz;
    htMat[5] = 1.0F - 2.0F * qx * qx - 2.0F * qz * qz;
    htMat[6] = 2.0F * qy * qz - 2.0F * qw * qx;
    htMat[7] = odometry->pose.pose.position.y;
    htMat[8] = 2.0F * qx * qz - 2.0F * qw * qy;
    htMat[9] = 2.0F * qy * qz + 2.0F * qw * qx;
    htMat[10] = 1.0F - 2.0F * qx * qx - 2.0F * qy * qy;
    htMat[11] = odometry->pose.pose.position.z;

    checkOdom = true;
}

void waypointInfo_cb(const autocarz::msg::WaypointInfo::SharedPtr info)
{
    if (!checkReadPath || globalNum <= 0) {
        return;
    }

    lane = info->lane;
    const int cur_way = wrapIndex(info->indx_in, globalNum);
    int output_count = 0;

    if (info->lane == 1 || is_livox) {
        int path_index = cur_way + roiNum_start;
        const int sample_count = roiNum / interval + 1;

        for (int i = 0; i < sample_count; ++i) {
            if (!appendMappedIndex(output_count, path_index)) {
                break;
            }
            path_index += interval;
        }
    } else {
        int path_index = cur_way - roiNum;
        const int sample_count = 2 * roiNum / interval + 1;

        for (int i = 0; i < sample_count; ++i) {
            if (!appendMappedIndex(output_count, path_index)) {
                break;
            }
            path_index += interval;
        }
    }

    way_indx[output_count] = -100;
    checkCurWaypointIndx = output_count > 0;
}

class RoiNode : public rclcpp::Node
{
public:
    RoiNode()
        : Node("roi_in_track")
    {
        const auto package_share = ament_index_cpp::get_package_share_directory("autocarz");

        const auto path_value = this->declare_parameter<std::string>("driving_path", "");
        const auto roi_value = this->declare_parameter<std::string>("roi_in", "");
        const auto mapping_value = this->declare_parameter<std::string>("mapping_in", "");

        const std::string path_file = resolvePackagePath(package_share, path_value);
        const std::string roi_file = resolvePackagePath(package_share, roi_value);
        const std::string mapping_file = resolvePackagePath(package_share, mapping_value);

        const auto pointcloud_topic = this->declare_parameter<std::string>("topics.pointcloud", "");
        const auto odom_topic = this->declare_parameter<std::string>("topics.odom", "/odom");
        const auto waypoint_topic = this->declare_parameter<std::string>("topics.waypoint_info", "/waypointInfo");
        const auto roi_output_topic = this->declare_parameter<std::string>("topics.roi", "roi_in");
        const auto max_roi_topic = this->declare_parameter<std::string>("topics.max_roi", "max_roi_dis_in");

        roiNum = this->declare_parameter<int>("roi_idxlen", 0);
        roiNum_start = this->declare_parameter<int>("roi_idx_start", 0);
        is_livox = this->declare_parameter<bool>("is_livox", false);
        radius_dim = this->declare_parameter<double>("radius_dim", 0.0);
        interval = this->declare_parameter<int>("interval", 2);

        if (interval <= 0) {
            throw std::runtime_error("interval must be greater than 0");
        }

        if (is_livox) {
            z_threshold = 0.4F;
        }

        int driving_path_count = 0;
        const bool path_ok = readDrivingPathFile(path_file, driving_path_count);
        const bool roi_ok = path_ok && readDenseRoiCsv(roi_file);
        const bool mapping_ok = roi_ok && readMappingCsv(mapping_file);
        const bool count_ok = mapping_ok && driving_path_count == globalNum;

        if (mapping_ok && !count_ok) {
            RCLCPP_ERROR(
                this->get_logger(),
                "Driving Path/mapping count mismatch: path=%d mapping=%d",
                driving_path_count,
                globalNum);
        }

        checkReadPath = path_ok && roi_ok && mapping_ok && count_ok;

        if (!checkReadPath) {
            RCLCPP_ERROR(this->get_logger(), "IN ROI/mapping initialization failed.");
        }

        const auto control_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile();

        pc_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            pointcloud_topic, rclcpp::SensorDataQoS(), pc_cb);
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            odom_topic, control_qos, odom_cb);
        waypoint_sub_ = this->create_subscription<autocarz::msg::WaypointInfo>(
            waypoint_topic, control_qos, waypointInfo_cb);

        cluster_pub_in = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            roi_output_topic, rclcpp::SensorDataQoS());

        if (is_livox) {
            max_roi_pub = this->create_publisher<std_msgs::msg::Float32>(
                max_roi_topic, control_qos);
        }
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<autocarz::msg::WaypointInfo>::SharedPtr waypoint_sub_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<RoiNode>());
    rclcpp::shutdown();
    return 0;
}
