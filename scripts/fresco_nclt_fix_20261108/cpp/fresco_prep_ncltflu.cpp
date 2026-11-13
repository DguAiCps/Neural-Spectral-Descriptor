// NCLT 좌표 적응 수정(G3') 전처리 드라이버 (2026-11-08). 기존 scripts/fresco_official/cpp/fresco_prep_driver.cpp의
// Oxford 경로와 같은 순서·입출력이고, 다른 점은 row 재계산 한 가지다.
//
// 입력 파일 = codec_fix.nclt_flu_official: Oxford 배치의 (-x', y', -z', 0), (x', y', z') = body 축 정렬 라이다 원점 FLU.
// 1) 공식 OxfordPointCloudSelect.cpp extractPointCloud(무수정 TU, build_fix.sh)가 읽고 x=-x, z=-z 하여
//    점 = (x', y', z'), col = round(az/360*1056), label = -2, row = round((-el+10.67)/1.3335) clamp를 만든다.
// 2) 검사: 공식 식을 같은 float 연산으로 다시 계산해(offset 10.67) 공식 row와 모든 점에서 같은지 확인한다.
//    하나라도 다르면 중단(rc 5). 그래서 3)의 앙각 식이 공식 식과 같은 값을 쓴다는 것이 매 프레임 보장된다.
// 3) 적응: row = round((-el + 30.67)/1.3335) clamp[0,31]. 거꾸로 장착된 HDL-32E의 물리 앙각 범위
//    (+30.67..-10.67도 = -(laser_angles.csv))를 공식 규약(row 0 = 물리 맨 위 빔, top-down)으로 놓는다.
//    간격 1.3335와 clamp는 공식 Oxford 값 그대로다.
// 이후 getOrderedCloud -> markGroundPoints -> computeAndSaveSingleBev(interval 1.0)는 공식 TU 무수정,
// sensor_params HDL_32E(32, 1056, GROUND_UPPER_SCAN 20), +2.0 m, *4, label0 제외 모두 공식 값.
// 입출력: stdin [u32 n][n bytes]... [u32 0], stdout 기록마다 [224*224 uint8 BEV][i32 x4 통계].
#include "BatchMultiBevGen.h"

#include <sys/stat.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

extern std::vector<std::pair<int, int>> four_neighbor_iterator_;
extern std::string output_single_bvm_img_dir_;
extern std::string output_single_bvm_csv_dir_;
extern SensorParams sensor_params_;
void setNeighbors();
void getOrderedCloud(pcl::PointCloud<pcl::PointXYZIRCT>::Ptr &input_cloud,
                     pcl::PointCloud<pcl::PointXYZIRCT>::Ptr &output_cloud);
void markGroundPoints(pcl::PointCloud<pcl::PointXYZIRCT>::Ptr &output_cloud, cv::Mat &ground_mat);
void computeAndSaveSingleBev(pcl::PointCloud<pcl::PointXYZIRCT>::Ptr cloud,
                             std::string str_cloud_idx, float interval);
extern std::string velodyne_cloud_dir_;
pcl::PointCloud<pcl::PointXYZIRCT>::Ptr extractPointCloud(int64_t timestamp);
static const char *RAW_NAME = "0000000000.bin";             // boost "%010ld"

static const double OFFICIAL_ROW_OFFSET = 10.67;   // OxfordPointCloudSelect.cpp:206
static const double NCLT_ROW_OFFSET = 30.67;       // -(min laser_angles.csv) = 물리 최고 앙각

// OxfordPointCloudSelect.cpp:205-207과 같은 식(float 점 좌표, float 앙각)
static int rowFromElevation(const pcl::PointXYZIRCT &point, double offset)
{
    float elevation_angle = std::atan2(point.z, std::sqrt(point.x * point.x + point.y * point.y)) / M_PI * 180.0f;
    int row_idx = round((-elevation_angle + offset) / 1.3335);
    row_idx = std::min(31, std::max(0, row_idx));
    return row_idx;
}

static bool read_exact(void *buf, size_t n)
{
    return std::fread(buf, 1, n, stdin) == n;
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <work_dir>\n", argv[0]);
        return 2;
    }
    std::cout.rdbuf(std::cerr.rdbuf());
    const std::string dir = std::string(argv[1]) + "/";
    mkdir((dir + "single_img").c_str(), 0755);
    mkdir((dir + "single_csv").c_str(), 0755);
    output_single_bvm_img_dir_ = dir + "single_img/";
    output_single_bvm_csv_dir_ = dir + "single_csv/";
    setNeighbors();
    sensor_params_ = getSensorParams(SensorType::HDL_32E);
    velodyne_cloud_dir_ = dir;
    const std::string raw_path = dir + RAW_NAME;
    std::vector<char> buf;
    for (;;) {
        std::uint32_t n = 0;
        if (!read_exact(&n, 4)) {
            std::fprintf(stderr, "FRESCO_PREP_ERROR truncated header\n");
            return 3;
        }
        if (n == 0) break;
        buf.resize(n);
        if (!read_exact(buf.data(), n)) {
            std::fprintf(stderr, "FRESCO_PREP_ERROR truncated record\n");
            return 3;
        }
        {
            std::ofstream f(raw_path, std::ios::out | std::ios::binary | std::ios::trunc);
            f.write(buf.data(), n);
        }
        pcl::PointCloud<pcl::PointXYZIRCT>::Ptr cloud = extractPointCloud(0);
        if (cloud->points.size() * 16 != n) {
            std::fprintf(stderr, "FRESCO_PREP_ERROR point count %zu bytes %u\n", cloud->points.size(), n);
            return 5;
        }
        long mismatch = 0;
        for (auto &point : cloud->points) {
            if (rowFromElevation(point, OFFICIAL_ROW_OFFSET) != static_cast<int>(point.row)) mismatch++;
            if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) mismatch++;
        }
        if (mismatch) {
            std::fprintf(stderr, "FRESCO_PREP_ERROR official row recompute mismatch %ld\n", mismatch);
            return 5;
        }
        for (auto &point : cloud->points) {
            point.row = static_cast<uint16_t>(rowFromElevation(point, NCLT_ROW_OFFSET));
        }
        pcl::PointCloud<pcl::PointXYZIRCT>::Ptr ordered(new pcl::PointCloud<pcl::PointXYZIRCT>());
        cv::Mat ground_mat;
        getOrderedCloud(cloud, ordered);
        markGroundPoints(ordered, ground_mat);
        computeAndSaveSingleBev(ordered, "bev", 1.0f);
        cv::Mat img = cv::imread(output_single_bvm_img_dir_ + "bev.png", cv::IMREAD_UNCHANGED);
        if (img.empty() || img.rows != 224 || img.cols != 224 || img.type() != CV_8UC1 ||
            !img.isContinuous()) {
            std::fprintf(stderr, "FRESCO_PREP_ERROR bad bev image\n");
            return 4;
        }
        std::int32_t stats[4] = {static_cast<std::int32_t>(cloud->points.size()),
                                 static_cast<std::int32_t>(ordered->points.size()), 0, 0};
        for (auto &p : ordered->points) stats[2] += (p.label == 0);
        stats[3] = cv::countNonZero(img);
        std::fwrite(img.data, 1, 224 * 224, stdout);
        std::fwrite(stats, sizeof(std::int32_t), 4, stdout);
        std::fflush(stdout);
    }
    return 0;
}
