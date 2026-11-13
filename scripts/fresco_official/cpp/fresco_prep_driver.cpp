// 공식 FreSCo 전처리(Point-Cloud-Preprocessing-Tools d94040e)를 키프레임 스트림에 적용하는 드라이버.
//
// 공식 TU를 수정 없이 함께 링크한다(build.sh): 선택기의 extractPointCloud(점 -> row/col 구조화),
// BatchMultiBevGen.cpp의 getOrderedCloud -> markGroundPoints -> computeAndSaveSingleBev
// (공식 main과 같은 순서, interval 1.0). 실제 PCL 1.12 / OpenCV 4.5 라이브러리를 쓴다.
// 공식 main과 다른 점(평가 어댑터, protocol.json A3·G2):
//   * 선택기의 키프레임 선택·자세 파일·pcd 저장/재읽기를 하지 않는다. 받은 원시 파일을
//     작업 폴더에 공식 이름으로 쓰고 extractPointCloud(0)을 부른 뒤 메모리의 점군을
//     그대로 넘긴다(pcd는 등록 필드 전부를 무손실 왕복하므로 같은 입력).
//   * 단층 BEV는 공식 함수가 쓴 PNG를 다시 읽어 uint8 224x224로 내보낸다(CSV와 같은 값).
//   * HeLiPR VLP-16(FRESCO_FAMILY_HELIPR)은 공식 선택기가 없어 MulRan 선택기와 같은
//     형식의 적응 함수 extractPointCloudVlp16을 쓴다: row = 15 - ring, col = 1800분할,
//     센서 값 N_SCAN 16, Horizon_SCAN 1800, GROUND_UPPER_SCAN 7.
// 입출력: stdin [u32 n][n bytes 공식 원시 파일]... [u32 0]
//         stdout 기록마다 [224*224 uint8 BEV][i32 x4 통계]. 공식 함수의 cout 출력은 stderr로 보낸다.
#include "BatchMultiBevGen.h"

#include <sys/stat.h>
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

#if defined(FRESCO_FAMILY_KITTI)
extern std::string velo_cloud_dir_;
pcl::PointCloud<pcl::PointXYZIRCT>::Ptr extractPointCloud(int64_t timestamp);
static const char *RAW_NAME = "000000.bin";                 // fmt "{:06d}.bin"
#elif defined(FRESCO_FAMILY_MULRAN)
extern std::string ouster_cloud_dir_;
pcl::PointCloud<pcl::PointXYZIRCT>::Ptr extractPointCloud(int64_t timestamp);
static const char *RAW_NAME = "0000000000.bin";             // boost "%010ld"
#elif defined(FRESCO_FAMILY_OXFORD)
extern std::string velodyne_cloud_dir_;
pcl::PointCloud<pcl::PointXYZIRCT>::Ptr extractPointCloud(int64_t timestamp);
static const char *RAW_NAME = "0000000000.bin";             // boost "%010ld"
#elif defined(FRESCO_FAMILY_HELIPR)
static const char *RAW_NAME = "0000000000.bin";
// 적응(공식에 VLP-16 선택기 없음): MulranPointCloudSelect.cpp:96-133의 형식을 따른다.
// HeLiPR 원시 점 22바이트(x,y,z,intensity float32, ring uint16, time float32), ring 0 = -15도.
// 공식 규약은 row 0 = 맨 위 빔이므로 row = 15 - ring. 파일 크기로 점 수를 정해 모두 읽는다.
pcl::PointCloud<pcl::PointXYZIRCT>::Ptr extractPointCloudVlp16(const std::string &file_name)
{
    pcl::PointCloud<pcl::PointXYZIRCT>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZIRCT>);
    std::ifstream file(file_name, std::ios::in | std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open point cloud file: " << file_name << std::endl;
        return cloud;
    }
    const long n = static_cast<long>(file.tellg()) / 22;
    file.seekg(0);
    for (long k = 0; k < n; k++) {
        pcl::PointXYZIRCT point;
        std::uint16_t ring = 0;
        float t = 0;
        file.read(reinterpret_cast<char *>(&point.x), sizeof(float));
        file.read(reinterpret_cast<char *>(&point.y), sizeof(float));
        file.read(reinterpret_cast<char *>(&point.z), sizeof(float));
        file.read(reinterpret_cast<char *>(&point.intensity), sizeof(float));
        file.read(reinterpret_cast<char *>(&ring), sizeof(std::uint16_t));
        file.read(reinterpret_cast<char *>(&t), sizeof(float));
        point.row = static_cast<uint16_t>(15 - static_cast<int>(ring));
        float azimuthal_angle = atan2(point.y, point.x) / M_PI * 180.0f;
        if (azimuthal_angle > 360.0f) {azimuthal_angle = azimuthal_angle - 360.0f;}
        else if (azimuthal_angle < 0.0f) {azimuthal_angle = azimuthal_angle + 360.0f;}
        point.col = static_cast<uint16_t>(round(azimuthal_angle / 360.0f * 1800));
        point.label = -2;
        cloud->push_back(point);
    }
    return cloud;
}
#else
#error "define FRESCO_FAMILY_{KITTI,MULRAN,OXFORD,HELIPR}"
#endif

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
    std::cout.rdbuf(std::cerr.rdbuf());   // 공식 함수의 진단 출력이 stdout(이진)을 오염시키지 않게
    const std::string dir = std::string(argv[1]) + "/";
    mkdir((dir + "single_img").c_str(), 0755);
    mkdir((dir + "single_csv").c_str(), 0755);
    output_single_bvm_img_dir_ = dir + "single_img/";
    output_single_bvm_csv_dir_ = dir + "single_csv/";
    setNeighbors();
#if defined(FRESCO_FAMILY_KITTI)
    sensor_params_ = getSensorParams(SensorType::HDL_64E);
    velo_cloud_dir_ = dir;
#elif defined(FRESCO_FAMILY_MULRAN)
    sensor_params_ = getSensorParams(SensorType::OS1_64);
    ouster_cloud_dir_ = dir;
#elif defined(FRESCO_FAMILY_OXFORD)
    sensor_params_ = getSensorParams(SensorType::HDL_32E);
    velodyne_cloud_dir_ = dir;
#else
    sensor_params_.N_SCAN = 16;
    sensor_params_.Horizon_SCAN = 1800;
    sensor_params_.GROUND_UPPER_SCAN = 7;
    sensor_params_.HEIGHT_RES = 0.5f;   // 다층 BEV 전용, 단층 BEV에 쓰이지 않음
#endif
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
#if defined(FRESCO_FAMILY_HELIPR)
        pcl::PointCloud<pcl::PointXYZIRCT>::Ptr cloud = extractPointCloudVlp16(raw_path);
#else
        pcl::PointCloud<pcl::PointXYZIRCT>::Ptr cloud = extractPointCloud(0);
#endif
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
