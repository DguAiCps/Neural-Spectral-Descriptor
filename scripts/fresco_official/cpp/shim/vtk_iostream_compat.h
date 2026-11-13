// 환경 shim(코드 수정 아님): 공식 선택기 TU(Kitti/Mulran/OxfordPointCloudSelect.cpp)는
// `ifstream`, `ios`를 std:: 없이 쓴다. 저자 빌드의 PCL->VTK(<=8) vtkIOStream.h가 하던
// using 선언을 강제 포함(-include)으로 재현한다. VTK 9.1 헤더에는 이 선언이 없어
// 무수정 컴파일이 실패한다(results/pdv2_fresco_official_20261107/pipeline/build.json).
// 수학 함수(atan2/sqrt/abs/round)의 오버로드 선택에는 영향을 주지 않는다.
#pragma once
#include <fstream>
#include <iostream>
using std::ifstream;
using std::ios;
