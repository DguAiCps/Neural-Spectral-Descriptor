#!/usr/bin/env bash
# 공식 FreSCo 전처리 빌드(서버 호스트, 사용자공간). sudo·전역 패키지 변경 없음.
#   bash scripts/fresco_official/cpp/build.sh
# 헤더·라이브러리: Ubuntu 22.04 저장소의 PCL 1.12.1 / OpenCV 4.5.4 / VTK 9.1 / Eigen 3.4 / Boost 1.74
# deb를 apt-get download 후 dpkg -x로 NAS env/{hdr,rt}에 풀었다(설치하지 않음).
# 플래그는 공식 CMakeLists.txt Release(-O3 -Wall -g -ggdb -DDEBUG, C++14 GNU, OpenMP)와 같다.
# fmt는 공식 서브모듈(80f8d34)을 헤더 전용으로 쓴다.
# KITTI 계열 드라이버는 공식 KittiPointCloudSelect.cpp에 고정 두 줄 회귀 수정(1974440 이전 빈 칸 의미,
# kitti_groundfix.py가 원본·결과 SHA256 확인)을 적용한 사본으로 빌드한다. 공식 원본은 고치지 않고,
# 무수정 공식 실행 파일(official/kitti_point_cloud_select)은 대조용으로 그대로 만든다.
# 기존 빌드를 덮지 않는다. 새로 빌드할 때는 빈 FRESCO_NAS_ROOT를 준다(헤더·런타임 env/는 그 아래에 둔다).
# 산출: $NAS/build/{official/*, kitti_groundfix/*, fresco_prep_*, build.json}
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
NAS=${FRESCO_NAS_ROOT:-/mnt/aicps-nas/qkrrlans00/nsd_fresco_official_20261107}
E=$REPO/_external/fresco_official_20261107/Point-Cloud-Preprocessing-Tools
H=$NAS/env/hdr/usr/include
L=$NAS/env/rt/usr/lib/x86_64-linux-gnu
SHIM=$REPO/scripts/fresco_official/cpp/shim
OUT=$NAS/build
[ ! -e "$OUT/build.json" ] || { echo "기존 빌드 $OUT 를 덮지 않는다" >&2; exit 1; }
mkdir -p "$OUT/obj" "$OUT/official"
[ "$(git -C "$E" rev-parse HEAD)" = d94040ed626c7275b6f428deddde2ddca74419d7 ]
[ -z "$(git -C "$E" status --porcelain --untracked-files=no)" ]

CXX=(g++ -std=gnu++14 -O3 -Wall -g -ggdb -DDEBUG -fopenmp -DFMT_HEADER_ONLY
     -I"$E/include" -I"$E" -I"$E/ThirdParties/fmt/include" -I"$H" -I"$H/pcl-1.12"
     -I"$H/eigen3" -I"$H/opencv4" -I"$H/vtk-9.1")
L2=$NAS/env/rt/usr/lib
LIBS=("$L/libpcl_io.so.1.12" "$L/libpcl_common.so.1.12" "$L/libopencv_imgcodecs.so.4.5d"
      "$L/libopencv_core.so.4.5d" "$L/libvtkCommonCore-9.1.so.1" "$L/libvtksys-9.1.so.1"
      -Wl,--disable-new-dtags -Wl,-rpath,"$L:$L2" -Wl,-rpath-link,"$L:$L2" -fopenmp)
COMPAT=(-include "$SHIM/vtk_iostream_compat.h")
cc() { echo "+ ${*: -1}" >&2; "${CXX[@]}" -w "$@"; }

# 공식 실행 파일(대조용, 무수정; 실행 시 출력 폴더를 rm -rf 하므로 NAS 고정 폴더만 준다)
cc -c "$E/src/Utility.cpp" -o "$OUT/obj/Utility.o"
cc -c "$E/BatchMultiBevGen.cpp" -o "$OUT/obj/BatchMultiBevGen_main.o"
cc "${COMPAT[@]}" -c "$E/KittiPointCloudSelect.cpp" -o "$OUT/obj/Kitti_main.o"
cc "${COMPAT[@]}" -c "$E/MulranPointCloudSelect.cpp" -o "$OUT/obj/Mulran_main.o"
cc "${COMPAT[@]}" -c "$E/OxfordPointCloudSelect.cpp" -o "$OUT/obj/Oxford_main.o"
g++ -o "$OUT/official/batch_multi_bev_gen" "$OUT/obj/BatchMultiBevGen_main.o" "$OUT/obj/Utility.o" "${LIBS[@]}"
g++ -o "$OUT/official/kitti_point_cloud_select" "$OUT/obj/Kitti_main.o" "${LIBS[@]}"
g++ -o "$OUT/official/mulran_point_cloud_select" "$OUT/obj/Mulran_main.o" "$OUT/obj/Utility.o" "${LIBS[@]}"
g++ -o "$OUT/official/oxford_point_cloud_select" "$OUT/obj/Oxford_main.o" "${LIBS[@]}"

# KITTI 계열 회귀 수정 사본(공식 원본 무수정). verify_prep.py의 KITTI 공식 체인 대조용 선택기도 같은 사본으로 만든다.
GF=$OUT/kitti_groundfix
python3 "$REPO/scripts/fresco_official/cpp/kitti_groundfix.py" "$E/KittiPointCloudSelect.cpp" "$GF/KittiPointCloudSelect.cpp"
cc "${COMPAT[@]}" -c "$GF/KittiPointCloudSelect.cpp" -o "$OUT/obj/Kitti_groundfix_main.o"
g++ -o "$OUT/official/kitti_point_cloud_select_groundfix" "$OUT/obj/Kitti_groundfix_main.o" "${LIBS[@]}"

# 드라이버용: 같은 공식 TU를 기호 이름만 바꿔(-D) 한 실행 파일에 링크한다(본문 무수정).
RB=(-Dmain=fresco_official_bevgen_main -DinitDirectories=fresco_official_bevgen_initDirectories)
cc "${RB[@]}" -c "$E/BatchMultiBevGen.cpp" -o "$OUT/obj/BatchMultiBevGen.o"
cc "${COMPAT[@]}" -Dmain=fresco_official_kitti_main -DinitDirectories=fresco_official_kitti_initDirectories \
   -DPose6f=FrescoKittiPose6f -c "$GF/KittiPointCloudSelect.cpp" -o "$OUT/obj/Kitti.o"
cc "${COMPAT[@]}" -Dmain=fresco_official_mulran_main -DinitDirectories=fresco_official_mulran_initDirectories \
   -c "$E/MulranPointCloudSelect.cpp" -o "$OUT/obj/Mulran.o"
cc "${COMPAT[@]}" -Dmain=fresco_official_oxford_main -DinitDirectories=fresco_official_oxford_initDirectories \
   -DPose6f=FrescoOxfordPose6f -c "$E/OxfordPointCloudSelect.cpp" -o "$OUT/obj/Oxford.o"
D=$REPO/scripts/fresco_official/cpp/fresco_prep_driver.cpp
for fam in kitti mulran oxford helipr; do
  F=$(echo $fam | tr a-z A-Z)
  cc -DFRESCO_FAMILY_$F -c "$D" -o "$OUT/obj/driver_$fam.o"
done
g++ -o "$OUT/fresco_prep_kitti" "$OUT/obj/driver_kitti.o" "$OUT/obj/Kitti.o" "$OUT/obj/BatchMultiBevGen.o" "$OUT/obj/Utility.o" "${LIBS[@]}"
g++ -o "$OUT/fresco_prep_mulran" "$OUT/obj/driver_mulran.o" "$OUT/obj/Mulran.o" "$OUT/obj/BatchMultiBevGen.o" "$OUT/obj/Utility.o" "${LIBS[@]}"
g++ -o "$OUT/fresco_prep_oxford" "$OUT/obj/driver_oxford.o" "$OUT/obj/Oxford.o" "$OUT/obj/BatchMultiBevGen.o" "$OUT/obj/Utility.o" "${LIBS[@]}"
g++ -o "$OUT/fresco_prep_helipr" "$OUT/obj/driver_helipr.o" "$OUT/obj/BatchMultiBevGen.o" "$OUT/obj/Utility.o" "${LIBS[@]}"

python3 - "$REPO" "$NAS" "$E" "$OUT" <<'EOF'
import hashlib, json, subprocess, sys, time
from pathlib import Path
repo, nas, e, out = map(Path, sys.argv[1:])
h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
srcs = ["BatchMultiBevGen.cpp", "BatchMultiBevGen.h", "KittiPointCloudSelect.cpp",
        "MulranPointCloudSelect.cpp", "OxfordPointCloudSelect.cpp", "src/Utility.cpp",
        "include/Utility.h", "CMakeLists.txt"]
bins = sorted(p for p in out.rglob("*") if p.is_file() and p.parent.name in ("build", "official")
              and p.suffix == "")
rec = {"built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "compiler": subprocess.run(["g++", "--version"], capture_output=True, text=True).stdout.splitlines()[0],
       "flags": "-std=gnu++14 -O3 -Wall -g -ggdb -DDEBUG -fopenmp -DFMT_HEADER_ONLY (official CMake Release)",
       "official_commit": subprocess.run(["git", "-C", str(e), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
       "fmt_commit": subprocess.run(["git", "-C", str(e / "ThirdParties/fmt"), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
       "official_sources_sha256": {s: h(e / s) for s in srcs},
       "driver_sha256": h(repo / "scripts/fresco_official/cpp/fresco_prep_driver.cpp"),
       "compat_sha256": h(repo / "scripts/fresco_official/cpp/shim/vtk_iostream_compat.h"),
       "kitti_groundfix": {**json.loads((out / "kitti_groundfix/patch.json").read_text()),
                           "patched_source": "kitti_groundfix/KittiPointCloudSelect.cpp",
                           "used_by": ["fresco_prep_kitti", "official/kitti_point_cloud_select_groundfix"]},
       "libraries": "Ubuntu jammy debs (apt-get download + dpkg -x): libpcl-dev/io/common 1.12.1+dfsg-3build1, "
                    "libopencv-*-dev/core/imgcodecs 4.5.4+dfsg-9ubuntu4, libvtk9-dev 9.1.0, libeigen3-dev 3.4.0-2ubuntu2, "
                    "libboost1.74-dev; runtime closure in env/rt",
       "binaries": {str(p.relative_to(nas)): h(p) for p in bins}}
(nas / "build/build.json").write_text(json.dumps(rec, indent=1))
print(json.dumps(rec, indent=1))
EOF
