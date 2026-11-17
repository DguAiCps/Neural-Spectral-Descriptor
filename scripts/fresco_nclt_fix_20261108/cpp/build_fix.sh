#!/usr/bin/env bash
# NCLT 수정 드라이버 빌드(서버 호스트, 사용자공간, sudo 없음). 기존 build.sh와 같은 컴파일러·플래그·헤더·라이브러리.
#   bash scripts/fresco_nclt_fix_20261108/cpp/build_fix.sh
# 헤더·런타임은 기존 NAS env(읽기만)를 쓰고, 목적 파일과 실행 파일은 새 NAS build/에만 쓴다.
# 공식 TU(BatchMultiBevGen.cpp, OxfordPointCloudSelect.cpp, src/Utility.cpp)는 무수정, 기호 이름만 -D로 바꾼다(기존과 같음).
# 대조용으로 기존 G3 드라이버(fresco_prep_oxford)도 같은 명령으로 새 build/에 다시 만들어 기존 바이너리와 해시를 비교한다.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
OLD=/mnt/aicps-nas/qkrrlans00/nsd_fresco_official_20261107
NEW=/mnt/aicps-nas/qkrrlans00/nsd_fresco_nclt_fix_20261108
E=$REPO/_external/fresco_official_20261107/Point-Cloud-Preprocessing-Tools
H=$OLD/env/hdr/usr/include
L=$OLD/env/rt/usr/lib/x86_64-linux-gnu
L2=$OLD/env/rt/usr/lib
SHIM=$REPO/scripts/fresco_official/cpp/shim
OUT=$NEW/build
mkdir -p "$OUT/obj"
[ "$(git -C "$E" rev-parse HEAD)" = d94040ed626c7275b6f428deddde2ddca74419d7 ]
[ -z "$(git -C "$E" status --porcelain --untracked-files=no)" ]

CXX=(g++ -std=gnu++14 -O3 -Wall -g -ggdb -DDEBUG -fopenmp -DFMT_HEADER_ONLY
     -I"$E/include" -I"$E" -I"$E/ThirdParties/fmt/include" -I"$H" -I"$H/pcl-1.12"
     -I"$H/eigen3" -I"$H/opencv4" -I"$H/vtk-9.1")
LIBS=("$L/libpcl_io.so.1.12" "$L/libpcl_common.so.1.12" "$L/libopencv_imgcodecs.so.4.5d"
      "$L/libopencv_core.so.4.5d" "$L/libvtkCommonCore-9.1.so.1" "$L/libvtksys-9.1.so.1"
      -Wl,--disable-new-dtags -Wl,-rpath,"$L:$L2" -Wl,-rpath-link,"$L:$L2" -fopenmp)
COMPAT=(-include "$SHIM/vtk_iostream_compat.h")
cc() { echo "+ ${*: -1}" >&2; "${CXX[@]}" -w "$@"; }

cc -c "$E/src/Utility.cpp" -o "$OUT/obj/Utility.o"
cc -Dmain=fresco_official_bevgen_main -DinitDirectories=fresco_official_bevgen_initDirectories \
   -c "$E/BatchMultiBevGen.cpp" -o "$OUT/obj/BatchMultiBevGen.o"
cc "${COMPAT[@]}" -Dmain=fresco_official_oxford_main -DinitDirectories=fresco_official_oxford_initDirectories \
   -DPose6f=FrescoOxfordPose6f -c "$E/OxfordPointCloudSelect.cpp" -o "$OUT/obj/Oxford.o"
cc -c "$HERE/fresco_prep_ncltflu.cpp" -o "$OUT/obj/driver_ncltflu.o"
cc -DFRESCO_FAMILY_OXFORD -c "$REPO/scripts/fresco_official/cpp/fresco_prep_driver.cpp" -o "$OUT/obj/driver_oxford.o"
g++ -o "$OUT/fresco_prep_ncltflu" "$OUT/obj/driver_ncltflu.o" "$OUT/obj/Oxford.o" "$OUT/obj/BatchMultiBevGen.o" "$OUT/obj/Utility.o" "${LIBS[@]}"
g++ -o "$OUT/fresco_prep_oxford_g3_rebuild" "$OUT/obj/driver_oxford.o" "$OUT/obj/Oxford.o" "$OUT/obj/BatchMultiBevGen.o" "$OUT/obj/Utility.o" "${LIBS[@]}"

python3 - "$REPO" "$E" "$OUT" "$OLD" <<'EOF'
import hashlib, json, subprocess, sys, time
from pathlib import Path
repo, e, out, old = map(Path, sys.argv[1:])
h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
srcs = ["BatchMultiBevGen.cpp", "BatchMultiBevGen.h", "OxfordPointCloudSelect.cpp", "src/Utility.cpp",
        "include/Utility.h"]
rec = {"built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "compiler": subprocess.run(["g++", "--version"], capture_output=True, text=True).stdout.splitlines()[0],
       "flags": "-std=gnu++14 -O3 -Wall -g -ggdb -DDEBUG -fopenmp -DFMT_HEADER_ONLY (기존 build.sh와 같음)",
       "official_commit": subprocess.run(["git", "-C", str(e), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
       "official_sources_sha256": {s: h(e / s) for s in srcs},
       "driver_ncltflu_cpp_sha256": h(repo / "scripts/fresco_nclt_fix_20261108/cpp/fresco_prep_ncltflu.cpp"),
       "build_fix_sh_sha256": h(repo / "scripts/fresco_nclt_fix_20261108/cpp/build_fix.sh"),
       "old_driver_cpp_sha256": h(repo / "scripts/fresco_official/cpp/fresco_prep_driver.cpp"),
       "binaries": {p.name: h(p) for p in sorted(out.glob("fresco_prep_*"))},
       "old_oxford_binary_sha256": h(old / "build/fresco_prep_oxford"),
       "old_build_official_sources_sha256": json.loads((old / "build/build.json").read_text())["official_sources_sha256"]}
rec["official_sources_equal_old_build"] = all(rec["old_build_official_sources_sha256"][s] == v
                                              for s, v in rec["official_sources_sha256"].items())
(out / "build_fix.json").write_text(json.dumps(rec, indent=1))
print(json.dumps(rec, indent=1))
EOF
