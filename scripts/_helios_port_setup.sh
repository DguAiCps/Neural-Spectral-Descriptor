#!/usr/bin/env bash
# Port the official HeLiOS (ICRA'25, github.com/minwoo0611/HeLiOS) inference
# stack to the `ring_baseline` docker container:
#   torch 2.10 nightly (NGC), CUDA 13.1, python 3.12, RTX 5080 (sm_120).
#
# The official stack pins torch 1.13 / CUDA 11.6 / MinkowskiEngine 0.5.4 /
# sptr / torch_scatter / torch_cluster. None of the official wheels run on
# sm_120, so everything native is rebuilt from source inside the container.
# Sources are copied to /opt/helios_build inside the container and patched
# THERE — the vendored copies under _external/HeLiOS/docker/thirdparty are
# left untouched.
#
# Run from the host:  bash scripts/_helios_port_setup.sh
# Then smoke-test:    docker exec ring_baseline python \
#                       /workspace/Neural-Spectral-Codec/scripts/_helios_infer_smoke.py
#
# Verified 2026-08-19: HeLiOS.pth forward pass OK, descriptor (1, 8448).
set -euo pipefail

C="docker exec ring_baseline"

# 0. If torch.cuda is unavailable inside the container with "Found no NVIDIA
#    driver", the injected WSL libcuda is stale (host driver was updated after
#    the container started). A container restart re-injects it:
#      docker restart ring_baseline

# 1. System dep for MinkowskiEngine (BLAS backend)
$C bash -c 'apt-get update -qq && apt-get install -y -qq libopenblas-dev'

# 2. Copy vendored sources into the container-local build dir + clone PyG ops
$C bash -c '
  mkdir -p /opt/helios_build && cd /opt/helios_build
  rm -rf SparseTransformer MinkowskiEngine
  cp -r /workspace/Neural-Spectral-Codec/_external/HeLiOS/docker/thirdparty/SparseTransformer .
  cp -r /workspace/Neural-Spectral-Codec/_external/HeLiOS/docker/thirdparty/MinkowskiEngine .
  [ -d pytorch_scatter ] || git clone -q --depth 1 https://github.com/rusty1s/pytorch_scatter   # f514c10 @ 2026-08
  [ -d pytorch_cluster ] || git clone -q --depth 1 https://github.com/rusty1s/pytorch_cluster   # e9a855c @ 2026-08
'

# 3. sptr (SparseTransformer CUDA ext) — builds unpatched.
#    Only the sptr_cuda extension is installed; the `sptr` python package is
#    used from the source tree via sys.path (same as HeLiOS eval/evaluate.py).
$C bash -c 'cd /opt/helios_build/SparseTransformer &&
  TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=12 FORCE_CUDA=1 \
  pip install -v --no-build-isolation .'

# 4. torch_scatter + torch_cluster from source — build unpatched.
#    (sptr only needs segment_csr/gather_csr and grid_cluster respectively.)
$C bash -c 'cd /opt/helios_build/pytorch_scatter &&
  TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=8 FORCE_CUDA=1 \
  pip install -v --no-build-isolation .'
$C bash -c 'cd /opt/helios_build/pytorch_cluster &&
  TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=8 FORCE_CUDA=1 \
  pip install -v --no-build-isolation .'

# 5. MinkowskiEngine 0.5.4 — needs 5 patches for CUDA 13 / CCCL 3.x / gcc 13.
$C bash -c 'cd /opt/helios_build/MinkowskiEngine && python - <<"EOF"
def patch(path, old, new, required=True):
    s = open(path).read()
    if old not in s:
        assert (not required) or (new.splitlines()[0] in s or new in s), path
        return
    open(path, "w").write(s.replace(old, new))

# (1) setup.py: keep incremental ninja builds between attempts (optional QoL)
patch("setup.py",
      "run_command(\"rm\", \"-rf\", \"build\")",
      "# run_command(\"rm\", \"-rf\", \"build\")  # local patch: incremental builds",
      required=False)

# (2) vendored cudf nvtx3.hpp clashes with the CUDA 13 nvtx3 headers
#     (ambiguous ::nvtx3 namespace) -> reduce ranges.hpp to no-ops
open("src/3rdparty/cudf/detail/nvtx/ranges.hpp", "w").write("""\
/* Patched for CUDA 13: vendored nvtx3.hpp clashes with the CUDA 13 CCCL nvtx3
 * headers (ambiguous ::nvtx3 namespace). NVTX profiling ranges are not needed
 * for inference builds, so this header is reduced to no-ops. */
#pragma once

namespace cudf {
struct libcudf_domain {
  static constexpr char const* name{"libcudf"};
};
struct thread_range {
  template <typename... Args> explicit thread_range(Args&&...) {}
};
}  // namespace cudf

#define CUDF_FUNC_RANGE()
""")

# (3) thrust::unary_function was removed in CCCL 3.0 -> drop the inheritance
patch("src/coordinate_map_functors.cuh",
      """struct find_coordinate
    : public thrust::unary_function<uint32_t,
                                    thrust::pair<uint32_t, uint32_t>> {""",
      "struct find_coordinate {")

# (4) CCCL 3.x thrust headers are no longer transitively included
patch("src/gpu.cuh",
      "#include <driver_types.h> // cuda driver types",
      """#include <driver_types.h> // cuda driver types

// CCCL 3.x (CUDA 13) no longer includes these transitively
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/remove.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>""")

# (5a) concurrent_unordered_map: execution_policy include + CUDA 13 removed
#      the (ptr, count, int device, stream) overload of cudaMemPrefetchAsync
patch("src/3rdparty/concurrent_unordered_map.cuh",
      "#include <thrust/pair.h>\n#include <thrust/count.h>",
      """#include <thrust/pair.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>  // CUDA 13/CCCL 3.x: no longer transitive

// CUDA 13 removed the legacy (ptr, count, int device, stream) overload of
// cudaMemPrefetchAsync. Wrap the cudaMemLocation-based replacement.
static inline cudaError_t me_mem_prefetch_async(const void* ptr, size_t count,
                                                int dev_id, cudaStream_t stream) {
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 13000)
  cudaMemLocation loc{};
  loc.type = cudaMemLocationTypeDevice;
  loc.id = dev_id;
  return cudaMemPrefetchAsync(ptr, count, loc, 0u, stream);
#else
  return cudaMemPrefetchAsync(ptr, count, dev_id, stream);
#endif
}""")
patch("src/3rdparty/concurrent_unordered_map.cuh",
      "cudaMemPrefetchAsync(m_hashtbl_values", "me_mem_prefetch_async(m_hashtbl_values",
      required=False)
patch("src/3rdparty/concurrent_unordered_map.cuh",
      "cudaMemPrefetchAsync(this", "me_mem_prefetch_async(this",
      required=False)

# (5b) coordinate_map_gpu.cuh: unique_ptr -> shared_ptr conversion is now
#      ambiguous (thrust::pair == cuda::std::pair poisons ADL, so the
#      unqualified __to_address call inside libstdc++ matches both std::
#      and cuda::std::); construct from raw pointer + deleter instead.
patch("src/coordinate_map_gpu.cuh",
      """      m_map = map_type::create(
          compute_hash_table_size(size, m_hashtable_occupancy),
          m_unused_element, m_unused_key, m_hasher, m_equal, m_map_allocator);""",
      """      // CUDA 13/CCCL 3.x: assigning the unique_ptr from create() to the
      // shared_ptr member is ambiguous (ADL pulls cuda::std::__to_address);
      // construct the shared_ptr from the raw pointer + deleter instead.
      auto new_map = map_type::create(
          compute_hash_table_size(size, m_hashtable_occupancy),
          m_unused_element, m_unused_key, m_hasher, m_equal, m_map_allocator);
      m_map = std::shared_ptr<map_type>(new_map.release(), new_map.get_deleter());""")

print("ME patches applied")
EOF'

# 6. Build MinkowskiEngine.
#    - CPATH: CUDA 13 moved thrust/cub to include/cccl/, which host g++ does
#      not search by default (nvcc does).
#    - --blas=openblas: the auto-detect path uses numpy.distutils, which was
#      removed in numpy 2.x.
$C bash -c 'cd /opt/helios_build/MinkowskiEngine &&
  export TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=10 CPATH=/usr/local/cuda/include/cccl:${CPATH:-} &&
  python setup.py install --force_cuda --blas=openblas'

# 7. HeLiOS repo-side addition (already committed in this repo):
#    _external/HeLiOS/datasets/__init__.py (empty) — makes `datasets` a
#    regular package so it wins over the HuggingFace `datasets` installed in
#    the NGC image (a regular site-packages package beats a namespace dir).
touch "$(dirname "$0")/../_external/HeLiOS/datasets/__init__.py"

echo "HeLiOS porting setup complete."
