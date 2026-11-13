// 표 3 보조 진단 전용 CUDA 요청 추적기. 주 계측에는 주입하지 않는다.
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <sstream>
#include <string>

namespace {
struct Allocation {
    uint64_t id;
    size_t bytes;
    std::string api;
    std::string owner;
};

struct State {
    std::mutex mutex;
    std::map<uintptr_t, Allocation> linear;
    std::map<uintptr_t, Allocation> arrays;
    uint64_t events = 0;
    uint64_t allocations = 0;
    uint64_t frees = 0;
    int fd = -1;
    State() {
        const char* path = std::getenv("NSD_CUDA_TRACE_PATH");
        if (path) fd = open(path, O_CREAT | O_WRONLY | O_APPEND, 0644);
    }
};

State& state() {
    // 종료 단계의 CUDA 해제도 추적하므로 프로세스 수명 동안 유지한다.
    static State* value = new State();
    return *value;
}

std::string quote(const std::string& input) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : input) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c == '\n') out << "\\n";
        else if (c == '\r') out << "\\r";
        else if (c == '\t') out << "\\t";
        else if (c >= 32) out << c;
    }
    out << '"';
    return out.str();
}

std::string pointer(uintptr_t ptr) {
    std::ostringstream out;
    out << "0x" << std::hex << ptr;
    return out.str();
}

std::string owner(void* address) {
    Dl_info info{};
    return dladdr(address, &info) && info.dli_fname ? info.dli_fname : "<unknown>";
}

void emit(State& s, const std::string& body) {
    if (s.fd < 0) return;
    std::string line = "{\"event_id\":" + std::to_string(++s.events) + "," + body + "}\n";
    const char* data = line.data();
    size_t remaining = line.size();
    while (remaining) {
        ssize_t wrote = write(s.fd, data, remaining);
        if (wrote <= 0) return;
        data += wrote;
        remaining -= static_cast<size_t>(wrote);
    }
}

template <class F> F resolve(const char* name) {
    void* address = dlsym(RTLD_NEXT, name);
    if (!address) {
        // CUDA 확장을 RTLD_LOCAL로 불러온 Python 프로세스의 보조 경로다.
        for (const char* library : {"libcudart.so.13", "libcudart.so.12", "libcudart.so"}) {
            void* handle = dlopen(library, RTLD_NOW | RTLD_NOLOAD);
            if (handle) address = dlsym(handle, name);
            if (address) break;
        }
    }
    if (!address) {
        const char* message = "NSD CUDA trace: runtime symbol resolution failed\n";
        ssize_t ignored = write(STDERR_FILENO, message, std::strlen(message));
        (void)ignored;
        _exit(127);
    }
    return reinterpret_cast<F>(address);
}

void allocated(void* ptr, size_t bytes, const char* api, void* caller,
               cudaError_t error, bool array) {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    std::string source = owner(caller);
    uintptr_t key = reinterpret_cast<uintptr_t>(ptr);
    if (error == cudaSuccess && ptr) {
        (array ? s.arrays : s.linear)[key] = {++s.allocations, bytes, api, source};
    }
    emit(s, "\"type\":\"allocate\",\"api\":" + quote(api) +
         ",\"owner\":" + quote(source) + ",\"pointer\":" + quote(pointer(key)) +
         ",\"requested_bytes\":" + std::to_string(bytes) +
         ",\"error\":" + std::to_string(static_cast<int>(error)) +
         ",\"array\":" + (array ? "true" : "false"));
}

void freed(void* ptr, const char* api, void* caller, cudaError_t error, bool array) {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    auto& entries = array ? s.arrays : s.linear;
    uintptr_t key = reinterpret_cast<uintptr_t>(ptr);
    auto found = entries.find(key);
    bool known = found != entries.end();
    size_t bytes = known ? found->second.bytes : 0;
    std::string source = known ? found->second.owner : "<untracked>";
    if (error == cudaSuccess && known) {
        entries.erase(found);
        ++s.frees;
    }
    emit(s, "\"type\":\"free\",\"api\":" + quote(api) +
         ",\"owner\":" + quote(source) + ",\"caller\":" + quote(owner(caller)) +
         ",\"pointer\":" + quote(pointer(key)) +
         ",\"requested_bytes\":" + std::to_string(bytes) +
         ",\"tracked\":" + (known ? "true" : "false") +
         ",\"error\":" + std::to_string(static_cast<int>(error)));
}

size_t array_bytes(const cudaChannelFormatDesc* channel, size_t width,
                   size_t height, size_t depth) {
    const size_t bits = std::abs(channel->x) + std::abs(channel->y) +
                        std::abs(channel->z) + std::abs(channel->w);
    return width * (height ? height : 1) * (depth ? depth : 1) * ((bits + 7) / 8);
}
}  // namespace

extern "C" cudaError_t cudaMalloc(void** ptr, size_t bytes) {
    static auto actual = resolve<cudaError_t (*)(void**, size_t)>("cudaMalloc");
    cudaError_t error = actual(ptr, bytes);
    allocated(error == cudaSuccess ? *ptr : nullptr, bytes, "cudaMalloc",
              __builtin_return_address(0), error, false);
    return error;
}

extern "C" cudaError_t cudaFree(void* ptr) {
    static auto actual = resolve<cudaError_t (*)(void*)>("cudaFree");
    cudaError_t error = actual(ptr);
    freed(ptr, "cudaFree", __builtin_return_address(0), error, false);
    return error;
}

extern "C" cudaError_t cudaMalloc3DArray(cudaArray_t* ptr,
        const cudaChannelFormatDesc* channel, cudaExtent extent, unsigned int flags) {
    static auto actual = resolve<cudaError_t (*)(cudaArray_t*, const cudaChannelFormatDesc*, cudaExtent, unsigned int)>("cudaMalloc3DArray");
    cudaError_t error = actual(ptr, channel, extent, flags);
    allocated(error == cudaSuccess ? *ptr : nullptr,
              array_bytes(channel, extent.width, extent.height, extent.depth),
              "cudaMalloc3DArray", __builtin_return_address(0), error, true);
    return error;
}

extern "C" cudaError_t cudaMallocArray(cudaArray_t* ptr,
        const cudaChannelFormatDesc* channel, size_t width, size_t height, unsigned int flags) {
    static auto actual = resolve<cudaError_t (*)(cudaArray_t*, const cudaChannelFormatDesc*, size_t, size_t, unsigned int)>("cudaMallocArray");
    cudaError_t error = actual(ptr, channel, width, height, flags);
    allocated(error == cudaSuccess ? *ptr : nullptr,
              array_bytes(channel, width, height, 1), "cudaMallocArray",
              __builtin_return_address(0), error, true);
    return error;
}

extern "C" cudaError_t cudaFreeArray(cudaArray_t ptr) {
    static auto actual = resolve<cudaError_t (*)(cudaArray_t)>("cudaFreeArray");
    cudaError_t error = actual(ptr);
    freed(ptr, "cudaFreeArray", __builtin_return_address(0), error, true);
    return error;
}

extern "C" void nsd_native_trace_snapshot(const char* label) {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    std::map<std::string, size_t> totals;
    size_t bytes = 0;
    std::ostringstream entries;
    bool first = true;
    for (const auto* bank : {&s.linear, &s.arrays}) {
        for (const auto& item : *bank) {
            const auto& allocation = item.second;
            bytes += allocation.bytes;
            totals[allocation.owner] += allocation.bytes;
            if (!first) entries << ',';
            first = false;
            entries << "{\"pointer\":" << quote(pointer(item.first))
                    << ",\"allocation_id\":" << allocation.id
                    << ",\"requested_bytes\":" << allocation.bytes
                    << ",\"owner\":" << quote(allocation.owner)
                    << ",\"api\":" << quote(allocation.api) << '}';
        }
    }
    std::ostringstream grouped;
    first = true;
    for (const auto& item : totals) {
        if (!first) grouped << ',';
        first = false;
        grouped << quote(item.first) << ':' << item.second;
    }
    emit(s, "\"type\":\"snapshot\",\"label\":" + quote(label ? label : "") +
         ",\"requested_live_bytes\":" + std::to_string(bytes) +
         ",\"allocations\":" + std::to_string(s.allocations) +
         ",\"frees\":" + std::to_string(s.frees) +
         ",\"by_owner\":{" + grouped.str() + "},\"live\":[" + entries.str() + "]");
}
