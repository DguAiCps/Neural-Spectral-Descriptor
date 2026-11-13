#!/usr/bin/env python3
"""모델 객체의 등록·비등록 수치 저장소를 실제 주소로 중복 제거한다."""
from __future__ import annotations

import argparse
import inspect
import json
from types import ModuleType
import weakref

import numpy as np
import torch


def _union_bytes(records: list) -> int:
    """중첩 뷰와 별칭을 포함한 주소 구간의 합집합 크기다."""
    intervals = sorted((row["address"], row["address"]+row["allocation_bytes"])
                       for row in records if row["allocation_bytes"])
    total, end = 0, 0
    for start, stop in intervals:
        total += max(0, stop-max(start, end))
        end = max(end, stop)
    return total


def _tensor_buffers(value: torch.Tensor, path: str) -> list:
    """텐서 뷰의 원본 저장소 전체 크기와 주소를 얻는다."""
    if value.layout != torch.strided:
        if value.layout == torch.sparse_coo:
            return (_tensor_buffers(value._indices(), path+"._indices")
                    + _tensor_buffers(value._values(), path+"._values"))
        raise TypeError(f"지원하지 않는 텐서 저장소 형식: {path}: {value.layout}")
    if value.device.type == "meta":
        return []
    storage = value.untyped_storage()
    return [{"path": path, "kind": "torch", "device": str(value.device),
             "address": storage.data_ptr(), "allocation_bytes": storage.nbytes(),
             "logical_bytes": value.numel()*value.element_size(),
             "shape": list(value.shape), "dtype": str(value.dtype),
             "allocation_known": True}]


def _numpy_buffer(value: np.ndarray, path: str) -> dict:
    """NumPy 뷰의 원본 할당을 따라가서 바이트를 센다."""
    root = value
    visited = set()
    while isinstance(root.base, np.ndarray):
        assert id(root) not in visited, f"NumPy base cycle: {path}"
        visited.add(id(root))
        root = root.base
    if isinstance(root.base, torch.Tensor):
        record = _tensor_buffers(root.base, path)[0]
        record.update({"kind": "numpy_alias_of_torch", "shape": list(value.shape),
                       "dtype": str(value.dtype), "logical_bytes": value.nbytes})
        return record
    address = int(root.__array_interface__["data"][0])
    size = int(root.nbytes)
    known = bool(root.flags.owndata)
    base_kind = "owned_numpy"
    if root.base is not None:
        base_kind = type(root.base).__name__
        try:
            # memoryview/bytearray/ctypes 같은 buffer protocol의 전체 할당이다.
            whole = np.frombuffer(root.base, dtype=np.uint8)
            address = int(whole.__array_interface__["data"][0])
            size = int(whole.nbytes)
            known = True
        except (TypeError, ValueError, BufferError):
            # opaque C capsule의 용량은 알 수 없어 접근 가능한 범위만 기록한다.
            known = False
    return {"path": path, "kind": "numpy", "device": "cpu", "address": address,
            "allocation_bytes": size, "logical_bytes": value.nbytes,
            "shape": list(value.shape), "dtype": str(value.dtype),
            "allocation_known": known, "base_kind": base_kind}


def audit_numeric_state(*roots, max_objects: int = 2_000_000) -> dict:
    """모델·설정 객체에서 도달하는 CPU 수치 버퍼를 중복 없이 센다.

    GPU 반환값은 텐서 목록의 감사용이다. 실제 CUDA allocated에 더하지 않는다.
    함수 closure와 모듈·클래스의 전역 자료는 따라가지 않는다.
    """
    stack = [(f"root{i}", value) for i, value in enumerate(roots)]
    seen, records, registered = set(), [], []
    ignored = {"module_or_function_or_class": 0, "weak_reference": 0,
               "scalar_or_string": 0, "no_instance_state": 0}
    while stack:
        path, value = stack.pop()
        if value is None or isinstance(value, (str, bytes, int, float, complex, bool, np.generic)):
            ignored["scalar_or_string"] += 1
            continue
        if isinstance(value, (ModuleType, type)) or inspect.isroutine(value) or isinstance(value, np.ufunc):
            ignored["module_or_function_or_class"] += 1
            continue
        if type(value) in weakref.ProxyTypes or isinstance(value, weakref.ReferenceType):
            ignored["weak_reference"] += 1
            continue
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)
        if len(seen) > max_objects:
            raise RuntimeError("모델 메모리 감사 객체 상한 초과; 조용히 자르지 않는다")
        if isinstance(value, torch.Tensor):
            records.extend(_tensor_buffers(value, path))
            continue
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                stack.extend((f"{path}.object[{i}]", v) for i, v in enumerate(value.flat))
            elif value.dtype.kind in "biufc":
                records.append(_numpy_buffer(value, path))
            continue
        if isinstance(value, dict):
            stack.extend((f"{path}[{str(key)!r}]", child) for key, child in value.items())
            continue
        if isinstance(value, (list, tuple, set, frozenset)):
            stack.extend((f"{path}[{i}]", child) for i, child in enumerate(value))
            continue
        if isinstance(value, torch.nn.Module):
            for name, parameter in value.named_parameters(recurse=False):
                registered.extend(_tensor_buffers(parameter, path+".parameter."+name))
            for name, buffer in value.named_buffers(recurse=False):
                registered.extend(_tensor_buffers(buffer, path+".buffer."+name))
        found = False
        if hasattr(value, "__dict__"):
            stack.extend((f"{path}.{key}", child) for key, child in vars(value).items())
            found = True
        # Minkowski KernelGenerator 등 slots-only 인스턴스도 포함한다.
        for cls in type(value).__mro__:
            slots = cls.__dict__.get("__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for slot in slots:
                if slot in ("__dict__", "__weakref__"):
                    continue
                try:
                    child = getattr(value, slot)
                except AttributeError:
                    continue
                stack.append((f"{path}.{slot}", child))
                found = True
        if not found:
            ignored["no_instance_state"] += 1
    cpu = [row for row in records if row["device"] == "cpu"]
    gpu_devices = sorted({row["device"] for row in records if row["device"].startswith("cuda")})
    gpu = {device: _union_bytes([row for row in records if row["device"] == device]) for device in gpu_devices}
    registered_cpu = _union_bytes([row for row in registered if row["device"] == "cpu"])
    total_cpu = _union_bytes(cpu)
    assert total_cpu >= registered_cpu
    unknown = [row["path"] for row in records if not row["allocation_known"]]
    return {"schema": "pdv2-model-numeric-memory-v1", "status": "passed" if not unknown else "capacity_unresolved",
            "cpu_unique_bytes": total_cpu, "registered_cpu_bytes": registered_cpu,
            "extra_cpu_bytes": total_cpu-registered_cpu,
            "gpu_tensor_unique_bytes": sum(gpu.values()), "gpu_tensor_bytes_by_device": gpu,
            "objects_visited": len(seen), "numeric_buffers_seen": len(records),
            "cpu_alias_bytes_removed": sum(row["allocation_bytes"] for row in cpu)-total_cpu,
            "unknown_capacity_paths": unknown, "ignored": ignored,
            "buffers": records, "registered_buffers": registered,
            "scope": "Instance vars/slots and containers reachable from supplied model/config roots only; numeric ndarray and tensor allocations, including base storage/capacity; overlapping CPU addresses merged across numpy/torch; object headers, Python scalar metadata, classes, modules, functions, closures and weak references excluded. GPU tensor union is diagnostic and must not be added to CUDA allocated again."}


def selftest() -> dict:
    """실제 공유 저장소·분리 복사·slots·순환 참조를 합성 입력으로 검사한다."""
    class SlotState:
        __slots__ = ("array",)

    class Example(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(12, dtype=torch.float32))
            self.register_buffer("bias", torch.zeros(5, dtype=torch.float64))
            self.weight_np_alias = self.weight.detach().numpy()[2:]
            self.plain_t = torch.ones(7, dtype=torch.int16)
            self.plain = np.arange(21, dtype=np.float32)
            self.views = [self.plain[:10], self.plain[2::3], self.plain]
            self.slot = SlotState()
            self.slot.array = np.arange(3, dtype=np.int64)
            self.child = {"self": self, "same_weight": self.weight}

    module = Example()
    result = audit_numeric_state(module)
    expected_registered = 12*4+5*8
    expected_total = expected_registered+7*2+21*4+3*8
    assert result["status"] == "passed"
    assert result["registered_cpu_bytes"] == expected_registered
    assert result["cpu_unique_bytes"] == expected_total
    assert result["extra_cpu_bytes"] == expected_total-expected_registered
    assert result["cpu_alias_bytes_removed"] > 0
    copy = module.plain.copy()
    assert audit_numeric_state(module, copy)["cpu_unique_bytes"] == expected_total+copy.nbytes
    assert audit_numeric_state(module, module)["cpu_unique_bytes"] == expected_total
    # Buffer-protocol 원본 bytearray 안의 좁은 NumPy 뷰도 전체 용량을 센다.
    data = bytearray(100)
    view = np.frombuffer(data, dtype=np.uint8)[20:40]
    assert audit_numeric_state(view)["cpu_unique_bytes"] == 100
    return {"status": "passed", "registered_cpu_bytes": expected_registered,
            "cpu_unique_bytes": expected_total,
            "checks": ["numpy overlapping/strided views", "numpy alias of registered torch storage",
                       "plain tensor", "slots-only object", "cyclic references", "separate allocation",
                       "repeated model root", "buffer-protocol owner capacity"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if not args.selftest:
        parser.error("직접 실행은 --selftest, 실제 모델은 audit_numeric_state(*roots)를 호출한다")
    print(json.dumps(selftest(), ensure_ascii=False, indent=2))
