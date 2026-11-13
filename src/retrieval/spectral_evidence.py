"""단일 회전 피크와 주행 회전을 보정한 시간 정렬 신뢰도를 비교한다."""
from typing import Dict

import numpy as np


def correlation_profiles(bank: tuple, query: np.ndarray,
                         candidate: np.ndarray, channel: str) -> np.ndarray:
    """주어진 프레임 쌍의 모든 회전각 점수를 반환한다."""
    query = np.asarray(query, dtype=np.int64)
    candidate = np.asarray(candidate, dtype=np.int64)
    assert query.shape == candidate.shape and query.ndim == 1
    if channel == "c_spec":
        _, fft, norm, _ = bank
        sectors = (fft.shape[-1] - 1) * 2
        profile = np.fft.irfft(fft[candidate] * np.conj(fft[query]),
                               n=sectors, axis=-1).sum(axis=1)
        return profile / (norm[candidate] * norm[query])[:, None]
    assert channel == "c_col", channel
    fft, mask_fft = bank[3]
    sectors = (fft.shape[-1] - 1) * 2
    profile = np.fft.irfft(fft[candidate] * np.conj(fft[query]),
                           n=sectors, axis=-1).sum(axis=1)
    count = np.fft.irfft(mask_fft[candidate] * np.conj(mask_fft[query]),
                        n=sectors, axis=-1)
    return np.where(count > .5, profile / np.maximum(count, 1.), -1.)


def peak_to_sidelobe(profile: np.ndarray, guard_bins: int = 3) -> np.ndarray:
    """피크 주변을 제외한 평균·표준편차로 피크 높이를 정규화한다."""
    profile = np.asarray(profile, dtype=np.float64)
    assert profile.ndim == 2 and profile.shape[1] > 2 * guard_bins + 1
    peak = np.argmax(profile, axis=1)
    delta = np.abs(np.arange(profile.shape[1])[None, :] - peak[:, None])
    mask = np.minimum(delta, profile.shape[1] - delta) > guard_bins
    mean = (profile * mask).sum(1) / mask.sum(1)
    var = (((profile - mean[:, None]) ** 2) * mask).sum(1) / mask.sum(1)
    return (profile.max(1) - mean) / np.maximum(np.sqrt(var), 1e-12)


def transport_profile(profile: np.ndarray, yaw_delta: np.ndarray) -> np.ndarray:
    """이웃의 회전 증가분 차이를 제거해 중심 쌍의 회전축으로 옮긴다."""
    n = profile.shape[1]
    frequencies = np.arange(n // 2 + 1)
    phase = np.exp(1j * yaw_delta[:, None] * frequencies)
    return np.fft.irfft(np.fft.rfft(profile, axis=1) * phase, n=n, axis=1)


def temporal_scores(bank: tuple, yaw: np.ndarray, query: np.ndarray,
                    candidate: np.ndarray, channel: str,
                    offset: int = 5, guard_bins: int = 3) -> Dict[str, np.ndarray]:
    """±offset의 순방향/역방향 정렬을 같은 예산에서 비교한다.

    후보 중심은 고정한다. 범위를 벗어난 이웃 쌍은 제외하며 GT를 쓰지 않는다.
    coherent는 회전 보정 후 평균 곡선의 최댓값, independent는 각 곡선의
    최댓값 평균이다. 후자는 공통 회전 제약을 제거한 대조다.
    """
    q, j = np.asarray(query, dtype=np.int64), np.asarray(candidate, dtype=np.int64)
    n = len(yaw)
    assert q.shape == j.shape and np.all((q >= 0) & (q < n))
    assert np.all((j >= 0) & (j < n))
    center = correlation_profiles(bank, q, j, channel)
    center_peak = center.max(1)
    independent, coherent, untransported, psr, supports = [], [], [], [], []
    for direction in (1, -1):
        total, plain = center.copy(), center.copy()
        peaks, count = center_peak.copy(), np.ones(len(q), dtype=np.int64)
        for step in (-offset, offset):
            qn, jn = q + step, j + direction * step
            valid = (qn >= 0) & (qn < n) & (jn >= 0) & (jn < n)
            if not valid.any():
                continue
            p = correlation_profiles(bank, qn[valid], jn[valid], channel)
            delta = ((yaw[qn[valid]] - yaw[q[valid]])
                     - (yaw[jn[valid]] - yaw[j[valid]]))
            total[valid] += transport_profile(p, delta)
            plain[valid] += p
            peaks[valid] += p.max(1)
            count[valid] += 1
        total /= count[:, None]
        independent.append(peaks / count)
        coherent.append(total.max(1))
        untransported.append((plain / count[:, None]).max(1))
        psr.append(peak_to_sidelobe(total, guard_bins))
        supports.append(count)
    return {"raw": center_peak, "psr": peak_to_sidelobe(center, guard_bins),
            "temporal_independent": np.maximum(*independent),
            "temporal_untransported": np.maximum(*untransported),
            "temporal_coherent": np.maximum(*coherent),
            "temporal_coherent_psr": np.maximum(*psr),
            "support_forward": supports[0], "support_reverse": supports[1]}
