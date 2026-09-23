from datetime import timezone

from omni_ai_controller.metric_store import METRIC_SERIES, RANGE_CONFIG, MetricStore


def test_snapshot_values_include_gpu_memory_and_power() -> None:
    values = MetricStore._snapshot_values(
        {
            "cpu": {"percent": 25.0, "temperature_c": 55.0},
            "memory": {"percent": 40.0},
            "disks": {"data": {"percent": 60.0}},
            "host_power_watts": None,
            "gpus": [
                {
                    "utilization_percent": 70.0,
                    "memory_used_mib": 12_000,
                    "memory_total_mib": 24_000,
                    "temperature_c": 65.0,
                    "power_watts": 320.0,
                }
            ],
        }
    )

    assert values == (25.0, 55.0, None, 40.0, 60.0, 70.0, 50.0, 65.0, 320.0)


def test_metric_ranges_use_progressively_coarser_resolutions() -> None:
    assert RANGE_CONFIG["5m"][:2] == (5, 5)
    assert RANGE_CONFIG["1d"][:2] == (60, 120)
    assert RANGE_CONFIG["7d"][:2] == (900, 900)
    assert RANGE_CONFIG["30d"][:2] == (900, 3600)
    assert RANGE_CONFIG["1y"][:2] == (3600, 28_800)
    assert any(item["key"] == "host_power_watts" for item in METRIC_SERIES["cpu"])
    assert any(item["key"] == "gpu_power_watts" for item in METRIC_SERIES["gpu"])


def test_bucket_is_aligned_to_resolution() -> None:
    bucket = MetricStore._bucket(1_700_000_123.0, 60)

    assert bucket.tzinfo == timezone.utc
    assert int(bucket.timestamp()) % 60 == 0


def test_gpu_memory_is_null_without_capacity() -> None:
    values = MetricStore._snapshot_values({"gpus": [{"memory_used_mib": 100}]})

    assert values[6] is None
