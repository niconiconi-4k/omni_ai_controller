from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import psutil


def _round(value: float) -> float:
    return round(value, 1)


def _gpu_number(value: str) -> float | None:
    try:
        return _round(float(value))
    except ValueError:
        return None


def gpu_status() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=4,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []

    gpus: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 7:
            continue
        gpus.append(
            {
                "index": fields[0],
                "name": fields[1],
                "temperature_c": _gpu_number(fields[2]),
                "utilization_percent": _gpu_number(fields[3]),
                "memory_used_mib": _gpu_number(fields[4]),
                "memory_total_mib": _gpu_number(fields[5]),
                "power_watts": _gpu_number(fields[6]),
            }
        )
    return gpus


def hardware_status(data_path: Path = Path("/opt/ai_server")) -> dict[str, Any]:
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    root_disk = shutil.disk_usage("/")
    data_disk = shutil.disk_usage(data_path if data_path.exists() else "/")
    network = psutil.net_io_counters()
    temperatures = psutil.sensors_temperatures(fahrenheit=False)
    cpu_temperatures = [
        reading.current
        for readings in temperatures.values()
        for reading in readings
        if reading.current is not None
    ]

    return {
        "timestamp": time.time(),
        "uptime_seconds": max(0, time.time() - psutil.boot_time()),
        "cpu": {
            "percent": _round(psutil.cpu_percent(interval=0.1)),
            "logical_cores": psutil.cpu_count(logical=True),
            "physical_cores": psutil.cpu_count(logical=False),
            "load_average": [_round(value) for value in psutil.getloadavg()],
            "temperature_c": _round(max(cpu_temperatures)) if cpu_temperatures else None,
        },
        "memory": {
            "percent": _round(memory.percent),
            "used_bytes": memory.used,
            "total_bytes": memory.total,
            "swap_used_bytes": swap.used,
            "swap_total_bytes": swap.total,
        },
        "disks": {
            "root": {
                "used_bytes": root_disk.used,
                "total_bytes": root_disk.total,
                "percent": _round(root_disk.used / root_disk.total * 100),
            },
            "data": {
                "path": str(data_path),
                "used_bytes": data_disk.used,
                "total_bytes": data_disk.total,
                "percent": _round(data_disk.used / data_disk.total * 100),
            },
        },
        "network": {
            "bytes_sent": network.bytes_sent,
            "bytes_received": network.bytes_recv,
        },
        "gpus": gpu_status(),
    }
