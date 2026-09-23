from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Literal

import psycopg
from psycopg.rows import dict_row

LOGGER = logging.getLogger("omni_ai_controller.metrics")
MetricName = Literal["cpu", "memory", "disk", "gpu"]
MetricRange = Literal["5m", "1d", "7d", "30d", "1y"]

RANGE_CONFIG: dict[MetricRange, tuple[int, int, timedelta, str]] = {
    "5m": (5, 5, timedelta(minutes=5), "5 秒"),
    "1d": (60, 120, timedelta(days=1), "2 分钟"),
    "7d": (900, 900, timedelta(days=7), "15 分钟"),
    "30d": (900, 3600, timedelta(days=30), "1 小时"),
    "1y": (3600, 28_800, timedelta(days=365), "8 小时"),
}

METRIC_SERIES: dict[MetricName, tuple[dict[str, str], ...]] = {
    "cpu": (
        {"key": "cpu_percent", "label": "CPU 使用率", "unit": "%", "color": "#5ee7d0"},
        {"key": "cpu_temperature_c", "label": "CPU 温度", "unit": "°C", "color": "#f1b95c"},
        {"key": "host_power_watts", "label": "整机功耗", "unit": "W", "color": "#ff6f7d"},
    ),
    "memory": (
        {"key": "memory_percent", "label": "内存使用率", "unit": "%", "color": "#63a9ff"},
    ),
    "disk": (
        {"key": "disk_percent", "label": "数据磁盘使用率", "unit": "%", "color": "#a893ff"},
    ),
    "gpu": (
        {"key": "gpu_utilization_percent", "label": "GPU 使用率", "unit": "%", "color": "#f1b95c"},
        {"key": "gpu_memory_percent", "label": "显存使用率", "unit": "%", "color": "#a893ff"},
        {"key": "gpu_temperature_c", "label": "GPU 温度", "unit": "°C", "color": "#ff8f70"},
        {"key": "gpu_power_watts", "label": "GPU 功耗", "unit": "W", "color": "#5ee7d0"},
    ),
}


class MetricStoreError(RuntimeError):
    """Raised when hardware metric history cannot be persisted or queried."""


class MetricStore:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
    ) -> None:
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password

    def _connect(self) -> psycopg.Connection[Any]:
        if not self.password:
            raise MetricStoreError("指标数据库凭据尚未配置")
        try:
            return psycopg.connect(
                host=self.host,
                port=self.port,
                dbname=self.database,
                user=self.user,
                password=self.password,
                connect_timeout=5,
            )
        except psycopg.Error as exc:
            raise MetricStoreError("无法连接指标数据库") from exc

    @staticmethod
    def _bucket(timestamp: float, resolution_seconds: int) -> datetime:
        bucket_epoch = int(timestamp // resolution_seconds) * resolution_seconds
        return datetime.fromtimestamp(bucket_epoch, tz=timezone.utc)

    @staticmethod
    def _snapshot_values(snapshot: dict[str, Any]) -> tuple[float | None, ...]:
        cpu = snapshot.get("cpu") or {}
        memory = snapshot.get("memory") or {}
        disk = (snapshot.get("disks") or {}).get("data") or {}
        gpu = ((snapshot.get("gpus") or []) + [{}])[0]
        memory_total = float(gpu.get("memory_total_mib") or 0)
        memory_used = float(gpu.get("memory_used_mib") or 0)
        gpu_memory_percent = memory_used / memory_total * 100 if memory_total > 0 else None
        return (
            cpu.get("percent"),
            cpu.get("temperature_c"),
            snapshot.get("host_power_watts"),
            memory.get("percent"),
            disk.get("percent"),
            gpu.get("utilization_percent"),
            gpu_memory_percent,
            gpu.get("temperature_c"),
            gpu.get("power_watts"),
        )

    def record_snapshot(self, snapshot: dict[str, Any]) -> None:
        timestamp = float(snapshot.get("timestamp") or time.time())
        values = self._snapshot_values(snapshot)
        columns = (
            "cpu_percent",
            "cpu_temperature_c",
            "host_power_watts",
            "memory_percent",
            "disk_percent",
            "gpu_utilization_percent",
            "gpu_memory_percent",
            "gpu_temperature_c",
            "gpu_power_watts",
        )
        average_assignments = ",\n".join(
            f"""
                {column} = CASE
                    WHEN EXCLUDED.{column} IS NULL THEN hardware_metric_samples.{column}
                    WHEN hardware_metric_samples.{column} IS NULL THEN EXCLUDED.{column}
                    ELSE ((hardware_metric_samples.{column} * hardware_metric_samples.sample_count)
                         + EXCLUDED.{column})
                         / (hardware_metric_samples.sample_count + 1)
                END""".strip()
            for column in columns
        )
        statement = f"""
            INSERT INTO hardware_metric_samples (
                resolution_seconds, recorded_at, sample_count, {", ".join(columns)}
            )
            VALUES (%s, %s, 1, {", ".join(["%s"] * len(columns))})
            ON CONFLICT (resolution_seconds, recorded_at) DO UPDATE SET
                {average_assignments},
                sample_count = hardware_metric_samples.sample_count + 1
        """
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                for resolution_seconds in (5, 60, 900, 3600):
                    cursor.execute(
                        statement,
                        (
                            resolution_seconds,
                            self._bucket(timestamp, resolution_seconds),
                            *values,
                        ),
                    )
        except psycopg.Error as exc:
            raise MetricStoreError("无法保存硬件指标") from exc

    def prune(self) -> None:
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    """
                    DELETE FROM hardware_metric_samples
                    WHERE (resolution_seconds = 5 AND recorded_at < CURRENT_TIMESTAMP - INTERVAL '48 hours')
                       OR (resolution_seconds = 60 AND recorded_at < CURRENT_TIMESTAMP - INTERVAL '45 days')
                       OR (resolution_seconds = 900 AND recorded_at < CURRENT_TIMESTAMP - INTERVAL '400 days')
                       OR (resolution_seconds = 3600 AND recorded_at < CURRENT_TIMESTAMP - INTERVAL '1095 days')
                    """
                )
        except psycopg.Error as exc:
            raise MetricStoreError("无法清理过期硬件指标") from exc

    def history(self, metric: MetricName, range_name: MetricRange) -> dict[str, Any]:
        source_resolution, resolution_seconds, duration, resolution_label = RANGE_CONFIG[range_name]
        series = list(METRIC_SERIES[metric])
        columns = [item["key"] for item in series]
        start = datetime.now(timezone.utc) - duration
        bucket_expression = (
            "to_timestamp(floor(extract(epoch FROM recorded_at) "
            f"/ {resolution_seconds}) * {resolution_seconds})"
        )
        try:
            with self._connect() as connection, connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    f"""
                    SELECT {bucket_expression} AS recorded_at,
                           {", ".join(f"avg({column}) AS {column}" for column in columns)}
                    FROM hardware_metric_samples
                    WHERE resolution_seconds = %s
                      AND recorded_at >= %s
                    GROUP BY 1
                    ORDER BY 1
                    LIMIT 1200
                    """,
                    (source_resolution, start),
                )
                rows = cursor.fetchall()
                cursor.execute(
                    """
                    SELECT EXISTS (
                        SELECT 1 FROM hardware_metric_samples
                        WHERE host_power_watts IS NOT NULL
                    ) AS available
                    """
                )
                host_power_available = bool(cursor.fetchone()["available"])
        except psycopg.Error as exc:
            raise MetricStoreError("无法读取硬件指标历史") from exc

        if not host_power_available:
            series = [item for item in series if item["key"] != "host_power_watts"]
            columns = [item["key"] for item in series]
        points = [
            {
                "timestamp": row["recorded_at"],
                "values": {column: row[column] for column in columns},
            }
            for row in rows
        ]
        summaries: dict[str, dict[str, float | None]] = {}
        for column in columns:
            values = [float(row[column]) for row in rows if row[column] is not None]
            summaries[column] = {
                "latest": values[-1] if values else None,
                "average": sum(values) / len(values) if values else None,
                "minimum": min(values) if values else None,
                "maximum": max(values) if values else None,
            }
        return {
            "metric": metric,
            "range": range_name,
            "resolution_seconds": resolution_seconds,
            "resolution_label": resolution_label,
            "from": start,
            "to": datetime.now(timezone.utc),
            "series": series,
            "points": points,
            "summaries": summaries,
            "capabilities": {"host_power": host_power_available},
        }


class MetricCollector:
    def __init__(
        self,
        store: MetricStore,
        sampler: Callable[[], dict[str, Any]],
        *,
        interval_seconds: float = 5.0,
    ) -> None:
        self.store = store
        self.sampler = sampler
        self.interval_seconds = max(5.0, interval_seconds)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="omni-hardware-metrics",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds + 1.0))

    def _run(self) -> None:
        sample_number = 0
        failure_logged = False
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self.store.record_snapshot(self.sampler())
                sample_number += 1
                if sample_number % 12 == 0:
                    self.store.prune()
                if failure_logged:
                    LOGGER.info("hardware metric collection recovered")
                    failure_logged = False
            except Exception:
                if not failure_logged:
                    LOGGER.exception("hardware metric collection failed")
                    failure_logged = True
            elapsed = time.monotonic() - started
            self._stop.wait(max(0.1, self.interval_seconds - elapsed))
