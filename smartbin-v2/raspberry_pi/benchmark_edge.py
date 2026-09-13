"""
Edge Hardware Benchmark Suite for Raspberry Pi 5 and Jetson Nano.
Measures throughput (FPS), latency per stage, CPU utilization, RAM usage, and SoC thermal profile.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import psutil

from smartbin_v2.inference.engine import create_inference_engine
from smartbin_v2.utils.logger import get_logger, setup_logging

logger = get_logger("benchmark_edge")


def get_soc_temperature() -> float:
    """Read hardware SoC temperature in Celsius."""
    # Raspberry Pi / Linux thermal zone
    thermal_file = Path("/sys/class/thermal/thermal_zone0/temp")
    if thermal_file.exists():
        try:
            with open(thermal_file, "r") as f:
                return float(f.read().strip()) / 1000.0
        except Exception:
            pass

    # Windows / Generic fallback via psutil if available
    try:
        if hasattr(psutil, "sensors_temperatures"):
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    if entries:
                        return float(entries[0].current)
    except Exception:
        pass

    return 45.0  # Nominal default


def benchmark_model(
    model_path: str = "best.pt",
    backend: str = "pytorch",
    iterations: int = 100,
    imgsz: int = 640,
) -> Dict[str, Any]:
    """
    Benchmark an inference engine on current edge hardware.
    """
    setup_logging(level="INFO")
    logger.info("====================================================================")
    logger.info(f"Benchmarking Backend: {backend.upper()} | Model: {model_path}")
    logger.info(f"Target: {platform.machine()} ({platform.system()}) | Iterations: {iterations}")
    logger.info("====================================================================")

    engine = create_inference_engine(backend=backend, model_path=model_path)
    
    # Generate mock synthetic test frame
    dummy_frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # Warmup runs
    logger.info("Performing 10 warmup inference iterations...")
    for _ in range(10):
        _ = engine.infer(dummy_frame)

    latencies: List[float] = []
    cpu_usages: List[float] = []
    mem_usages: List[float] = []
    start_temp = get_soc_temperature()

    logger.info("Running benchmark measurement loops...")
    for i in range(iterations):
        t0 = time.perf_counter()
        _ = engine.infer(dummy_frame)
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000.0)  # ms
        cpu_usages.append(psutil.cpu_percent(interval=None))
        mem_usages.append(psutil.virtual_memory().used / (1024 * 1024))  # MB

    end_temp = get_soc_temperature()

    avg_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    fps = float(1000.0 / avg_latency) if avg_latency > 0 else 0.0
    avg_cpu = float(np.mean(cpu_usages))
    avg_ram = float(np.mean(mem_usages))

    results = {
        "hardware_platform": platform.machine(),
        "os": platform.system(),
        "backend": backend,
        "model_path": str(model_path),
        "input_size": [imgsz, imgsz],
        "iterations": iterations,
        "mean_latency_ms": round(avg_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "fps": round(fps, 2),
        "cpu_utilization_pct": round(avg_cpu, 1),
        "ram_usage_mb": round(avg_ram, 1),
        "start_temperature_c": round(start_temp, 1),
        "end_temperature_c": round(end_temp, 1),
        "temperature_delta_c": round(end_temp - start_temp, 1),
    }

    logger.info(f"--- Benchmark Results ({backend.upper()}) ---")
    logger.info(f"Throughput: {results['fps']} FPS | Latency: {results['mean_latency_ms']} ms (P95: {results['p95_latency_ms']} ms)")
    logger.info(f"CPU Usage: {results['cpu_utilization_pct']}% | RAM: {results['ram_usage_mb']} MB")
    logger.info(f"Temperature: {results['start_temperature_c']}C -> {results['end_temperature_c']}C")

    # Save to file
    out_dir = Path("smartbin-v2/docs/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / f"benchmark_{backend}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge Inference Benchmark")
    parser.add_argument("--model", type=str, default="best.pt")
    parser.add_argument("--backend", type=str, default="pytorch", choices=["pytorch", "onnx", "openvino", "tflite"])
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    benchmark_model(model_path=args.model, backend=args.backend, iterations=args.iterations)
