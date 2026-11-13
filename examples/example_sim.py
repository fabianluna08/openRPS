#!/usr/bin/env python3
"""
OpenRPS - Example Multi-Cell & Multipath Propagation Simulation
"""

import os
import numpy as np


from open_rps import (
    PropagationSimulator,
    Transmitter,
    PlanarReflector,
    MapVisualizer,
)


def run_demo():
    print("==========================================================")
    print("      OpenRPS - Radiowave Propagation Simulator Demo      ")
    print("==========================================================")

    sim = PropagationSimulator()

    # 1. Setup Base Stations (Transmitters)
    # Tx 0: Macrocell Base Station 1 (at x=200m, y=250m, height=25m, 43 dBm / 20W, 2.4 GHz)
    tx0 = Transmitter(tx_id=0, x=200.0, y=250.0, z=25.0, tx_power_dbm=43.0, frequency_hz=2.4e9, antenna_gain_dbi=12.0)
    # Tx 1: Macrocell Base Station 2 (at x=800m, y=750m, height=25m, 40 dBm / 10W, 2.4 GHz)
    tx1 = Transmitter(tx_id=1, x=800.0, y=750.0, z=25.0, tx_power_dbm=40.0, frequency_hz=2.4e9, antenna_gain_dbi=12.0)

    sim.add_transmitter(tx0)
    sim.add_transmitter(tx1)
    print(f"[+] Added 2 Transmitters: Tx0 at ({tx0.x}, {tx0.y}), Tx1 at ({tx1.x}, {tx1.y})")

    # 2. Setup Environment Reflectors (e.g., a large concrete building facade)
    wall1 = PlanarReflector(
        p0=[500.0, 500.0, 0.0],
        normal=[1.0, 0.0, 0.0], # Facing East/West
        reflectivity=0.6,
        name="Building Facade A"
    )
    sim.add_reflector(wall1)
    print(f"[+] Added Planar Reflector: {wall1.name} at {wall1.p0.tolist()}")

    # 3. Execute 3D Ray-Tracing Simulation over 1000m x 1000m grid
    print("\n[+] Running 3D Ray-Tracing Multipath Simulation (1000m x 1000m)...")
    res_ray = sim.run_raytracing_simulation(
        x_range=(0.0, 1000.0),
        y_range=(0.0, 1000.0),
        resolution_m=25.0, # 41x41 grid
        rx_height_m=1.5
    )

    # 4. Display ASCII Heatmap of Received Power
    print("\n" + MapVisualizer.render_ascii_heatmap(
        res_ray.max_rx_power_dbm,
        title="Rx Power (P_r) Coverage Map",
        val_range=None,
        unit="dBm"
    ))

    # 5. Display ASCII Heatmap of RMS Delay Spread
    print("\n" + MapVisualizer.render_ascii_heatmap(
        res_ray.rms_delay_spread_ns,
        title="RMS Delay Spread (sigma_tau)",
        val_range=None,
        unit="ns"
    ))

    # 6. Statistics
    print("\n--- Simulation Summary ---")
    print(f"Max Received Power:  {np.nanmax(res_ray.max_rx_power_dbm):.2f} dBm")
    print(f"Min Received Power:  {np.nanmin(res_ray.max_rx_power_dbm):.2f} dBm")
    print(f"Mean Received Power: {np.nanmean(res_ray.max_rx_power_dbm):.2f} dBm")
    print(f"Mean Delay Spread:   {np.nanmean(res_ray.rms_delay_spread_ns):.2f} ns")
    print(f"Max Delay Spread:    {np.nanmax(res_ray.rms_delay_spread_ns):.2f} ns")
    print(f"Mean Angular Spread: {np.nanmean(res_ray.angular_spread_bs_phi):.2f} deg")

    # 7. Export outputs
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "simulation_results.csv")
    json_path = os.path.join(output_dir, "simulation_summary.json")
    svg_power_path = os.path.join(output_dir, "coverage_rx_power.svg")
    svg_ds_path = os.path.join(output_dir, "coverage_delay_spread.svg")
    svg_server_path = os.path.join(output_dir, "coverage_best_server.svg")

    res_ray.export_csv(csv_path)
    res_ray.export_json(json_path)
    MapVisualizer.export_svg(res_ray, svg_power_path, metric="rx_power")
    MapVisualizer.export_svg(res_ray, svg_ds_path, metric="delay_spread")
    MapVisualizer.export_svg(res_ray, svg_server_path, metric="best_server")

    print("\n[+] Exported files:")
    print(f"    - CSV Table:      {csv_path}")
    print(f"    - JSON Summary:   {json_path}")
    print(f"    - Rx Power SVG:   {svg_power_path}")
    print(f"    - Delay Spread:   {svg_ds_path}")
    print(f"    - Best Server ID: {svg_server_path}")
    print("\nSimulation completed successfully!")


if __name__ == "__main__":
    run_demo()
