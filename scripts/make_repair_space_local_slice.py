#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair-space-file", required=True)
    parser.add_argument("--center-candidate-id", type=int, required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--step-norm", type=float, default=0.18)
    parser.add_argument("--max-radius-degrees", type=float, default=30.0)
    parser.add_argument("--radius-step-degrees", type=float, default=2.0)
    parser.add_argument("--azimuth-step-degrees", type=float, default=10.0)
    args = parser.parse_args()
    data = json.loads(Path(args.repair_space_file).read_text(encoding="utf-8"))
    center = np.asarray(data["candidates"][args.center_candidate_id]["coordinates"], dtype=np.float64)
    center /= np.linalg.norm(center)
    tangent = []
    for basis_index in range(len(center)):
        vector = np.zeros_like(center)
        vector[basis_index] = 1.0
        vector -= np.dot(vector, center) * center
        for previous in tangent:
            vector -= np.dot(vector, previous) * previous
        norm = np.linalg.norm(vector)
        if norm > 1e-8:
            tangent.append(vector / norm)
        if len(tangent) == 2:
            break
    candidates = [{
        "step_norm": args.step_norm,
        "strength_multiple": args.step_norm / 0.0006,
        "active_rank": len(center),
        "coordinates": center.tolist(),
        "selection_origins": [{"kind": "local_slice", "radius_degrees": 0.0, "azimuth_degrees": 0.0}],
    }]
    radii = np.arange(args.radius_step_degrees, args.max_radius_degrees + 1e-9, args.radius_step_degrees)
    azimuths = np.arange(0.0, 360.0, args.azimuth_step_degrees)
    for radius in radii:
        radius_radians = np.deg2rad(radius)
        for azimuth in azimuths:
            azimuth_radians = np.deg2rad(azimuth)
            local_tangent = np.cos(azimuth_radians) * tangent[0] + np.sin(azimuth_radians) * tangent[1]
            direction = np.cos(radius_radians) * center + np.sin(radius_radians) * local_tangent
            candidates.append({
                "step_norm": args.step_norm,
                "strength_multiple": args.step_norm / 0.0006,
                "active_rank": len(center),
                "coordinates": direction.tolist(),
                "selection_origins": [{
                    "kind": "local_slice",
                    "radius_degrees": float(radius),
                    "azimuth_degrees": float(azimuth),
                }],
            })
    output = {
        "format": "repair_space_candidate_input_v1",
        "source_file": args.repair_space_file,
        "center_candidate_id": args.center_candidate_id,
        "center_coordinates": center.tolist(),
        "tangent_axes": [value.tolist() for value in tangent],
        "candidates": candidates,
    }
    destination = Path(args.output_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(candidates)} candidates to {destination}")


if __name__ == "__main__":
    main()
