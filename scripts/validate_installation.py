#!/usr/bin/env python
import sys

import numpy as np

from core_shell.detection import global_nonuniform_mesh
from core_shell.physics import make_materials_from_alpha_beta, total_internal_energy
from core_shell.regimes import get_regime


def main():
    x = global_nonuniform_mesh(
        x_min=1e-4,
        x_max=0.01,
        x_split=0.005,
        n_low=20,
        n_high=20,
    )

    for regime_name in ("density", "mixed", "velocity"):
        regime = get_regime(regime_name)
        materials = make_materials_from_alpha_beta(
            0.01,
            0.02,
            contrast_in=regime.contrast_in,
            mix_factor=regime.mix_factor,
        )
        energy = total_internal_energy(
            x,
            0.5,
            *materials,
            lmax=2,
        )

        if not np.all(np.isfinite(energy)):
            raise RuntimeError(
                f"Non-finite energy values for regime={regime_name}."
            )

        print(
            f"{regime_name:8s}: OK | "
            f"{len(x)} points | "
            f"energy range = [{energy.min():.3e}, {energy.max():.3e}]"
        )

    print("\nLocal installation validation passed.")


if __name__ == "__main__":
    main()
