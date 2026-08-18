import numpy as np

from core_shell.physics import eta_from_alpha_beta, make_materials_from_alpha_beta, total_internal_energy
from core_shell.regimes import get_regime


def test_eta_symmetry():
    assert np.isclose(eta_from_alpha_beta(1.0, 1.0), 0.0)


def test_material_regimes():
    for name in ("density", "mixed", "velocity"):
        regime = get_regime(name)
        materials = make_materials_from_alpha_beta(
            0.01, 0.02, contrast_in=regime.contrast_in
        )
        assert len(materials) == 6
        assert np.all(np.isfinite(materials))


def test_energy_smoke():
    regime = get_regime("density")
    materials = make_materials_from_alpha_beta(
        0.01, 0.02, contrast_in=regime.contrast_in
    )
    energy = total_internal_energy(
        np.array([1e-3, 2e-3]), 0.5, *materials, lmax=2
    )
    assert energy.shape == (2,)
    assert np.all(np.isfinite(energy))


def test_mixed_reference():
    from core_shell.models import sphere_reference
    alpha = np.array([0.01, 1.0, 4.0])
    expected = np.sqrt(3.0) * alpha ** 0.75
    assert np.allclose(
        sphere_reference(alpha, "mixed", mix_factor=0.5),
        expected,
    )


def test_manual_correction_tables_are_present():
    from core_shell.manual_corrections import MANUAL_CORRECTIONS

    mixed = MANUAL_CORRECTIONS["mixed"]
    velocity = MANUAL_CORRECTIONS["velocity"]

    assert sum(len(v) for v in mixed["remover_D0"].values()) == 86
    assert sum(len(v) for v in mixed["remover_picos"].values()) == 1256
    assert len(mixed["remover_todos_os_picos"]) == 193
    assert len(mixed["inserir_picos"]) == 0

    assert sum(len(v) for v in velocity["remover_D0"].values()) == 2
    assert sum(len(v) for v in velocity["remover_picos"].values()) == 574
    assert len(velocity["remover_todos_os_picos"]) == 37
    assert len(velocity["inserir_picos"]) == 4


def test_repository_variants_are_separated():
    from core_shell.checkpoints import repository_variant

    assert repository_variant("density", None) == "final"
    assert repository_variant("mixed", None) == "manual_corrected"
    assert repository_variant("mixed", "raw") == "raw"
    assert repository_variant("velocity", None) == "manual_corrected"
    assert repository_variant("velocity", "raw") == "raw"
