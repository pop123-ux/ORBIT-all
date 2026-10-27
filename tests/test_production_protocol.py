from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "production_optimizer_protocol.json"


def load_protocol() -> dict:
    return json.loads(PROTOCOL.read_text())


def test_v1_protocol_keeps_orbit_all_frozen() -> None:
    protocol = load_protocol()
    assert protocol["name"] == "ORBIT-all-production-optimizer-bench-v1"
    assert protocol["grids"]["orbit_all"] == [
        {
            "id": "orbit-all-frozen-shared-03",
            "matrix_lr": 0.018,
            "aux_lr": 0.0085,
            "weight_decay": 0.0045,
        }
    ]


def test_v1_production_baselines_get_nine_candidates() -> None:
    protocol = load_protocol()
    for name in protocol["optimizers"]:
        grid = protocol["grids"][name]
        if name == "orbit_all":
            assert len(grid) == 1
        else:
            assert len(grid) == 9
            assert len({candidate["id"] for candidate in grid}) == 9


def test_moonshot_v1_grid_uses_shared_base_lr_for_aux_path() -> None:
    protocol = load_protocol()
    for name in ("muon_moonlight", "muonclip_kimi"):
        for candidate in protocol["grids"][name]:
            assert candidate["matrix_lr"] == candidate["aux_lr"]


def test_confirmation_seeds_remain_unseen_reserved_set() -> None:
    protocol = load_protocol()
    assert protocol["phases"]["confirm"]["seeds"] == list(range(910, 920))
    assert "no seeds 910-919 were evaluated" in protocol["pilot_history"]["v0_outcome"]
