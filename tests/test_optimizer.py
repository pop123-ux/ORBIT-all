import torch

from orbit_all import ModelConfig, OrbitAll, ResearchLM


def tiny_config(**kwargs):
    base = dict(
        vocab_size=128,
        block_size=16,
        n_layer=1,
        n_head=4,
        n_kv_head=2,
        n_embd=32,
        mlp_hidden=64,
        qk_norm=True,
        rope_mode="learned_log_frequency",
        moe_experts=0,
    )
    base.update(kwargs)
    return ModelConfig(**base)


def run_one_step(mode: str, config: ModelConfig):
    torch.manual_seed(0)
    model = ResearchLM(config)
    opt = OrbitAll(
        model,
        mode=mode,
        matrix_lr=0.01,
        aux_lr=0.003,
        weight_decay=0.001,
    )
    tokens = torch.randint(0, config.vocab_size, (2, config.block_size))
    loss = model(tokens, tokens).loss
    loss.backward()
    before = [p.detach().clone() for p in model.parameters()]
    opt.step()
    opt.zero_grad(set_to_none=True)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    return model, opt


def test_muon_control_runs():
    run_one_step("muon", tiny_config(rope_mode="fixed"))


def test_orbit_all_runs_with_gqa_qknorm_and_learned_rope():
    model, opt = run_one_step("orbit_all", tiny_config())
    assert model.blocks[0].attn.q_second.shape == (4, 4, 2, 2)
    assert model.blocks[0].attn.k_second.shape == (2, 4, 2, 2)
    assert opt.families == {"qk", "vo", "swiglu", "embedding", "rope", "moe"}


def test_family_ablation_modes_run():
    for mode in (
        "orbit_qk",
        "orbit_qk_vo",
        "orbit_qk_vo_swiglu",
        "orbit_all_minus_qk",
        "orbit_all_minus_vo",
        "orbit_all_minus_swiglu",
        "orbit_all_minus_embedding",
    ):
        run_one_step(mode, tiny_config(rope_mode="fixed"))


def test_moe_router_and_expert_geometry_runs():
    config = tiny_config(moe_experts=2, rope_mode="fixed")
    model, opt = run_one_step("orbit_all", config)
    moe = model.blocks[0].ffn
    assert torch.isfinite(moe.router_fisher).all()
    assert torch.isclose(moe.expert_load.sum(), torch.tensor(1.0), atol=1e-5)
    assert "moe" in opt.families
