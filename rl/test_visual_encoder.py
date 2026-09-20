"""Unit tests for the visual-inertial encoder: shapes, state handling,
determinism, ego-motion warp correctness, and ablation flags."""

import torch

from rl.visual_encoder import VisualEncoder, VisualEncoderConfig


def make(**kw):
    torch.manual_seed(0)
    return VisualEncoder(VisualEncoderConfig(**kw))


def test_output_shapes():
    enc = make()
    state = enc.initial_state(3)
    frame = torch.rand(3, 3, *VisualEncoderConfig().input_hw)
    motion = torch.zeros(3, 5)
    action = torch.rand(3, 2)
    out, state = enc.update(frame, state, motion, action)
    assert out["h"].shape == (3, 64, 12, 16)
    assert out["g"].shape == (3, 128)
    assert out["f"].shape == (3, 64, 12, 16)
    assert out["mask"].shape == (3, 1, 12, 16)
    assert out["probe"].shape == (3, 4)

    assert out["occupancy"].shape == (3, 32)
    assert out["future_f"].shape == (3, 64, 6, 8)


def test_uint8_input():
    enc = make()
    state = enc.initial_state(1)
    frame = torch.randint(0, 256, (1, 3, *VisualEncoderConfig().input_hw), dtype=torch.uint8)
    out, _ = enc.update(frame, state, torch.zeros(1, 5), torch.zeros(1, 2))
    assert torch.isfinite(out["h"]).all()


def test_state_is_explicit():
    """Two sequences with independent state must not interact."""
    enc = make().eval()
    s1, s2 = enc.initial_state(1), enc.initial_state(1)
    f_a = torch.rand(1, 3, *VisualEncoderConfig().input_hw)
    f_b = torch.rand(1, 3, *VisualEncoderConfig().input_hw)
    m = torch.zeros(1, 5)
    a = torch.zeros(1, 2)
    with torch.no_grad():
        _, s1 = enc.update(f_a, s1, m, a)
        _, s2 = enc.update(f_b, s2, m, a)
        _, s1b = enc.update(f_a, s1, m, a)
        _, s1c = enc.update(f_a, s1, m, a)
    assert torch.equal(s1b["h"], s1c["h"])
    assert not torch.equal(s1["h"], s2["h"])


def test_determinism_eval():
    enc = make().eval()
    state = enc.initial_state(2)
    frame = torch.rand(2, 3, *VisualEncoderConfig().input_hw)
    m, a = torch.rand(2, 5), torch.rand(2, 2)
    with torch.no_grad():
        o1, _ = enc.update(frame, state, m, a)
        o2, _ = enc.update(frame, state, m, a)
    assert torch.equal(o1["h"], o2["h"]) and torch.equal(o1["g"], o2["g"])


def test_propagate_identity_at_init():
    """Zero-init flow => propagate is identity regardless of motion."""
    enc = make()
    state = enc.initial_state(1)
    state["h"] = torch.randn_like(state["h"])
    prop = enc.propagate(state, torch.randn(1, 5))
    assert torch.allclose(prop["h"], state["h"], atol=1e-6)
    assert (prop["mask"] == 1).all()


def test_warp_shifts_and_marks_invalid():
    """A known +1-cell x shift must move features and invalidate the edge."""
    enc = make()
    warp = enc.warp_net
    with torch.no_grad():
        warp.flow[-1].bias[1] = 1.0  # +1 cell in x
    state = enc.initial_state(1)
    state["h"] = torch.randn_like(state["h"])
    prop = enc.propagate(state, torch.zeros(1, 5))
    h, mask = prop["h"], prop["mask"]
    # cell (y,x) samples source (y, x+1): h[...,x] == h0[...,x+1]
    assert torch.allclose(h[..., :-1], state["h"][..., 1:], atol=1e-5)
    assert (mask[..., -1] == 0).all()  # last column sampled out of frame
    assert (mask[..., :-1] == 1).all()


def test_no_warp_flag():
    enc = make(warp=False)
    state = enc.initial_state(1)
    state["h"] = torch.randn_like(state["h"])
    prop = enc.propagate(state, torch.randn(1, 5))
    assert torch.equal(prop["h"], state["h"])
    assert (prop["mask"] == 1).all()


def test_frame_independent_variant():
    enc = make(temporal=False, warp=False, diff_signal=False)
    state = enc.initial_state(2)
    out, state = enc.update(
        torch.rand(2, 3, *VisualEncoderConfig().input_hw), state, torch.zeros(2, 5), torch.zeros(2, 2)
    )
    assert out["h"].shape == (2, 64, 12, 16)


def test_motion_conditioning_changes_output():
    enc = make().eval()
    state = enc.initial_state(1)
    state["h"] = torch.randn_like(state["h"])
    frame = torch.rand(1, 3, *VisualEncoderConfig().input_hw)
    a = torch.zeros(1, 2)
    with torch.no_grad():
        o1, _ = enc.update(frame, state, torch.zeros(1, 5), a)
        o2, _ = enc.update(frame, state, torch.randn(1, 5), a)
    # zero-init motion bias => identical at init; after perturbing the bias
    # layer the conditioning path must actually reach the output
    with torch.no_grad():
        enc.fusion.motion_bias.bias.normal_(0, 0.1)
        o3, _ = enc.update(frame, state, torch.randn(1, 5), a)
    assert torch.equal(o1["h"], o2["h"])  # sanity: zero-init
    assert not torch.equal(o2["h"], o3["h"])


def test_forward_sequence():
    enc = make()
    frames = torch.rand(2, 5, 3, *VisualEncoderConfig().input_hw)
    motions = torch.zeros(2, 5, 5)
    actions = torch.rand(2, 5, 2)
    outs, state = enc.forward_sequence(frames, motions, actions)
    assert outs["h"].shape == (2, 5, 64, 12, 16)
    assert outs["g"].shape == (2, 5, 128)
    assert state["h"].shape == (2, 64, 12, 16)


def test_global_temporal_variant():
    enc = make(global_temporal=True)
    state = enc.initial_state(2)
    assert "g_hist" in state
    for _ in range(3):
        out, state = enc.update(
            torch.rand(2, 3, *VisualEncoderConfig().input_hw), state, torch.zeros(2, 5), torch.zeros(2, 2)
        )
    assert out["g"].shape == (2, 128)
    assert state["g_hist"].shape == (2, 8, 128)


if __name__ == "__main__":
    for name, fn in sorted(
        {k: v for k, v in globals().items() if k.startswith("test_")}.items()
    ):
        fn()
        print(f"ok {name}")
