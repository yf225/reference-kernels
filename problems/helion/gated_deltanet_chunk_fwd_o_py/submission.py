from task import input_t, output_t

import torch
import helion
import helion.language as hl


@helion.kernel(
    static_shapes=True,
    dot_precision="ieee",
    config=helion.Config(block_sizes=[], num_warps=8, num_stages=2),
)
def gated_chunk_attn(
    q: torch.Tensor,     # [B, T, H, K]
    k: torch.Tensor,     # [B, T, H, K]
    v: torch.Tensor,     # [B, T, H, V]
    h: torch.Tensor,     # [B, NT, H, K, V]
    g: torch.Tensor,     # [B, T, H]
    scale: float,
) -> torch.Tensor:
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = 64
    K = hl.specialize(K)
    V = hl.specialize(V)

    out = torch.empty_like(v)

    BH = B * H
    for flat_bh, tile_t in hl.tile([BH, T], block_size=[1, C]):
        b_idx = flat_bh.begin // H
        h_idx = flat_bh.begin % H
        c_idx = tile_t.begin // C

        g_vals = g[b_idx, tile_t, h_idx]
        q_s = q[b_idx, tile_t, h_idx, :] * torch.exp(g_vals)[:, None]
        k_s = k[b_idx, tile_t, h_idx, :] * torch.exp(-g_vals)[:, None]

        sim = hl.dot(q_s, k_s.T)
        idx = hl.arange(tile_t.block_size)
        mask = idx[:, None] >= idx[None, :]
        sim = torch.where(mask, sim, 0.0)
        local_out = hl.dot(sim.to(v.dtype), v[b_idx, tile_t, h_idx, :])

        global_out = hl.dot(q_s, h[b_idx, c_idx, h_idx, :, :])

        out[b_idx, tile_t, h_idx, :] = ((global_out + local_out) * scale).to(out.dtype)

    return out


def custom_kernel(data: input_t) -> output_t:
    q, k, v_new, h, g = data
    scale = q.shape[-1] ** -0.5
    return gated_chunk_attn(q, k, v_new, h, g, scale)
