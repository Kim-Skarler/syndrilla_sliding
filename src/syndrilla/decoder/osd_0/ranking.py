import torch

RANKING_METHODS = ("standard", "hardware_user", "hardware_lzc")


def ranking_key(llr, method, int_width, frac_width):
    """Per-column key whose stable ascending sort is the OSD column order.

    standard returns llr itself. The hardware methods first round llr to the signed
    fixed-point grid Q(int_width).(frac_width), v = round(llr * 2^frac_width) clamped
    to [-2^(int_width + frac_width), 2^(int_width + frac_width) - 1], then grade it
    coarsely; columns of equal grade keep their index order. Negative values always
    come before nonnegative ones.
        hardware_user: a bit-serial minimum search over the offset value v + 2^W
            (W = int_width + frac_width), run separately on the negative and the
            nonnegative columns. From the top bit down, each step drops the columns
            whose bit is above the minimum bit of the columns still in; the grade
            is the number of steps a column stays in, and a longer stay comes first.
        hardware_lzc: the bit length of |v| (0 for v = 0); the largest bit length
            comes first among negative values and the smallest among the others."""
    if method == "standard":
        return llr
    if method not in RANKING_METHODS:
        raise ValueError(f"ranking_method must be one of {RANKING_METHODS}, got {method!r}")
    bits = int_width + frac_width
    limit = 1 << bits
    v = torch.round(llr * float(1 << frac_width)).long().clamp(-limit, limit - 1)
    neg = v < 0
    if method == "hardware_lzc":
        # bit length by shifts: CUDA's log2 returns 5.999... for 64 (and 2.999... for 8),
        # so floor(log2) would put those values one grade too low on the GPU
        mag = v.abs()
        length = torch.zeros_like(v)
        for pos in range(bits + 1):
            length += (mag >> pos) != 0
        return torch.where(neg, -(bits + 2) - length, length).to(llr.dtype)

    q = v + limit

    def stay(dead):
        out = dead.clone()
        steps = torch.zeros_like(q)
        for pos in range(bits, -1, -1):
            bit = (q >> pos) & 1
            low = bit.masked_fill(out, 1).min(dim=1, keepdim=True).values
            out |= bit > low
            steps += (~out).long()
        return steps

    return torch.where(neg, -(bits + 2) - stay(~neg), -stay(neg)).to(llr.dtype)
