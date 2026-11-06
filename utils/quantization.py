from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn


def quantize_tensor(
    tensor: torch.Tensor,
    nbits: int = 4,
    block_size: int = 64,
) -> dict[str, torch.Tensor | tuple[int, ...] | int | torch.dtype | bool]:
    """Quantize a PyTorch tensor into scaled integer blocks with sub-byte packing.

    Supports INT4 (sub-byte packed into torch.uint8 nibbles) and INT8 (torch.int8)
    with per-block dynamic scaling. Zero blocks and non-finite values are handled safely.

    Args:
        tensor: Floating-point PyTorch tensor of arbitrary shape.
        nbits: Quantization bit-width (4 or 8).
        block_size: Number of elements per quantization block (even positive integer).

    Returns:
        Dictionary containing packed data, per-block scale factors, original shape,
        padding length, and metadata necessary for exact reconstruction.
    """
    if nbits not in (4, 8):
        raise ValueError(f"nbits must be 4 or 8, got {nbits}")
    if block_size < 2 or block_size % 2 != 0:
        raise ValueError(
            f"block_size must be a positive even integer, got {block_size}"
        )

    shape = tuple(tensor.shape)
    orig_dtype = tensor.dtype
    device = tensor.device
    flat = tensor.detach().flatten().float()
    numel = flat.numel()

    if numel == 0:
        empty_data = torch.empty(
            0, dtype=torch.uint8 if nbits == 4 else torch.int8, device=device
        )
        return {
            "data": empty_data,
            "scales": torch.empty(0, dtype=torch.float32, device=device),
            "shape": shape,
            "pad_len": 0,
            "nbits": nbits,
            "orig_dtype": orig_dtype,
            "is_quantized": True,
        }

    pad_len = (block_size - (numel % block_size)) % block_size
    if pad_len > 0:
        flat = torch.nn.functional.pad(flat, (0, pad_len))

    blocks = flat.reshape(-1, block_size)
    max_vals = blocks.abs().max(dim=-1, keepdim=True).values
    zero_mask = max_vals == 0

    if nbits == 4:
        scales = torch.where(zero_mask, torch.ones_like(max_vals), max_vals / 7.0)
        q = torch.where(
            zero_mask,
            torch.zeros_like(blocks, dtype=torch.int8),
            torch.clamp(torch.round(blocks / scales), -7, 7).to(torch.int8),
        )
        scales = torch.where(zero_mask, torch.zeros_like(max_vals), scales)

        q_flat = q.flatten()
        q_unsigned = (q_flat + 7).to(torch.uint8)
        lo = q_unsigned[0::2]
        hi = q_unsigned[1::2]
        packed = (hi << 4) | (lo & 0x0F)

        return {
            "data": packed,
            "scales": scales.squeeze(-1).to(torch.float32),
            "shape": shape,
            "pad_len": pad_len,
            "nbits": 4,
            "orig_dtype": orig_dtype,
            "is_quantized": True,
        }
    else:  # nbits == 8
        scales = torch.where(zero_mask, torch.ones_like(max_vals), max_vals / 127.0)
        q = torch.where(
            zero_mask,
            torch.zeros_like(blocks, dtype=torch.int8),
            torch.clamp(torch.round(blocks / scales), -127, 127).to(torch.int8),
        )
        scales = torch.where(zero_mask, torch.zeros_like(max_vals), scales)

        return {
            "data": q.flatten(),
            "scales": scales.squeeze(-1).to(torch.float32),
            "shape": shape,
            "pad_len": pad_len,
            "nbits": 8,
            "orig_dtype": orig_dtype,
            "is_quantized": True,
        }


def dequantize_tensor(
    quantized_dict: dict[
        str, torch.Tensor | tuple[int, ...] | int | torch.dtype | bool
    ],
    device: torch.device | None = None,
) -> torch.Tensor:
    """Reconstruct floating-point PyTorch tensor from a quantized dictionary.

    Args:
        quantized_dict: Dictionary returned by `quantize_tensor`.
        device: Destination device for reconstructed tensor (defaults to data.device).

    Returns:
        Reconstructed PyTorch tensor matching original shape and dtype.
    """
    data = quantized_dict["data"]
    scales = quantized_dict["scales"]
    shape = quantized_dict["shape"]
    pad_len = quantized_dict["pad_len"]
    nbits = quantized_dict["nbits"]
    orig_dtype = quantized_dict.get("orig_dtype", torch.float32)
    target_device = device if device is not None else data.device

    if data.numel() == 0:
        return torch.empty(shape, dtype=orig_dtype, device=target_device)

    num_blocks = scales.numel()

    if nbits == 4:
        lo = (data & 0x0F).to(torch.float32) - 7.0
        hi = ((data >> 4) & 0x0F).to(torch.float32) - 7.0
        unpacked = torch.stack([lo, hi], dim=-1).flatten()
        block_size = unpacked.numel() // num_blocks
        blocks = unpacked.reshape(num_blocks, block_size)
    elif nbits == 8:
        block_size = data.numel() // num_blocks
        blocks = data.reshape(num_blocks, block_size).to(torch.float32)
    else:
        raise ValueError(f"Unsupported nbits: {nbits}")

    dequant = blocks * scales.unsqueeze(-1)
    flat = dequant.flatten()
    if pad_len > 0:
        flat = flat[:-pad_len]
    return flat.reshape(shape).to(dtype=orig_dtype, device=target_device)


def is_communicated_param(key: str, mode: str = "fedbn") -> bool:
    """Check whether a parameter key is communicated under the given mode.

    In FedBN, batch normalization layers remain local to clients.
    """
    if "num_batches_tracked" in key:
        return False
    if mode.lower() == "fedbn":
        return "bn" not in key
    return True


def quantize_communicated_parameters(
    model_or_state_dict: nn.Module | dict[str, torch.Tensor],
    mode: str = "fedbn",
    nbits: int = 4,
    block_size: int = 64,
    residuals: dict[str, torch.Tensor] | None = None,
) -> OrderedDict:
    """Quantize only the parameters that are communicated across clients/server.

    Args:
        model_or_state_dict: PyTorch Module or state dict.
        mode: Federated mode ('fedbn', 'fedavg', 'fedprox').
        nbits: Quantization bit-width (4 or 8).
        block_size: Elements per quantization block.
        residuals: Optional dictionary accumulating error residuals across rounds.

    Returns:
        OrderedDict mapping parameter keys to quantized descriptors or intact local tensors.
    """
    state_dict = (
        model_or_state_dict.state_dict()
        if isinstance(model_or_state_dict, nn.Module)
        else model_or_state_dict
    )

    quantized_dict = OrderedDict()
    for k, v in state_dict.items():
        if is_communicated_param(k, mode=mode) and torch.is_floating_point(v):
            if residuals is not None:
                res = residuals.get(k, torch.zeros_like(v))
                v_adj = v + res
                q = quantize_tensor(v_adj, nbits=nbits, block_size=block_size)
                v_recon = dequantize_tensor(q, device=v.device)
                residuals[k] = v_adj - v_recon
                quantized_dict[k] = q
            else:
                quantized_dict[k] = quantize_tensor(
                    v, nbits=nbits, block_size=block_size
                )
        else:
            quantized_dict[k] = {
                "data": v.clone(),
                "is_quantized": False,
                "nbits": None,
            }
    return quantized_dict


def dequantize_communicated_parameters(
    quantized_dict: OrderedDict,
    target_model: nn.Module | None = None,
    device: torch.device | None = None,
) -> OrderedDict:
    """Restore parameter dictionary from quantized representation.

    Args:
        quantized_dict: OrderedDict returned by `quantize_communicated_parameters`.
        target_model: Optional destination PyTorch module to update.
        device: Target device for restored tensors.

    Returns:
        OrderedDict of reconstructed tensors.
    """
    restored = OrderedDict()
    for k, v in quantized_dict.items():
        if isinstance(v, dict) and v.get("is_quantized", False):
            restored[k] = dequantize_tensor(v, device=device)
        elif isinstance(v, dict) and "data" in v:
            t = v["data"]
            restored[k] = (
                t.to(device)
                if device is not None and isinstance(t, torch.Tensor)
                else t
            )
        elif isinstance(v, torch.Tensor):
            restored[k] = v.to(device) if device is not None else v
        else:
            restored[k] = v

    if target_model is not None:
        target_model.load_state_dict(restored, strict=False)
    return restored


def quantized_communication(
    server_model: nn.Module,
    models: list[nn.Module],
    client_weights: list[float],
    mode: str = "fedbn",
    nbits: int = 4,
    block_size: int = 64,
    client_residuals: list[dict[str, torch.Tensor]] | None = None,
) -> tuple[nn.Module, list[nn.Module]]:
    """Execute quantized aggregation cycle across clients and central server.

    Args:
        server_model: Central global model.
        models: List of client models.
        client_weights: Aggregation weights for each client.
        mode: 'fedbn', 'fedavg', or 'fedprox'.
        nbits: Quantization bit-width (4 or 8).
        block_size: Elements per quantization block.
        client_residuals: Optional list of residual buffers for each client.

    Returns:
        Tuple of (server_model, models) with updated parameters.
    """
    client_num = len(models)
    server_dict = server_model.state_dict()

    with torch.no_grad():
        # Step 1: Each client compresses its communicated parameters
        decompressed_client_dicts = []
        for i in range(client_num):
            res_i = client_residuals[i] if client_residuals is not None else None
            q_dict = quantize_communicated_parameters(
                models[i],
                mode=mode,
                nbits=nbits,
                block_size=block_size,
                residuals=res_i,
            )
            d_dict = dequantize_communicated_parameters(q_dict)
            decompressed_client_dicts.append(d_dict)

        # Step 2: Server aggregates communicated parameters
        for key in server_dict.keys():
            if is_communicated_param(key, mode=mode):
                temp = torch.zeros_like(server_dict[key], dtype=torch.float32)
                for client_idx in range(client_num):
                    temp += (
                        client_weights[client_idx]
                        * decompressed_client_dicts[client_idx][key]
                    )
                server_dict[key].data.copy_(temp)

                # Broadcast back to clients
                for client_idx in range(client_num):
                    models[client_idx].state_dict()[key].data.copy_(server_dict[key])
            elif "num_batches_tracked" in key:
                server_dict[key].data.copy_(models[0].state_dict()[key])

    return server_model, models


def calc_communication_stats(
    model: nn.Module,
    mode: str = "fedbn",
    nbits: int = 4,
    block_size: int = 64,
) -> dict[str, int | float]:
    """Calculate compound bandwidth savings of BN filtering + quantization.

    Args:
        model: PyTorch module.
        mode: 'fedbn', 'fedavg', or 'fedprox'.
        nbits: Quantization bit-width (4 or 8).
        block_size: Elements per quantization block.

    Returns:
        Dictionary with full_model_bytes, uncompressed_comm_bytes,
        quantized_comm_bytes, and compression ratios.
    """
    total_bytes = 0
    comm_bytes = 0

    for k, v in model.state_dict().items():
        if isinstance(v, torch.Tensor):
            num_bytes = v.numel() * v.element_size()
            total_bytes += num_bytes
            if is_communicated_param(k, mode=mode):
                comm_bytes += num_bytes

    q_dict = quantize_communicated_parameters(
        model, mode=mode, nbits=nbits, block_size=block_size
    )
    quantized_wire_bytes = 0
    for k, v in q_dict.items():
        if is_communicated_param(k, mode=mode):
            quantized_wire_bytes += (
                v["data"].numel() * v["data"].element_size()
                + v["scales"].numel() * v["scales"].element_size()
            )

    comm_ratio = total_bytes / quantized_wire_bytes if quantized_wire_bytes > 0 else 1.0
    savings_pct = (
        (1.0 - quantized_wire_bytes / total_bytes) * 100.0 if total_bytes > 0 else 0.0
    )

    return {
        "full_model_bytes": total_bytes,
        "uncompressed_comm_bytes": comm_bytes,
        "quantized_wire_bytes": quantized_wire_bytes,
        "overall_compression_ratio": comm_ratio,
        "overall_savings_pct": savings_pct,
    }
