import os
import sys
import unittest

import torch

# Ensure repo root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nets.models import DigitModel
from utils.quantization import (
    calc_communication_stats,
    dequantize_communicated_parameters,
    dequantize_tensor,
    is_communicated_param,
    quantize_communicated_parameters,
    quantize_tensor,
    quantized_communication,
)


class TestFedBNQuantization(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_quantize_dequantize_tensor_int4(self):
        for shape in [(64,), (32, 32), (16, 3, 5, 5), (7, 13, 11)]:
            tensor = torch.randn(shape, dtype=torch.float32)
            q_dict = quantize_tensor(tensor, nbits=4, block_size=32)
            recon = dequantize_tensor(q_dict)

            self.assertEqual(recon.shape, tensor.shape)
            self.assertEqual(recon.dtype, tensor.dtype)
            cos_sim = torch.nn.functional.cosine_similarity(
                tensor.flatten(), recon.flatten(), dim=0
            )
            self.assertGreater(cos_sim.item(), 0.98)

    def test_quantize_dequantize_tensor_int8(self):
        tensor = torch.randn(64, 64, dtype=torch.float32)
        q_dict = quantize_tensor(tensor, nbits=8, block_size=64)
        recon = dequantize_tensor(q_dict)

        self.assertEqual(recon.shape, tensor.shape)
        cos_sim = torch.nn.functional.cosine_similarity(
            tensor.flatten(), recon.flatten(), dim=0
        )
        self.assertGreater(cos_sim.item(), 0.999)

    def test_zero_and_empty_tensor(self):
        zero_tensor = torch.zeros(128, dtype=torch.float32)
        q_dict = quantize_tensor(zero_tensor, nbits=4, block_size=32)
        recon = dequantize_tensor(q_dict)

        self.assertTrue(torch.equal(zero_tensor, recon))
        self.assertFalse(torch.isnan(recon).any())
        self.assertFalse(torch.isinf(recon).any())

        empty_tensor = torch.empty((0, 5), dtype=torch.float32)
        q_empty = quantize_tensor(empty_tensor, nbits=4, block_size=32)
        recon_empty = dequantize_tensor(q_empty)
        self.assertEqual(recon_empty.numel(), 0)

    def test_sub_byte_bit_packing(self):
        numel = 256
        tensor = torch.randn(numel, dtype=torch.float32)
        q_dict = quantize_tensor(tensor, nbits=4, block_size=64)

        self.assertEqual(q_dict["data"].dtype, torch.uint8)
        self.assertEqual(q_dict["data"].numel(), numel // 2)

    def test_is_communicated_param(self):
        self.assertTrue(is_communicated_param("conv1.weight", mode="fedbn"))
        self.assertTrue(is_communicated_param("fc1.bias", mode="fedbn"))
        self.assertFalse(is_communicated_param("bn1.weight", mode="fedbn"))
        self.assertFalse(is_communicated_param("bn1.running_mean", mode="fedbn"))
        self.assertFalse(is_communicated_param("bn1.num_batches_tracked", mode="fedbn"))

        # In FedAvg mode, BN weights are communicated
        self.assertTrue(is_communicated_param("bn1.weight", mode="fedavg"))

    def test_quantize_communicated_parameters_fedbn(self):
        model = DigitModel()
        q_dict = quantize_communicated_parameters(
            model, mode="fedbn", nbits=4, block_size=32
        )
        restored = dequantize_communicated_parameters(q_dict)

        for k in model.state_dict():
            self.assertEqual(restored[k].shape, model.state_dict()[k].shape)
            if "bn" in k:
                # BN parameters must be preserved identically
                self.assertTrue(torch.equal(model.state_dict()[k], restored[k]))
            elif torch.is_floating_point(model.state_dict()[k]):
                cos = torch.nn.functional.cosine_similarity(
                    model.state_dict()[k].flatten(),
                    restored[k].flatten(),
                    dim=0,
                )
                self.assertGreater(cos.item(), 0.98)

    def test_quantized_communication_cycle(self):
        server_model = DigitModel()
        client_models = [DigitModel() for _ in range(3)]
        weights = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]

        # Give client 0 a distinct BN state
        with torch.no_grad():
            client_models[0].bn1.weight.fill_(5.0)

        quantized_communication(
            server_model,
            client_models,
            weights,
            mode="fedbn",
            nbits=4,
            block_size=64,
        )

        # BN1 weight of client 0 should still be 5.0 (local BN preservation)
        self.assertEqual(client_models[0].bn1.weight[0].item(), 5.0)
        # Conv1 weight should be synchronized across clients
        self.assertTrue(
            torch.allclose(
                client_models[0].conv1.weight,
                client_models[1].conv1.weight,
                atol=1e-5,
            )
        )

    def test_error_feedback_residuals(self):
        residuals = {}
        true_accum = torch.zeros(64)
        quant_accum = torch.zeros(64)

        for _ in range(10):
            step_delta = torch.randn(64) * 0.05
            true_accum += step_delta

            p_dict = {"conv1.weight": step_delta}
            q_dict = quantize_communicated_parameters(
                p_dict,
                mode="fedavg",
                nbits=4,
                block_size=32,
                residuals=residuals,
            )
            recon = dequantize_communicated_parameters(q_dict)
            quant_accum += recon["conv1.weight"]

        drift = torch.norm(true_accum - quant_accum).item()
        self.assertLess(drift, 0.05)

    def test_calc_communication_stats(self):
        model = DigitModel()
        stats = calc_communication_stats(model, mode="fedbn", nbits=4, block_size=64)

        self.assertGreater(stats["overall_compression_ratio"], 6.0)
        self.assertGreater(stats["overall_savings_pct"], 80.0)
        self.assertGreater(stats["full_model_bytes"], stats["quantized_wire_bytes"])

    def test_invalid_inputs(self):
        tensor = torch.randn(32)
        with self.assertRaises(ValueError):
            quantize_tensor(tensor, nbits=5)
        with self.assertRaises(ValueError):
            quantize_tensor(tensor, nbits=4, block_size=7)


if __name__ == "__main__":
    unittest.main()
