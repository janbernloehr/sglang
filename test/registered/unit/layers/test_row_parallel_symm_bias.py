"""RowParallelLinear runs the same GEMM on every rank inside symmetric memory."""

import unittest
from contextlib import nullcontext
from unittest.mock import MagicMock, Mock, patch

import torch
import torch.nn.functional as F

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    SymmetricMemoryContext,
)
from sglang.srt.layers import linear
from sglang.srt.runtime_context import SpawnRanks, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.parallel_groups import parallel_scope, publish
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, stage="weekly", runner_config="cpu")

TP_SIZE = 2


def build_layer(rank):
    reset_context()
    server = ServerArgs(model_path="dummy", device="cpu", tp_size=TP_SIZE)
    publish(server, role="test", ranks=SpawnRanks(world_rank=rank))
    layer = linear.RowParallelLinear(8, 4, bias=True, reduce_results=False)
    with torch.no_grad():
        layer.weight.copy_(
            (torch.arange(layer.weight.numel()) % 7 - 3).reshape(layer.weight.shape) / 8
        )
        layer.bias.copy_(torch.arange(layer.output_size) / 4 + 0.5)
    return layer


class TestRowParallelSymmBias(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)

    def run_layer(self, layer, symm_active):
        x = (torch.arange(2 * layer.input_size_per_partition) % 5 - 2).reshape(
            2, -1
        ).to(layer.weight.dtype) / 4
        context = nullcontext()
        if symm_active:
            context = MagicMock(spec=SymmetricMemoryContext)
            context.__exit__.return_value = False
        gemm_biases = []
        apply = layer.quant_method.apply

        def record_apply(module, input_, bias=None):
            # Only the GEMM inside the context matters for pool symmetry.
            gemm_biases.append(bias)
            return apply(module, input_, bias=bias)

        with (
            parallel_scope(tp_group=Mock()),
            patch.object(linear, "is_allocation_symmetric", return_value=True),
            patch.object(linear, "use_symmetric_memory", return_value=context),
            patch.object(layer.quant_method, "apply", side_effect=record_apply),
        ):
            output, output_bias = layer(x)
        self.assertIsNone(output_bias)
        return x, output, gemm_biases

    def test_symmetric_memory_runs_unbiased_gemm_on_every_rank(self):
        for rank in range(TP_SIZE):
            layer = build_layer(rank)
            x, output, gemm_biases = self.run_layer(layer, symm_active=True)
            self.assertEqual(gemm_biases, [None])
            # Rank 0 still contributes the bias exactly once to the TP sum.
            expected = F.linear(x, layer.weight, layer.bias if rank == 0 else None)
            torch.testing.assert_close(output, expected)

    def test_without_symmetric_memory_rank0_keeps_fused_bias(self):
        for rank in range(TP_SIZE):
            layer = build_layer(rank)
            x, output, gemm_biases = self.run_layer(layer, symm_active=False)
            self.assertEqual(len(gemm_biases), 1)
            if rank == 0:
                self.assertIs(gemm_biases[0], layer.bias)
            else:
                self.assertIsNone(gemm_biases[0])
            expected = F.linear(x, layer.weight, layer.bias if rank == 0 else None)
            torch.testing.assert_close(output, expected)


if __name__ == "__main__":
    unittest.main()
