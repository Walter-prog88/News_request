"""No optimization steps: check only initialization and adapter interfaces."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'learning' / '05_lora_forgetting'))
sys.path.insert(0, str(ROOT / 'learning' / '06_continual_lora'))
HAS_TORCH = importlib.util.find_spec('torch') is not None
HAS_SKLEARN = importlib.util.find_spec('sklearn') is not None

@unittest.skipUnless(HAS_TORCH, 'PyTorch unavailable: interface checks skipped')
class LearningInterfacesTest(unittest.TestCase):
    def test_zero_B_matches_frozen_base(self):
        import torch
        from lora_vs_full import LoRALinear
        torch.manual_seed(4)
        base = torch.nn.Linear(4, 3)
        layer = LoRALinear(base, r=2)
        x = torch.ones(2, 4)
        torch.testing.assert_close(layer(x), base(x))
        self.assertTrue(all(not p.requires_grad for p in base.parameters()))
        self.assertTrue(layer.A.requires_grad and layer.B.requires_grad)

    def test_merge_preserves_current_forward_output(self):
        import torch
        from lora_vs_full import MLP, LoRALinear
        from continual_lora import merge_and_reset
        torch.manual_seed(5)
        net = MLP()
        net.layers = torch.nn.ModuleList(LoRALinear(layer, 2) for layer in net.layers)
        with torch.no_grad():
            for layer in net.layers:
                layer.B.fill_(0.01)
            x = torch.ones(2, 64)
            before = net(x).clone()
            merge_and_reset(net)
            torch.testing.assert_close(net(x), before, atol=1e-6, rtol=1e-5)

    @unittest.skipUnless(HAS_SKLEARN, 'scikit-learn unavailable in this Python environment')
    def test_public_data_loader_shapes_and_ranges(self):
        from lora_vs_full import load
        xtr, ytr, xte, yte = load(0)
        self.assertEqual(tuple(xtr.shape), (1437, 64))
        self.assertEqual(tuple(xte.shape), (360, 64))
        self.assertEqual(tuple(ytr.shape), (1437,))
        self.assertEqual(tuple(yte.shape), (360,))
        self.assertGreaterEqual(float(xtr.min()), 0.0)
        self.assertLessEqual(float(xtr.max()), 1.0)

if __name__ == '__main__':
    unittest.main()
