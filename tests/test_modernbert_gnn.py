"""Small, offline regression checks for the ModernBERT migration."""

import tempfile
import unittest
from unittest.mock import patch

import torch

from Pretrain.pretrain_config import PretrainConfig
from Pretrain.pretrain_model import MoCoPretrainModel, ModernBertPretrainModel


class ModernBertTests(unittest.TestCase):
    def config(self, **kwargs):
        return PretrainConfig(
            vocab_size=64, hidden_size=32, num_hidden_layers=2,
            num_attention_heads=4, intermediate_size=64, embedding_size=16,
            max_seq_length=32, moco_buffer_size=8, graph_dropout=0.0,
            attention_backend="sdpa",
            **kwargs,
        )

    def batch(self):
        ids = torch.randint(1, 64, (2, 16))
        ids[:, -4:] = 0
        batch = dict(input_ids=ids, attention_mask=(ids != 0).long())
        for branch in ("cfg", "ddg"):
            batch[f"{branch}_node_spans"] = torch.tensor([[0, 3], [4, 7], [0, 3], [4, 7]])
            batch[f"{branch}_node_batch"] = torch.tensor([0, 0, 1, 1])
            batch[f"{branch}_edge_index"] = torch.tensor([[0, 2], [1, 3]])
        batch["cfg_edge_attr"] = torch.ones(2, 1)
        batch["labels"] = ids.masked_fill(ids == 0, -100)
        batch["group_ids"] = torch.tensor([0, 1])
        return batch

    def test_graph_modes_and_roundtrip(self):
        for cfg, ddg in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(cfg=cfg, ddg=ddg):
                model = ModernBertPretrainModel(self.config(use_cfg=cfg, use_ddg=ddg))
                batch = self.batch()
                result = model(**batch)
                self.assertTrue(torch.isfinite(result["loss"]))
                (result["loss"] + result["embedding"][:, 0].sum()).backward()
                for branch, enabled in (("cfg", cfg), ("ddg", ddg)):
                    if enabled:
                        graph = getattr(model.graph_encoder, f"{branch}_branch")
                        self.assertTrue(any(
                            p.grad is not None and p.grad.abs().sum() > 0
                            for p in graph.parameters()
                        ))
                model.eval()
                with torch.no_grad():
                    expected = model.forward_embedding(**batch)["embedding"]
                    torch.testing.assert_close(expected.norm(dim=1), torch.ones(2))
                with tempfile.TemporaryDirectory() as directory:
                    model.save_pretrained(directory, safe_serialization=False)
                    restored = ModernBertPretrainModel.from_pretrained(directory).eval()
                    with torch.no_grad():
                        torch.testing.assert_close(restored.forward_embedding(**batch)["embedding"], expected)
                    self.assertIs(restored.get_input_embeddings().weight, restored.decoder.weight)

    def test_moco_checkpointing_and_attention(self):
        model = MoCoPretrainModel(self.config())
        model.gradient_checkpointing_enable({"use_reentrant": False})
        batch = self.batch()
        result = model(view1=batch, view2=batch)
        result["loss"].backward()
        self.assertTrue(torch.isfinite(result["loss"]))
        self.assertEqual(model.queue_ptr.item(), 2)
        self.assertTrue(all(p.grad is None for p in model.encoder_k.parameters()))
        model.eval()
        with self.assertRaisesRegex(ValueError, "export is disabled"):
            model.encoder_q.forward_embedding(**batch, output_attentions=True)
        self.assertEqual(model.config._attn_implementation, "sdpa")

    def test_reject_old_config(self):
        with self.assertRaises(ValueError):
            self.config(model_type="roformer")

    def test_sparse_mlm_matches_full_loss_and_gradients(self):
        model = ModernBertPretrainModel(self.config(use_cfg=True, use_ddg=True)).eval()
        batch = self.batch()
        batch["labels"][:] = -100
        batch["labels"][:, 2] = batch["input_ids"][:, 2]
        full_loss, logits, full_embedding = model(**batch, return_dict=False)
        self.assertEqual(logits.shape, (2, 16, 64))
        (full_loss + full_embedding[:, 0].sum()).backward()
        expected = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        with patch.object(model.mlm_decoder, "forward", wraps=model.mlm_decoder.forward) as decoder:
            sparse = model(**batch, return_dict=True)
            self.assertEqual(decoder.call_args.args[0].shape, (2, 32))
        torch.testing.assert_close(sparse["loss"], full_loss)
        torch.testing.assert_close(sparse["embedding"], full_embedding)
        (sparse["loss"] + sparse["embedding"][:, 0].sum()).backward()
        actual = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
        self.assertEqual(actual.keys(), expected.keys())
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], atol=2e-6, rtol=2e-4)

    def test_mlm_empty_and_missing_labels_skip_decoder(self):
        model = ModernBertPretrainModel(self.config())
        batch = self.batch()
        batch["labels"][:] = -100
        with patch.object(model.mlm_decoder, "forward", side_effect=AssertionError("unused decoder")):
            result = model(**batch)
            self.assertEqual(result["loss"].item(), 0)
            (result["loss"] + result["embedding"][:, 0].sum()).backward()
            del batch["labels"]
            self.assertIsNone(model(**batch)["loss"])

    def test_key_skips_mlm(self):
        model = MoCoPretrainModel(self.config())
        batch = self.batch()
        with patch.object(model.encoder_k.mlm_transform, "forward", side_effect=AssertionError("key MLM")):
            result = model(batch, batch)
            result["loss"].backward()
        self.assertTrue(model.encoder_k.training)
        self.assertTrue(all(p.grad is None for p in model.encoder_k.parameters()))

    def test_embedding_api_and_checkpoint_names_remain_compatible(self):
        model = ModernBertPretrainModel(self.config()).eval()
        batch = self.batch()
        with torch.no_grad():
            expected = model.forward_embedding(**batch)["embedding"]
            torch.testing.assert_close(model.encode(**batch)["embedding"], expected)
            torch.testing.assert_close(
                model.forward_embedding(**batch, return_dict=False)[0], expected
            )
        self.assertIs(model.mlm_transform, model.head)
        self.assertIs(model.mlm_decoder, model.decoder)
        self.assertIs(model.embedding_projection, model.linear)
        state = model.state_dict()
        for prefix in ("head.", "decoder.", "linear."):
            self.assertTrue(any(key.startswith(prefix) for key in state))
        for prefix in ("mlm_transform.", "mlm_decoder.", "embedding_projection."):
            self.assertFalse(any(key.startswith(prefix) for key in state))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_bfloat16(self):
        model = MoCoPretrainModel(self.config()).cuda()
        model.gradient_checkpointing_enable({"use_reentrant": False})
        batch = {key: value.cuda() for key, value in self.batch().items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(view1=batch, view2=batch)
        result["loss"].backward()
        self.assertTrue(torch.isfinite(result["loss"]))


if __name__ == "__main__":
    unittest.main()
