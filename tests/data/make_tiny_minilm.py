# /// script
# requires-python = ">=3.12"
# dependencies = ["onnx>=1.17", "tokenizers>=0.21", "numpy>=2"]
# ///
"""Generate tests/data/tiny-minilm/: a few-KB stand-in for all-MiniLM-L6-v2.

    uv run tests/data/make_tiny_minilm.py

Same interface as the real files - ``model.onnx`` with the inputs ``input_ids``,
``attention_mask``, ``token_type_ids`` (int64, [batch, tokens]) and one output
``last_hidden_state`` (float32, [batch, tokens, dim]); ``tokenizer.json`` a BERT
WordPiece tokenizer with [CLS] ... [SEP] - so the tests exercise ONNX Runtime and
the tokenizer exactly as the real model does, offline. The "network" is only an
embedding lookup, so it knows nothing about meaning; that is not what it tests.
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from tokenizers import Tokenizer, normalizers, pre_tokenizers, processors
from tokenizers.models import WordPiece

OUT = Path(__file__).parent / "tiny-minilm"
VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "the", "event", "loop", "walrus", "cat", "##s", "gil", "python"]
DIM = 8


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(8)
    words = numpy_helper.from_array(rng.normal(size=(len(VOCAB), DIM)).astype(np.float32), "word_embeddings")
    types = numpy_helper.from_array(rng.normal(size=(2, DIM)).astype(np.float32), "type_embeddings")
    graph = helper.make_graph(
        [
            helper.make_node("Gather", ["word_embeddings", "input_ids"], ["w"]),
            helper.make_node("Gather", ["type_embeddings", "token_type_ids"], ["t"]),
            helper.make_node("Add", ["w", "t"], ["last_hidden_state"]),
        ],
        "tiny-minilm",
        [
            helper.make_tensor_value_info(name, TensorProto.INT64, ["batch", "tokens"])
            for name in ("input_ids", "attention_mask", "token_type_ids")
        ],
        [helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT, ["batch", "tokens", DIM])],
        initializer=[words, types],
    )
    opset = [helper.make_opsetid("", 17)]
    model = helper.make_model(graph, opset_imports=opset, producer_name="findex-tests")
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, OUT / "model.onnx")

    tokenizer = Tokenizer(WordPiece({w: i for i, w in enumerate(VOCAB)}, unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    tokenizer.save(str(OUT / "tokenizer.json"))
    print(f"wrote {OUT}/model.onnx and tokenizer.json")


if __name__ == "__main__":
    main()
