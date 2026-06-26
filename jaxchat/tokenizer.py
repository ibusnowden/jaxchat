"""
BPE Tokenizer in the style of GPT-4.

Two implementations are available:
1) HuggingFace Tokenizer that can do both training and inference but is really confusing
2) Our own RustBPE Tokenizer for training and tiktoken for efficient inference
"""

from __future__ import annotations

import argparse
import os
import copy
import sys
from functools import lru_cache
from typing import Sequence

SPECIAL_TOKENS = [
    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    "<|bos|>",
    # tokens below are only used during finetuning to render Conversations into token ids
    "<|user_start|>", # user messages
    "<|user_end|>",
    "<|assistant_start|>", # assistant messages
    "<|assistant_end|>",
    "<|python_start|>", # assistant invokes python REPL tool
    "<|python_end|>",
    "<|output_start|>", # python REPL outputs back to assistant
    "<|output_end|>",
]

# NOTE: this split pattern deviates from GPT-4 in that we use \p{N}{1,2} instead of \p{N}{1,3}
# I did this because I didn't want to "waste" too many tokens on numbers for smaller vocab sizes.
# I verified that 2 is the sweet spot for vocab size of 32K. 1 is a bit worse, 3 was worse still.
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FINEWEB_TOKENIZER_DIR = os.path.join(PROJECT_ROOT, "data", "fineweb32k")
DEFAULT_FINEWEB_TOKENIZER_DATASETS = ("HuggingFaceFW/fineweb-edu", "HuggingFaceFW/fineweb")
DEFAULT_FINEWEB_VOCAB_SIZE = 32768
DEFAULT_RUST65K_TOKENIZER_DIR = os.path.join(PROJECT_ROOT, "data", "fineweb65k", "tokenizer")


def resolve_tokenizer_json_path(tokenizer_path_or_dir: str) -> str:
    if tokenizer_path_or_dir.endswith(".json"):
        return tokenizer_path_or_dir
    return os.path.join(tokenizer_path_or_dir, "tokenizer.json")


def resolve_tokenizer_pkl_path(tokenizer_path_or_dir: str) -> str:
    if tokenizer_path_or_dir.endswith(".pkl"):
        return tokenizer_path_or_dir
    return os.path.join(tokenizer_path_or_dir, "tokenizer.pkl")


def _tokenizer_json_candidates(tokenizer_path_or_dir: str) -> tuple[str, ...]:
    tokenizer_json = resolve_tokenizer_json_path(tokenizer_path_or_dir)
    candidates = [tokenizer_json]
    if not os.path.isabs(tokenizer_json):
        repo_relative = os.path.join(PROJECT_ROOT, tokenizer_json)
        if repo_relative not in candidates:
            candidates.append(repo_relative)
    return tuple(candidates)


def _find_existing_tokenizer_json(tokenizer_path_or_dir: str) -> str | None:
    for candidate in _tokenizer_json_candidates(tokenizer_path_or_dir):
        if os.path.exists(candidate):
            return candidate
    return None


def _tokenizer_pkl_candidates(tokenizer_path_or_dir: str) -> tuple[str, ...]:
    tokenizer_pkl = resolve_tokenizer_pkl_path(tokenizer_path_or_dir)
    candidates = [tokenizer_pkl]
    if not os.path.isabs(tokenizer_pkl):
        repo_relative = os.path.join(PROJECT_ROOT, tokenizer_pkl)
        if repo_relative not in candidates:
            candidates.append(repo_relative)
    return tuple(candidates)


def _find_existing_tokenizer_pkl(tokenizer_path_or_dir: str) -> str | None:
    for candidate in _tokenizer_pkl_candidates(tokenizer_path_or_dir):
        if os.path.exists(candidate):
            return candidate
    return None


def find_existing_tokenizer_path(tokenizer_path_or_dir: str) -> str | None:
    """Return an existing tokenizer artifact path, preferring HF JSON over Rust pickle."""
    return _find_existing_tokenizer_json(tokenizer_path_or_dir) or _find_existing_tokenizer_pkl(tokenizer_path_or_dir)


def normalize_dataset_specs(
    dataset_names: str | Sequence[str],
    dataset_configs: str | Sequence[str | None] | None = None,
) -> tuple[tuple[str, str | None], ...]:
    if isinstance(dataset_names, str):
        names = tuple(part.strip() for part in dataset_names.split(",") if part.strip())
    else:
        names = tuple(str(name).strip() for name in dataset_names if str(name).strip())
    if not names:
        raise ValueError("At least one dataset name must be provided.")

    if dataset_configs is None:
        configs = (None,) * len(names)
    elif isinstance(dataset_configs, str):
        configs = tuple(part.strip() or None for part in dataset_configs.split(","))
    else:
        configs = tuple(config.strip() if isinstance(config, str) else config for config in dataset_configs)

    if len(configs) == 1 and len(names) > 1:
        configs = configs * len(names)
    if len(configs) != len(names):
        raise ValueError(
            f"Dataset/config count mismatch: got {len(names)} dataset names and {len(configs)} configs."
        )
    return tuple(zip(names, configs))


def iter_hf_dataset_rows(
    dataset_names: str | Sequence[str],
    *,
    dataset_configs: str | Sequence[str | None] | None = None,
    split: str = "train",
):
    from datasets import load_dataset

    for dataset_name, dataset_config in normalize_dataset_specs(dataset_names, dataset_configs):
        dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
        for row in dataset:
            yield row


def iter_hf_dataset_text(
    dataset_names: str | Sequence[str],
    *,
    dataset_configs: str | Sequence[str | None] | None = None,
    split: str = "train",
    text_field: str = "text",
    max_documents: int | None = None,
):
    emitted = 0
    for row in iter_hf_dataset_rows(dataset_names, dataset_configs=dataset_configs, split=split):
        text = row.get(text_field)
        if not isinstance(text, str) or not text:
            continue
        yield text
        emitted += 1
        if max_documents is not None and emitted >= max_documents:
            return

# -----------------------------------------------------------------------------
# Generic GPT-4-style tokenizer based on HuggingFace Tokenizer
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import pre_tokenizers, decoders, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

class HuggingFaceTokenizer:
    """Light wrapper around HuggingFace Tokenizer for some utilities"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, hf_path):
        # init from a HuggingFace pretrained tokenizer (e.g. "gpt2")
        tokenizer = HFTokenizer.from_pretrained(hf_path)
        return cls(tokenizer)

    @classmethod
    def from_file(cls, tokenizer_path):
        tokenizer = HFTokenizer.from_file(tokenizer_path)
        return cls(tokenizer)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        # init from a local directory on disk (e.g. "out/tokenizer")
        return cls.from_file(resolve_tokenizer_json_path(tokenizer_dir))

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # train from an iterator of text
        # Configure the HuggingFace Tokenizer
        tokenizer = HFTokenizer(BPE(
            byte_fallback=True, # needed!
            unk_token=None,
            fuse_unk=False,
        ))
        # Normalizer: None
        tokenizer.normalizer = None
        # Pre-tokenizer: GPT-4 style
        # the regex pattern used by GPT-4 to split text into groups before BPE
        # NOTE: The pattern was changed from \p{N}{1,3} to \p{N}{1,2} because I suspect it is harmful to
        # very small models and smaller vocab sizes, because it is a little bit wasteful in the token space.
        # (but I haven't validated this! TODO)
        gpt4_split_regex = Regex(SPLIT_PATTERN) # huggingface demands that you wrap it in Regex!!
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(pattern=gpt4_split_regex, behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        ])
        # Decoder: ByteLevel (it pairs together with the ByteLevel pre-tokenizer)
        tokenizer.decoder = decoders.ByteLevel()
        # Post-processor: None
        tokenizer.post_processor = None
        # Trainer: BPE
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            min_frequency=0, # no minimum frequency
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=SPECIAL_TOKENS,
        )
        # Kick off the training
        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_special_tokens(self):
        special_tokens_map = self.tokenizer.get_added_tokens_decoder()
        special_tokens = [w.content for w in special_tokens_map.values()]
        return special_tokens

    def id_to_token(self, id):
        return self.tokenizer.id_to_token(id)

    def _encode_one(self, text, prepend=None, append=None, num_threads=None):
        # encode a single string
        # prepend/append can be either a string of a special token or a token id directly.
        # num_threads is ignored (only used by the nanochat Tokenizer for parallel encoding)
        assert isinstance(text, str)
        ids = []
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            ids.append(prepend_id)
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
            ids.append(append_id)
        return ids

    def encode_special(self, text):
        # encode a single special token via exact match
        return self.tokenizer.token_to_id(text)

    def get_bos_token_id(self):
        # Different HuggingFace models use different BOS tokens and there is little consistency
        # 1) attempt to find a <|bos|> token
        bos = self.encode_special("<|bos|>")
        # 2) if that fails, attempt to find a <|endoftext|> token (e.g. GPT-2 models)
        if bos is None:
            bos = self.encode_special("<|endoftext|>")
        # 3) if these fail, it's better to crash than to silently return None
        assert bos is not None, "Failed to find BOS token in tokenizer"
        return bos

    def encode(self, text, *args, **kwargs):
        if isinstance(text, str):
            return self._encode_one(text, *args, **kwargs)
        elif isinstance(text, list):
            return [self._encode_one(t, *args, **kwargs) for t in text]
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, tokenizer_dir):
        # save the tokenizer to disk
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")

    def render_conversation(self, conversation, max_tokens=2048):
        return _render_conversation(self, conversation, max_tokens=max_tokens)

    def render_for_completion(self, conversation):
        return _render_for_completion(self, conversation)

    def visualize_tokenization(self, ids, mask, with_token_id=False):
        return _visualize_tokenization(self, ids, mask, with_token_id=with_token_id)


def _render_conversation(tokenizer, conversation, max_tokens=2048):
    """
    Tokenize a single Chat conversation (which we call a "doc" or "document" here).
    Returns:
    - ids: list[int] is a list of token ids of this rendered conversation
    - mask: list[int] of same length, mask = 1 for tokens that the Assistant is expected to train on.
    """
    ids, mask = [], []
    def add_tokens(token_ids, mask_val):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        ids.extend(token_ids)
        mask.extend([mask_val] * len(token_ids))

    if conversation["messages"][0]["role"] == "system":
        conversation = copy.deepcopy(conversation)
        messages = conversation["messages"]
        assert messages[1]["role"] == "user", "System message must be followed by a user message"
        messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
        messages = messages[1:]
    else:
        messages = conversation["messages"]
    assert len(messages) >= 1, f"Conversation has less than 1 message: {messages}"

    bos = tokenizer.get_bos_token_id()
    user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
    assistant_start, assistant_end = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")
    python_start, python_end = tokenizer.encode_special("<|python_start|>"), tokenizer.encode_special("<|python_end|>")
    output_start, output_end = tokenizer.encode_special("<|output_start|>"), tokenizer.encode_special("<|output_end|>")

    add_tokens(bos, 0)
    for i, message in enumerate(messages):
        must_be_from = "user" if i % 2 == 0 else "assistant"
        assert message["role"] == must_be_from, f"Message {i} is from {message['role']} but should be from {must_be_from}"
        content = message["content"]

        if message["role"] == "user":
            assert isinstance(content, str), "User messages are simply expected to be strings"
            value_ids = tokenizer.encode(content)
            add_tokens(user_start, 0)
            add_tokens(value_ids, 0)
            add_tokens(user_end, 0)
        elif message["role"] == "assistant":
            add_tokens(assistant_start, 0)
            if isinstance(content, str):
                value_ids = tokenizer.encode(content)
                add_tokens(value_ids, 1)
            elif isinstance(content, list):
                for part in content:
                    value_ids = tokenizer.encode(part["text"])
                    if part["type"] == "text":
                        add_tokens(value_ids, 1)
                    elif part["type"] == "python":
                        add_tokens(python_start, 1)
                        add_tokens(value_ids, 1)
                        add_tokens(python_end, 1)
                    elif part["type"] == "python_output":
                        add_tokens(output_start, 0)
                        add_tokens(value_ids, 0)
                        add_tokens(output_end, 0)
                    else:
                        raise ValueError(f"Unknown part type: {part['type']}")
            else:
                raise ValueError(f"Unknown content type: {type(content)}")
            add_tokens(assistant_end, 1)

    ids = ids[:max_tokens]
    mask = mask[:max_tokens]
    return ids, mask


def _render_for_completion(tokenizer, conversation):
    """
    Used during Reinforcement Learning. In that setting, we want to
    render the conversation priming the Assistant for a completion.
    Unlike the Chat SFT case, we don't need to return the mask.
    """
    conversation = copy.deepcopy(conversation)
    messages = conversation["messages"]
    assert messages[-1]["role"] == "assistant", "Last message must be from the Assistant"
    messages.pop()

    ids, _mask = _render_conversation(tokenizer, conversation)

    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    ids.append(assistant_start)
    return ids


def _visualize_tokenization(tokenizer, ids, mask, with_token_id=False):
    RED = '\033[91m'
    GREEN = '\033[92m'
    RESET = '\033[0m'
    GRAY = '\033[90m'
    tokens = []
    for i, (token_id, mask_val) in enumerate(zip(ids, mask)):
        token_str = tokenizer.decode([token_id])
        color = GREEN if mask_val == 1 else RED
        tokens.append(f"{color}{token_str}{RESET}")
        if with_token_id:
            tokens.append(f"{GRAY}({token_id}){RESET}")
    return '|'.join(tokens)


def train_hf_tokenizer_from_dataset(
    *,
    tokenizer_dir: str,
    dataset_names: str | Sequence[str],
    dataset_configs: str | Sequence[str | None] | None = None,
    split: str = "train",
    text_field: str = "text",
    vocab_size: int = DEFAULT_FINEWEB_VOCAB_SIZE,
    max_documents: int | None = None,
):
    text_iter = iter_hf_dataset_text(
        dataset_names,
        dataset_configs=dataset_configs,
        split=split,
        text_field=text_field,
        max_documents=max_documents,
    )
    tokenizer = HuggingFaceTokenizer.train_from_iterator(text_iter, vocab_size=vocab_size)
    tokenizer.save(tokenizer_dir)
    return tokenizer


def load_hf_tokenizer(tokenizer_path_or_dir: str):
    tokenizer_json = _find_existing_tokenizer_json(tokenizer_path_or_dir)
    if tokenizer_json is None:
        checked = _tokenizer_json_candidates(tokenizer_path_or_dir)
        raise FileNotFoundError("Tokenizer not found at " + " or ".join(checked))
    return HuggingFaceTokenizer.from_file(tokenizer_json)


def load_tokenizer(tokenizer_path_or_dir: str):
    """Load either a HuggingFace ``tokenizer.json`` or RustBPE ``tokenizer.pkl`` artifact."""
    tokenizer_json = _find_existing_tokenizer_json(tokenizer_path_or_dir)
    if tokenizer_json is not None:
        return HuggingFaceTokenizer.from_file(tokenizer_json)
    tokenizer_pkl = _find_existing_tokenizer_pkl(tokenizer_path_or_dir)
    if tokenizer_pkl is not None:
        return RustBPETokenizer.from_directory(os.path.dirname(tokenizer_pkl) or ".")
    checked = _tokenizer_json_candidates(tokenizer_path_or_dir) + _tokenizer_pkl_candidates(tokenizer_path_or_dir)
    raise FileNotFoundError("Tokenizer not found at " + " or ".join(checked))


def ensure_hf_tokenizer(
    tokenizer_path_or_dir: str,
    *,
    train_if_missing: bool = False,
    dataset_names: str | Sequence[str] = DEFAULT_FINEWEB_TOKENIZER_DATASETS,
    dataset_configs: str | Sequence[str | None] | None = None,
    split: str = "train",
    text_field: str = "text",
    vocab_size: int = DEFAULT_FINEWEB_VOCAB_SIZE,
    max_documents: int | None = None,
):
    existing_tokenizer_json = _find_existing_tokenizer_json(tokenizer_path_or_dir)
    if existing_tokenizer_json is not None:
        return load_hf_tokenizer(existing_tokenizer_json)
    tokenizer_json = resolve_tokenizer_json_path(tokenizer_path_or_dir)
    if not train_if_missing:
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_json}")
    tokenizer_dir = os.path.dirname(tokenizer_json) or "."
    print(
        f"Training new {vocab_size}-token tokenizer at {tokenizer_json} "
        f"from {normalize_dataset_specs(dataset_names, dataset_configs)}"
    )
    return train_hf_tokenizer_from_dataset(
        tokenizer_dir=tokenizer_dir,
        dataset_names=dataset_names,
        dataset_configs=dataset_configs,
        split=split,
        text_field=text_field,
        vocab_size=vocab_size,
        max_documents=max_documents,
    )


def train_rust_tokenizer_from_dataset(
    *,
    tokenizer_dir: str,
    dataset_names: str | Sequence[str],
    dataset_configs: str | Sequence[str | None] | None = None,
    split: str = "train",
    text_field: str = "text",
    vocab_size: int = 65536,
    max_documents: int | None = None,
):
    text_iter = iter_hf_dataset_text(
        dataset_names,
        dataset_configs=dataset_configs,
        split=split,
        text_field=text_field,
        max_documents=max_documents,
    )
    tokenizer = RustBPETokenizer.train_from_iterator(text_iter, vocab_size=vocab_size)
    tokenizer.save(tokenizer_dir)
    return tokenizer


def ensure_tokenizer(
    tokenizer_path_or_dir: str,
    *,
    impl: str = "auto",
    train_if_missing: bool = False,
    dataset_names: str | Sequence[str] = DEFAULT_FINEWEB_TOKENIZER_DATASETS,
    dataset_configs: str | Sequence[str | None] | None = None,
    split: str = "train",
    text_field: str = "text",
    vocab_size: int = DEFAULT_FINEWEB_VOCAB_SIZE,
    max_documents: int | None = None,
):
    """Load or train a tokenizer using ``impl`` = auto|hf|rust."""
    if impl not in {"auto", "hf", "rust"}:
        raise ValueError(f"Unknown tokenizer impl {impl!r}; expected auto|hf|rust.")
    if impl in {"auto", "hf"}:
        existing_json = _find_existing_tokenizer_json(tokenizer_path_or_dir)
        if existing_json is not None:
            return HuggingFaceTokenizer.from_file(existing_json)
    if impl in {"auto", "rust"}:
        existing_pkl = _find_existing_tokenizer_pkl(tokenizer_path_or_dir)
        if existing_pkl is not None:
            return RustBPETokenizer.from_directory(os.path.dirname(existing_pkl) or ".")
    if not train_if_missing:
        checked = _tokenizer_json_candidates(tokenizer_path_or_dir) + _tokenizer_pkl_candidates(tokenizer_path_or_dir)
        raise FileNotFoundError("Tokenizer not found at " + " or ".join(checked))
    if impl == "auto":
        impl = "rust" if vocab_size >= 65536 else "hf"
    if impl == "rust":
        tokenizer_dir = (
            os.path.dirname(resolve_tokenizer_pkl_path(tokenizer_path_or_dir))
            if tokenizer_path_or_dir.endswith(".pkl")
            else tokenizer_path_or_dir
        )
        print(
            f"Training new RustBPE {vocab_size}-token tokenizer at {resolve_tokenizer_pkl_path(tokenizer_dir)} "
            f"from {normalize_dataset_specs(dataset_names, dataset_configs)}"
        )
        return train_rust_tokenizer_from_dataset(
            tokenizer_dir=tokenizer_dir,
            dataset_names=dataset_names,
            dataset_configs=dataset_configs,
            split=split,
            text_field=text_field,
            vocab_size=vocab_size,
            max_documents=max_documents,
        )
    tokenizer_dir = os.path.dirname(resolve_tokenizer_json_path(tokenizer_path_or_dir)) or "."
    print(
        f"Training new HuggingFace {vocab_size}-token tokenizer at {resolve_tokenizer_json_path(tokenizer_dir)} "
        f"from {normalize_dataset_specs(dataset_names, dataset_configs)}"
    )
    return train_hf_tokenizer_from_dataset(
        tokenizer_dir=tokenizer_dir,
        dataset_names=dataset_names,
        dataset_configs=dataset_configs,
        split=split,
        text_field=text_field,
        vocab_size=vocab_size,
        max_documents=max_documents,
    )

# -----------------------------------------------------------------------------
# Tokenizer based on rustbpe + tiktoken combo
import pickle

try:
    import rustbpe
except ImportError:  # pragma: no cover - optional during tests
    rustbpe = None

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional during tests
    tiktoken = None

class RustBPETokenizer:
    """Light wrapper around tiktoken (for efficient inference) but train with rustbpe"""

    def __init__(self, enc, bos_token):
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        if rustbpe is None or tiktoken is None:
            raise RuntimeError(
                "RustBPETokenizer training requires both `rustbpe` and `tiktoken` to be installed."
            )
        # 1) train using rustbpe
        tokenizer = rustbpe.Tokenizer()
        # the special tokens are inserted later in __init__, we don't train them here
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256, f"vocab_size_no_special must be at least 256, got {vocab_size_no_special}"
        tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)
        # 2) construct the associated tiktoken encoding for inference
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks, # dict[bytes, int] (token bytes -> merge priority rank)
            special_tokens=special_tokens, # dict[str, int] (special token name -> token id)
        )
        return cls(enc, "<|bos|>")

    @classmethod
    def from_directory(cls, tokenizer_dir):
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc, "<|bos|>")

    @classmethod
    def from_pretrained(cls, tiktoken_name):
        if tiktoken is None:
            raise RuntimeError("RustBPETokenizer.from_pretrained requires `tiktoken` to be installed.")
        # https://github.com/openai/tiktoken/blob/eedc8563/tiktoken_ext/openai_public.py
        enc = tiktoken.get_encoding(tiktoken_name)
        # tiktoken calls the special document delimiter token "<|endoftext|>"
        # yes this is confusing because this token is almost always PREPENDED to the beginning of the document
        # it most often is used to signal the start of a new sequence to the LLM during inference etc.
        # so in nanoChat we always use "<|bos|>" short for "beginning of sequence", but historically it is often called "<|endoftext|>".
        return cls(enc, "<|endoftext|>")

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_special_tokens(self):
        return self.enc.special_tokens_set

    def id_to_token(self, id):
        return self.enc.decode([id])

    @lru_cache(maxsize=32)
    def encode_special(self, text):
        return self.enc.encode_single_token(text)

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, append=None, num_threads=8):
        # text can be either a string or a list of strings

        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id) # TODO: slightly inefficient here? :( hmm
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id) # TODO: same
            if append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

        return ids

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.enc.decode(ids)

    def save(self, tokenizer_dir):
        # save the encoding object to disk
        os.makedirs(tokenizer_dir, exist_ok=True)
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(self.enc, f)
        print(f"Saved tokenizer encoding to {pickle_path}")

    def render_conversation(self, conversation, max_tokens=2048):
        return _render_conversation(self, conversation, max_tokens=max_tokens)

    def render_for_completion(self, conversation):
        return _render_for_completion(self, conversation)

    def visualize_tokenization(self, ids, mask, with_token_id=False):
        return _visualize_tokenization(self, ids, mask, with_token_id=with_token_id)

# -----------------------------------------------------------------------------
# nanochat-specific convenience functions

def get_tokenizer():
    from jaxchat.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    # return HuggingFaceTokenizer.from_directory(tokenizer_dir)
    return RustBPETokenizer.from_directory(tokenizer_dir)

def get_token_bytes(device="cpu"):
    import torch
    from jaxchat.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    assert os.path.exists(token_bytes_path), f"Token bytes not found at {token_bytes_path}? It gets written by tok_train.py"
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device)
    return token_bytes


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Train the FineWeb tokenizer used by jaxchat.")
    parser.add_argument(
        "--dataset-name",
        default=",".join(DEFAULT_FINEWEB_TOKENIZER_DATASETS),
        help="Comma-separated Hugging Face dataset names to train on.",
    )
    parser.add_argument(
        "--dataset-config",
        default=None,
        help="Comma-separated dataset configs aligned with --dataset-name.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--tokenizer-dir", default=DEFAULT_FINEWEB_TOKENIZER_DIR)
    parser.add_argument("--impl", choices=("hf", "rust"), default="hf",
                        help="Tokenizer trainer implementation. Use rust for the 65K target.")
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_FINEWEB_VOCAB_SIZE)
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    artifact_path = (
        resolve_tokenizer_pkl_path(args.tokenizer_dir)
        if args.impl == "rust"
        else resolve_tokenizer_json_path(args.tokenizer_dir)
    )
    if os.path.exists(artifact_path) and not args.overwrite:
        print(f"Tokenizer already exists at {artifact_path}; pass --overwrite to retrain.")
        return 0

    if args.impl == "rust":
        train_rust_tokenizer_from_dataset(
            tokenizer_dir=os.path.dirname(artifact_path) or ".",
            dataset_names=args.dataset_name,
            dataset_configs=args.dataset_config,
            split=args.split,
            text_field=args.text_field,
            vocab_size=args.vocab_size,
            max_documents=args.max_documents,
        )
    else:
        train_hf_tokenizer_from_dataset(
            tokenizer_dir=os.path.dirname(artifact_path) or ".",
            dataset_names=args.dataset_name,
            dataset_configs=args.dataset_config,
            split=args.split,
            text_field=args.text_field,
            vocab_size=args.vocab_size,
            max_documents=args.max_documents,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
