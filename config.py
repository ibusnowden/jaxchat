# ModelConfig, TrainConfig, GenerationConfig
from dataclasses import dataclass

"""
config.py — MiniGPT Course: Central Configuration
===================================================
All model, training, and generation hyperparameters live here.
Edit values once; they propagate to every other module.
"""

# MODEL
@dataclass
class ModelConfig:
    """
    Default values produce ~20 M parameters (weight-tied):

    Component                   Parameters
    ──────────────────────────────────────
    Token embedding (8192×512)   4,194,304
    Pos  embedding  (256 ×512)     131,072
    Per TransformerBlock:
      QKV proj  (512→1536)         786,432
      Out proj  (512→512)          262,144
      FFN fc1   (512→2048)       1,048,576
      FFN fc2   (2048→512)       1,048,576
      LayerNorms (×2, scale+bias)    2,048
      Subtotal                   3,147,776
    × 5 blocks                  15,738,880
    Final LayerNorm                  1,024
    LM head            (tied →  0 extra)
    ──────────────────────────────────────
    TOTAL                       20,065,280 

    TP = (V x d_model) x n_layer x (12 x d^2_model)
    """
    # Vocabulary  and context
    vocab_size:   int = 8_192     # characters ≈ 65 for char-level; 8192 for small BPE
    max_seq_len:  int = 256       # context window (tokens)

    # Width / depth
    d_model:      int = 512     # embedding and hidden dimension
    n_heads:      int = 8        # attention heads (d_model % n_heads == 0)
    n_layers:     int = 8        # transformer blocks stacked
    d_ff:         int = 2048      # FFN inner dim (=4 x d_model)

    # Regularisation
    dropout:      float = 0.1
    attn_dropout: float = 0.1

    # Weight tying: share token-embd matrix with LM head
    tie_weights:  bool = True

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, (
            f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}"
        )
        self.d_head: int = self.d_model // self.n_heads 

    @property
    def approx_params(self) -> int:
        emb = self.vocab_size * self.d_model + self.max_seq_len * self.d_model
        block =  (self.d_model * 3 * self.d_model           # QKV
                + self.d_model * self.d_model               # out proj 
                + self.d_model * self.d_ff                  # fc1
                + self.d_ff * self.d_model)                  # fc2
        head = 0 if self.tie_weights else self.vocab_size * self.d_model
        return emb + self.n_layers * block + head
    
# Training
@dataclass
class TrainConfig:
    # path
    data_dir:       str = "./data"
    checkpoint_dir: str = "./checkpoints"

    # batching
    batch_size:     int = 32
    seq_len:        int = 256       # must match ModelConfig.max_seq_len

    # optimizer adam -> muon later
    learning_rate:  float = 3e-4
    weight_decay:   float = 0.1
    beta1:          float = 0.9
    beat2:          float = 0.95
    grad_clip:      float = 1.0

    # lr schedule
    warmup_step:    int = 500
    total_steps:    int = 10_000
    min_lr:         float = 3e-5     # cosine decay floor

    # logging/ checkpointing
    log_every:      int = 50
    eval_every:     int = 500     # validation batches to average over
    save_every:     int = 1_000
    keep_checkpointing: int = 3 

    # repro
    seed:           int = 42


# Generation
@dataclass
class GenerationConfig:
  max_new_tokens:    int = 256
  temperature:       float = 0.8
  top_k:             int = 50
  top_p:             float = 0.9       # nucleus sampling
  repetition_penalty:float = 1.1       # >1 discourages repeats


# Defaults
DEFAULT_MODEL = ModelConfig()
DEFAULT_TRAIN = TrainConfig()
DEFAULT_GEN = GenerationConfig()


if __name__ == "__main__":
    cfg = ModelConfig()
    print(f"ModelConfig -> approx {cfg.approx_params:,} params (~{cfg.approx_params/1e6:.1f} M)")


