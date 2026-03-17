# ModelConfig, TrainConfig, GenerationConfig
from dataclasses import dataclass

"""
All model, training, and generation hyperparameters live here.
python config.py
"""

# MODEL
@dataclass
class ModelConfig:
    """
    TinyStories default:
      vocab_size=100,280
      max_seq_len=256
      d_model=704
      n_heads=11
      n_layers=9
      d_ff=2816

    With tied embeddings this lands at ~124.3M parameters by the
    analytic estimate and ~124.4M parameters in the instantiated model.
    """
    # Vocabulary  and context
    vocab_size:   int = 100_280   # TinyStories tokenizer vocab: cl100k_base + BOS/EOS/PAD
    max_seq_len:  int = 256       # context window (tokens)

    # Width / depth
    d_model:      int = 704       # embedding and hidden dimension
    n_heads:      int = 11        # 704 / 11 = 64-dim heads
    n_layers:     int = 9         # transformer blocks stacked
    d_ff:         int = 2816      # FFN inner dim (=4 x d_model)

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
    checkpoint_dir: str = "./checkpoints/124m-tinystories"

    # batching
    batch_size:     int = 8
    seq_len:        int = 256       # must match ModelConfig.max_seq_len

    # optimizer adam -> muon later
    learning_rate:  float = 2e-4
    weight_decay:   float = 0.1
    beta1:          float = 0.9
    beta2:          float = 0.95
    grad_clip:      float = 1.0

    # lr schedule
    warmup_step:    int = 1_000
    total_steps:    int = 50_000
    min_lr:         float = 2e-5     # cosine decay floor

    # logging/ checkpointing
    log_every:      int = 50
    eval_every:     int = 500     # run validation every N steps
    eval_iters:     int = 20      # number of validation batches to average over
    save_every:     int = 1_000
    keep_checkpointing: int = 5

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

