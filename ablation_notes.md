# https://github.com/karpathy/nanochat/blob/master/dev/LOG.md
# Compares ideas with our original baseline  baseline vs single ablation.
Think about optimizing this pretraining code for rtx gpu
https://github.com/karpathy/nanochat/discussions/481
- Trition-jax attention
- FA4 -> https://github.com/karpathy/nanochat/pull/609
- Long tokenizer from google deepresearch char skill2.md
- Attention residual -> https://github.com/karpathy/nanochat/pull/646
- ZeRO-3 style sharding: each rank owns a slice of optimizer state
- Fitting scaling laws (124M for ablations before testing on 1.5B target eval core bpb+ wall clock instead of val/loss)

These are the ones I didn’t keep/implement (might still work though):

- Value embeddings - couldn’t get either the vanilla implementation or the U-Net to work without a major slowdown earlier.
- Block sliding window - I was lazy and didn’t want to implement sliding window attention, though it’s certainly possible.
- FP8 - Not available on TPU v6e (though v7p will likely have it). I am considering int8 training for future versions, given that the flops and bytes per second both double.
- Batched Muon - I was lazy and did not implement this.
- Custom hardware/PyTorch optimizations - Skipped custom communication strategies and other parallelism tricks from the official speedrun.
- mup sweep -> More hparam tuning could shave off some steps.

Questions?
- The question remains: why is HBM utilization only ~50%? Some of the candidates could be suboptimal overlapping, kernel launch overhead, or communication issues not fully saturating the memory bus.

- The official speedrun saw some benefit from manual communication primitives. This could be explored in JAX.

- Custom kernels in Pallas (lowers to Mosaic on TPUs and Triton on GPUs):
    - A Pallas kernel for block-sparse flex attention could be useful. I wasted a lot of time trying to integrate a kernel that claimed   to do this but wasn’t actually block-sparse.
    - I tried integrating existing Flash/Splash Attention implementations, but they didn’t work on the first attempt, so I moved on. It would be great if someone could get these working.
    - A custom Pallas kernel for the cut cross entropy loss could help improve MFU (according to someone whose MFU on GPUs got fixed after using a CUDA kernel for that).
- The computation for the optimizers likely happens on every PyTree leaf individually and with replicated computation across shards. This may or may not a bottleneck right now, and could be helpful to keep in mind (to avoid extraneous computation/unfused ops). Sharding computation requires using an all-gather which incurs overhead and should be measured, as with everything else.
- Using Microsoft’s Dion optimizer instead of Muon could be interesting.
- Going a bit further, trying out other parallelism strategies like FSDP/TP/hybrid strategies could be looked into.

- holding architecture + optimizer + compute fixed, have you ever tried systematically varying the data curation / filtering regime, rather than the training stack, and measuring how much variance shows up downstream?

- The first thing that came to my mind was to maybe try curriculum learning. It would look something like this :

  1 - Fix everything (seed, model depth, token budget, eval interval).
  2 - Compare:

Baseline random sampling
Curriculum warmup (first 10–20% tokens) then baseline sampling
I assume this would involve training a classifier first to score documents from Fineweb-Edu according to their "difficulty" (score categories could be 0-5). For this we can build a tiny subset of documents annotated by a LLM and train the classifier on it.n

- Have you ever tried to train the same (small) model with different training data permutations, and maybe also with different random seeds. I wonder what the random variance is for the same architecture, and whether minor differences in validation loss of different architectural tweaks are really meaningful.


Also from nanochat discussion 481
Test ideas that did not work.