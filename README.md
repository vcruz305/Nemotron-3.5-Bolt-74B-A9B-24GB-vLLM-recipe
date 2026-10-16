# Nemotron 3.5 Bolt 74B-A9B 24GB — vLLM recipe

Unofficial serving recipe for NVIDIA's [24GB IQ2 checkpoint](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB) on one DGX Spark (GB10, 128 GB unified memory). **The model and draft weights are not in this repository.** The quick start uses the fastest *measured* profile, LiLiCorr via vLLM's DFlash method; `no-draft` is the measured control. Nothing here starts, stops, or configures `rexl3-manager`.

## Hugging Face models

- **Target:** [NVIDIA Nemotron 3.5 Bolt 74B-A9B 24GB](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB) ([exact measured revision](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB/tree/a326822af111290f2af5ac59ef4b5040347dabc4)).
- **Optional draft:** [NVIDIA Nemotron 3.5 Bolt LiLiCorr](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr). The speed comparisons below use these separate target/draft checkpoints.

## Measured configurations

- Target: `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB`, revision `a326822af111290f2af5ac59ef4b5040347dabc4`; 118 verified files, 24,120,362,058 bytes on disk. Mixed ModelOpt IQ2 (`IQ2_XXS` with selected Q8_0/NVFP4/unquantized tensors), **not an NVFP4-only target**.
- Draft (DFlash profile only): [NVIDIA LiLiCorr](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr), revision `abd478065678ee83a93821c4b3f512da24e4cecb`. Its `model.safetensors` is 962,485,872 bytes and is NVFP4. The draft is separate from the IQ2 target.
- Engine: `vllm==0.32.0rc1.dev100+g77154ca61` (commit `77154ca6116ae34cbfe314beef1016bde9b63989`), `b12x==1.5.1`, FlashInfer attention, b12x IQ2 linears/experts, FP8 KV. One GB10, TP=1, max sequence length **262,144**, max concurrent sequences 8, GPU memory utilization 0.55. DFlash: 7 speculative tokens, standard rejection, FlashInfer draft attention. Target flags are in [`serve.sh`](serve.sh); all variants use `--seed 42`.
- The context number is the *served cap*, not a tested 262k-token request. The server ran on the Spark; the recorded Sixcat client used `127.0.0.1:18000/v1` on a separate Linux host over an SSH forward to the Spark's `127.0.0.1:8000/v1`. Client TTFT and effective prefill therefore include the tunnel. These are not multi-Spark model benchmarks.

## Results

Sixcat **speed** used synthetic prompts, strict/temperature-0 policy with thinking on, 32 confirmation requests per profile. These are unscored speed measurements, **not** quality scores. Decode and prefill p50 are the C=1 curve (four requests in the DFlash run); balanced throughput is C=8 confirmation (32 requests). Historical on-host launch configurations are reproduced by the scripts, but the new scripts have not yet been replayed end-to-end on a GPU.

- **Decode, C=1 p50:** no draft **30.88 tok/s** → LiLiCorr/DFlash **87.82 tok/s** (2.84×).
- **Balanced, C=8 aggregate output:** **136.49** → **195.06 tok/s** (+42.9%). Includes prefill/queue wall time.
- **Long-prompt prefill, C=1 client-effective p50:** **1,584.18** → **1,327.53 tok/s** (−16.2%). This includes network and first-token handoff, not a server-native prefill counter.
- **Balanced TTFT, C=8 p50:** **1,677** → **1,871 ms** (+11.6%).
- DFlash confirmations: decode 32/32, balanced 32/32, prefill 32/32. No-draft prefill confirmation was **24/32** (8 timeouts), so do not infer a throughput win from that incomplete control.
- vLLM `/metrics` after the DFlash run: 34,630 accepted of 62,377 draft tokens (**55.5%**); counters include smoke/probes, not only the final 32 requests.

[Raw speed receipts](bench/receipts/) have only the operator's absolute `server_props` root path replaced with `${MODEL_DIR}`; numerical measurements are unchanged. The repo's **quality** summary is [here](bench/quality-summary.json): Bolt **77.5/100**, 120/120 items completed, six categories including executed Code, strict/thinking-on. **That score belongs to the no-draft serve, not DFlash; DFlash quality was not rerun.** Truncation and loop flags remain disclosed in the summary.

## Quick start on a DGX Spark

Install the pinned engine in an isolated Python 3.12 environment with CUDA 13. The installed on-host runtime used the vLLM wheel from `https://wheels.vllm.ai/77154ca6116ae34cbfe314beef1016bde9b63989/cu130` and `b12x==1.5.1`. For a fresh installation, follow the [official target card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB) for the matching architecture/wheel and run `vllm download-kernels` in that environment. **Fresh bootstrap from this repository has not been validated**; `serve.sh` uses an already installed runtime and verified model directories.

Download both pinned revisions with `huggingface_hub.snapshot_download` (or `hf download`) into separate directories; verify filenames/sizes before serving:

```bash
mkdir -p "$HOME/models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB" \
         "$HOME/models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr"
hf download nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB \
  --revision a326822af111290f2af5ac59ef4b5040347dabc4 \
  --local-dir "$HOME/models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB"
hf download nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr \
  --revision abd478065678ee83a93821c4b3f512da24e4cecb \
  --local-dir "$HOME/models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr"
```

These downloads are documented, not re-run by the recipe. On the measured box those are:

```bash
export VLLM_BIN="$HOME/vllm-bolt-nightly-venv/bin/vllm"
export MODEL_DIR="$HOME/models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB"
export DRAFT_DIR="$HOME/models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr"
DRY_RUN=1 ./serve.sh dflash   # inspect the exact command; no GPU access
./serve.sh dflash             # user systemd units + persistent host-memory guard
./serve.sh status
python3 scripts/smoke.py --model nemotron-bolt --context 262144 --expect-spec
# Requires an installed Sixcat checkout and its Python environment:
SIXCAT_PYTHON=/var/lib/sixcat-eval/venv/bin/python ./bench.sh
./serve.sh down
```

For the no-draft speed control, start `./serve.sh no-draft` and run `EXPECT_DRAFT=0 ./bench.sh` after smoke. `bench.sh` defaults to the Spark-local `:8000/v1`. To match the **historical client transport**, run the same checked-out `bench.sh` from a separate Linux client with Sixcat installed, create an SSH local forward (`ssh -N -L 18000:127.0.0.1:8000 <your-spark-host>` in another terminal), then use `BASE_URL=http://127.0.0.1:18000/v1 ./bench.sh`. The tunnel alters client-observed latency; do not compare it interchangeably with a direct-on-Spark run. Do not start both profiles simultaneously. `serve.sh` refuses to launch while another GPU compute process or the intended port is occupied; it never kills someone else's model. The guard stops **only its own recipe unit** if host available RAM falls below 4 GiB, swap free below 2 GiB, or an initially active `rexl3-manager` exits. Unit logs: `journalctl --user -u vllm-bolt-iq2-recipe -f`; guard log: `~/.local/state/model-serving-recipes/vllm-bolt-iq2-recipe.service.jsonl`.

## Limits

This is a measured **text** speed recipe. No multimodal speed, 262k actual prompt, simultaneous Super VL deployment, or DFlash quality score is established here. The 24GB checkpoint size is not runtime memory use: the measured DFlash serve reduced host available RAM by about 75 GiB on a 128 GB Spark. See the JSON receipts for workload definitions and sample counts. The scripts do not modify model files or download private assets.
