# KiT Model Checkpoints

Pre-trained model checkpoints for [KiT: A Foundation Model for Candlestick Time-Series Forecasting via Diffusion Transformers](https://arxiv.org/abs/2609.34507).

## Checkpoint bundles

Each checkpoint is stored with its matching instrument mapping and normalization statistics:

```text
step_market_284M/
  step_market_284M.pt
  instruments.json
  stats.json
step_market_284M_train260410/
  step_market_284M_train260410.pt
  instruments.json
  stats.json
```

| Bundle | Markets | Instruments | Normalization entries | Training cutoff |
| --- | --- | ---: | ---: | --- |
| [step_market_284M](./step_market_284M) | Chinese A-shares, U.S. equities, cryptocurrency | 4,830 | 105 | 2025-12-31 |
| [step_market_284M_train260410](./step_market_284M_train260410) | Chinese A-shares | 3,914 | 35 | 2026-03-09 |

### `step_market_284M`

The official open-source checkpoint used for the paper's reported results and evaluations. Its EMA weights exactly match the original three-market experiment's step 64,000 checkpoint. The existing `.pt` file retains a legacy `step=50000` metadata field; the weight tensors are unchanged.

### `step_market_284M_train260410`

The A-share checkpoint whose EMA weights exactly match the original experiment's step 63,000 checkpoint. The filename is retained for compatibility. The original training configuration and launch records specify training through **2026-03-09**, with **2026-03-10 through 2026-04-10** held out for validation; the filename does not mean that training included all data through April 10. This checkpoint has no reported test-set benchmark.

## Using the matching metadata

- `instruments.json` maps each symbol to its `market`, integer `sector`, and original `vocab_index`. Preserve these IDs: they select the trained instrument embeddings.
- `stats.json` contains the normalization scales used for input normalization and output denormalization, keyed by `<market>|<timescale>|<feature>`.
- For the three-market bundle, A-share and U.S. statistics come from their original `stats.json` files; cryptocurrency statistics come from the original **`stats_trainend2025.json`**, as selected by the training merge code.
- These files were reconstructed from the available original market metadata using the training merge logic, without renumbering instruments or refitting statistics. Eight real-data inference cases across both checkpoints were compared against the matching original experiment weights and source metadata using identical seeds: normalized predictions and reconstructed OHLCV matched exactly. This verifies compatibility on the tested cases; training-time JSON hashes were not recorded.

## Download and inference

Download either bundle, for example:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Lucifer744/KiT_model",
    allow_patterns=["config.json", "step_market_284M/*"],
    local_dir="ckpt",
)
```

The checkpoint path is now `ckpt/step_market_284M/step_market_284M.pt`. For the A-share-only bundle, select `step_market_284M_train260410/*` (keep `config.json` in `allow_patterns`) and use the corresponding nested checkpoint path.

The current [code repository](https://github.com/Luciferbobo/KiT) reads `instruments.json` and `stats.json` from `--data-root`. Place the chosen bundle's two JSON files in that data directory alongside your market/timeframe parquet files and trading calendar, then pass the nested checkpoint path explicitly:

```bash
python eval/infer_for_pred.py --ckpt ckpt/step_market_284M/step_market_284M.pt --data-root data
```

The bundled evaluation scripts target A-shares. Use the lower-level inference API with the appropriate market and calendar for U.S. equities or cryptocurrency. Keep the metadata from the same bundle as the selected checkpoint.

The 100-stock A-share demo data is available at [KiT_data_demo](https://huggingface.co/datasets/Lucifer744/KiT_data_demo).

## Citation

```bibtex
@article{kit2026,
  title={KiT: A Foundation Model for Candlestick Time-Series Forecasting via Diffusion Transformers},
  author={Boyu Zhang, Haorui Li},
  journal={arXiv preprint arXiv:2609.34507},
  year={2026}
}
```

## License

See the main repository for license information.
