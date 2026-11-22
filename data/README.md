# KiT Data Demo

This is the demo dataset for [KiT: A Foundation Model for Candlestick Time-Series Forecasting via Diffusion Transformers](https://arxiv.org/abs/2609.34507).

## Dataset Description

This dataset contains 100 Chinese A-share stocks with multi-timescale OHLCV (Open, High, Low, Close, Volume) candlestick data.

### Data Format

The dataset is organized as follows:

```
data/
  instruments.json      # market / sector / vocab_index of the 100 stocks
  stats.json            # per-bucket normalisation statistics from training
  trade_dates.csv       # A-share trading calendar
  ashare/{1m,5m,15m,30m,1h,2h,1d}/<code>.parquet  # OHLCV data by timescale
```

Each `.parquet` file contains OHLCV candlestick data for one stock at one timescale, with columns:
- Open
- High
- Low
- Close
- Volume

### Timescales

The data includes 7 different timescales:
- 1m (1 minute)
- 5m (5 minutes)
- 15m (15 minutes)
- 30m (30 minutes)
- 1h (1 hour)
- 2h (2 hours)
- 1d (1 day)


## Code Repository

The corresponding code is available at: [https://github.com/Luciferbobo/KiT](https://github.com/Luciferbobo/KiT)

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
