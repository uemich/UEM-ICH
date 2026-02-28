# Model Weights

This directory should contain the trained model checkpoints required for evaluation.

## Downloads

Download the following `.pth` files from the shared drive and place them in this directory:

- `best_multitask.pth`: Pretrained ConvNeXtV2 encoder + SparK decoder (used by all scripts).
- `best_aggregator.pth`: Pretrained scan-level transformer aggregator (used by MBH and CQ500 aggregator scripts).
- `best_multiclass_decoder.pth`: Pretrained SegFormer decoder for multi-class segmentation (used by `mbh_inference.py`).

## File Structure

```
weights/
├── best_multitask.pth
├── best_aggregator.pth
└── best_multiclass_decoder.pth
```
