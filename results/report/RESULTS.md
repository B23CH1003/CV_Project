# Results summary (collected by collect_results.py)

All waste numbers are in percentage points (pp). Held-out = the 4 photos YOLO never saw (yolo_ds/split.json) and their generated after images.

## Segmentation, held-out photos (scored inside labelled dish areas)

| method    | images   |   n |   mIoU |   food_IoU |
|:----------|:---------|----:|-------:|-----------:|
| classical | before   |  21 |  0.029 |      0.741 |
| classical | after    |  65 |  0.041 |      0.532 |
| classical | all      |  86 |  0.034 |      0.644 |
| yolo      | before   |  21 |  0.475 |      0.834 |
| yolo      | after    |  65 |  0.338 |      0.82  |
| yolo      | all      |  86 |  0.409 |      0.829 |

## Waste MAE, classical food mask, held-out photos

| config   |   daylight |   evening |   ALL |
|:---------|-----------:|----------:|------:|
| full     |      13.84 |      6.19 |  8.51 |

## Ablation with YOLO, held-out photos (full / no_align / no_color / neither)

| config   |   daylight |   evening |   ALL |
|:---------|-----------:|----------:|------:|
| full     |      11.75 |      1.75 |  4.79 |
| no_align |      12.66 |      5.22 |  7.48 |
| no_color |       8.12 |      1.9  |  3.79 |
| neither  |       9.09 |      5.08 |  6.3  |

## Waste MAE, YOLO, all 615 plates (includes photos YOLO trained on - not the fair accuracy number)

| config   |   daylight |   evening |   ALL |
|:---------|-----------:|----------:|------:|
| full     |       2.36 |      1.41 |  1.87 |

## Pattern recovery - TRUE waste (upper bound)

| dish    | weekday   | meal      | pattern_type     | notes                      | recovered   |
|:--------|:----------|:----------|:-----------------|:---------------------------|:------------|
| fish    | Friday    | dinner    | chronic          | mean 50% sd 15             | True        |
| bread   | Sunday    | breakfast | weekly_recurring | mean 55% sd 15             | True        |
| chicken | Wednesday | dinner    | weekly_recurring | mean 55% sd 15             | True        |
| rice    | all       | all       | drifting         | start 12% +6%/week         | True        |
| pasta   | all       | all       | serving_decay    | +20% when minutes > 60     | True        |
| salad   | all       | all       | pairing          | +25% when served with eggs | True        |

## Analytics metrics - TRUE waste

```
{
  "analytics_source": "true waste (oracle)",
  "pattern_precision": 1.0,
  "pattern_recall": 1.0,
  "pattern_f1": 1.0,
  "pattern_healthy_specificity": 1.0,
  "forecast_mae_seasonal_naive": 9.45,
  "forecast_mae_cell_mean": 8.69,
  "forecast_mae_global_mean": 9.31
}
```

## Pattern recovery - YOLO-MEASURED waste (end-to-end headline)

| dish    | weekday   | meal      | pattern_type     | notes                      | recovered   |
|:--------|:----------|:----------|:-----------------|:---------------------------|:------------|
| fish    | Friday    | dinner    | chronic          | mean 50% sd 15             | True        |
| bread   | Sunday    | breakfast | weekly_recurring | mean 55% sd 15             | True        |
| chicken | Wednesday | dinner    | weekly_recurring | mean 55% sd 15             | True        |
| rice    | all       | all       | drifting         | start 12% +6%/week         | True        |
| pasta   | all       | all       | serving_decay    | +20% when minutes > 60     | True        |
| salad   | all       | all       | pairing          | +25% when served with eggs | True        |

## Analytics metrics - MEASURED waste

```
{
  "analytics_source": "predicted (waste_predictions.csv)",
  "pattern_precision": 0.857,
  "pattern_recall": 1.0,
  "pattern_f1": 0.923,
  "pattern_healthy_specificity": 0.889,
  "forecast_mae_seasonal_naive": 9.63,
  "forecast_mae_cell_mean": 8.84,
  "forecast_mae_global_mean": 9.18
}
```

## Pattern recovery vs added measurement noise (sd, pp)

|   noise_sd |   precision |   recall |       f1 |
|-----------:|------------:|---------:|---------:|
|          0 |    1        | 1        | 1        |
|          5 |    1        | 1        | 1        |
|         10 |    0.904762 | 1        | 0.948718 |
|         15 |    0.869048 | 1        | 0.92674  |
|         20 |    0.861111 | 0.944444 | 0.896825 |

## Figures (report/figures/)

- **01_generator_before_after.jpg**: Synthetic eating generator: real before photo | generated after photo
- **02_alignment_check.jpg**: Alignment: before | after | after aligned | overlay (grey = aligned)
- **03_segmentation_examples.jpg**: Segmentation on held-out photos: image | ground truth | classical | YOLO
- **05_waste_check_classical.jpg**: Waste measurement with the classical colour mask: before | aligned after | food mask
- **06_heatmap_waste_TRUE.png**: Mean waste % per dish x slot (true waste)
- **07_heatmap_waste_MEASURED.png**: Mean waste % per dish x slot (YOLO-measured waste)
- **08_heatmap_twci_MEASURED.png**: TWCI = mean waste x recurrence (measured)
- **09_drift_MEASURED.png**: Weekly mean waste per dish (measured) - rice drift
- **10_serving_decay_MEASURED.png**: Waste vs minutes since service start (measured) - pasta decay
- **11_pairing_MEASURED.png**: Pairing lift (measured) - salad with eggs
- **12_robustness_noise.png**: Pattern recovery vs added measurement noise
- **16_leftover_model_training_curves.png**: Leftover model training: results.png
- **17_leftover_model_val_predictions.jpg**: Leftover model training: val_batch0_pred.jpg

## Not found (run these steps, then collect again)

- results/yolo/waste_check.jpg
- results/yolo/waste_ablation.csv
