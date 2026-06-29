# Velox Anomaly Detection Pipeline Report

This report presents the performance of the Word2Vec + MLP anomaly detection model on provenance graph evaluation data.

## 1. Benchmarking Times
| Stage | Time (seconds) |
| --- | --- |
| Word2Vec Training | 728.63 s |
| MLP Training | 1249.29 s |
| **Total Training Time** | **1977.91 s** |
| **Inference/Detection Time** | **4.76 s** |

## 2. Performance Metrics for Different Thresholds

| Threshold Setting | Value | TP | FP | TN | FN | AUROC | Precision | Recall | F1 Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Max Val Loss | 16.5547 | 9 | 4698 | 10802870 | 240 | 0.96787 | 0.00191 | 0.03614 | 0.00363 |
| 99.9% Percentile | 5.0918 | 248 | 1699569 | 9107999 | 1 | 0.96787 | 0.00015 | 0.99598 | 0.00029 |
| 99% Percentile | 1.3684 | 249 | 2842836 | 7964732 | 0 | 0.96787 | 0.00009 | 1.00000 | 0.00018 |

## 3. Dataset Summary
- Benign Train Events: 10903359
- Benign Train Split (80%): 8722687
- Benign Val Split (20%): 2180672
- Evaluation Events: 10807817
- Evaluation Malicious Events: 249
