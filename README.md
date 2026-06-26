# Standalone Database-less Velox PIDS Pipeline

This repository implements a standalone, database-free version of the **Velox** Provenance-based Intrusion Detection System (PIDS) pipeline. Unlike the original framework, this version runs entirely without a PostgreSQL server, storing and querying all node and event data using local **Parquet** files.

---

## 📋 Table of Contents
1. [Overview](#-overview)
2. [Data Logging with eAudit](#-data-logging-with-eaudit)
3. [Data Preprocessing & Parquet Store](#-data-preprocessing--parquet-store)
4. [Mock Database Connector](#-mock-database-connector)
5. [Training the Models](#-training-the-models)
6. [Evaluating the GNN Model](#-evaluating-the-gnn-model)
7. [Running End-to-End](#-running-end-to-end)

---

## 🔍 Overview

The pipeline runs inside a Docker container (`velox-pids`) and consists of three interface stages:
* **Preprocessing**: Ingests eAudit syscall logs and writes them to compressed Parquet format (`data/`).
* **Training**: Builds graph representations, trains Word2Vec node embeddings, and trains the GNN (TGN/GAT) model on CPU.
* **Evaluation**: Measures detection performance, computes diagnostic stats, and generates SVG analysis plots.

---

## 📥 Data Logging with eAudit

To generate your own training and detection logs from host system calls:

### 1. Capture Raw Kernel Events
Run the capture daemon `ecapd` to record system call ring buffer logs to a binary file:
```bash
# Run this on the host machine inside the eaudit directory
sudo ./ecapd -- -C /tmp/my_capture.bin
```
*Tip: Keep executing commands, scripts, or application routines in another shell prompt to generate background activity during logging.*

### 2. Parse and Serialize Logs
Convert the binary capture file into a serialized text event format:
```bash
sudo ./eaudit -I /tmp/my_capture.bin -P /tmp/my_parsed.txt
```

---

## 📊 Data Preprocessing & Parquet Store

Ingest the serialized text logs and generate tabular provenance structures:
```bash
python preprocess.py --input-log /path/to/my_parsed.txt --output-dir ./data
```

This step generates four key Parquet files in the `data/` directory:
- **`events.parquet`**: Direct causal relationships (e.g. read, write, execute edges).
- **`subjects.parquet`**: Process execution contexts (PID, command, binary paths).
- **`files.parquet`**: File node metadata.
- **`netflows.parquet`**: Network connections and sockets.

---

## 🔌 Mock Database Connector

To keep the graph construction components in `PIDSMaker` intact without a running database server, we utilize a **Mock Database Connector** located in `pidsmaker/utils/utils.py`. 

This connector intercepts standard PostgreSQL database queries and executes them as local Pandas DataFrame operations directly on the Parquet files:
- Simulates SQL filtering queries (`BETWEEN`, `IN`, `ORDER BY`).
- Emulates cursor and fetch methods (`fetchall()`, `cursor()`).
- Bypasses any PostgreSQL socket connections automatically.

---

## 🚀 Training the Models

To train the Word2Vec and Velox GNN model:
```bash
python train.py --data-dir ./data --artifacts-dir ./artifacts
```
* This constructs a temporal graph sequence from your Parquet tables.
* Trains a Word2Vec model to generate node semantic embeddings.
* Trains the GNN model using self-supervised learning (by default for 11 epochs on CPU).
* Saves training progress checkpoints and logs under the `artifacts/` folder.

---

## 📈 Evaluating the GNN Model

You can evaluate the trained model's performance on the **train split** (or test split).

### Run Train Split Evaluation
```bash
python eval.py --data-dir ./data --artifacts-dir ./artifacts --eval-on-train
```

* **`--eval-on-train`**: Copies the training split loss logs into the test split directory, forcing the evaluation script to calculate detection metrics against your training set.
* Threshold calibration is automatically computed based on the validation set bounds (e.g., max or mean val losses).
* Generates metrics (Precision, Recall, FPR, F-Score) and plots (`scores_model_epoch_X.png`, `neat_scores_model_epoch_X.svg`).

---

## ⚙️ Running End-to-End

A wrapper script `run_pipeline.sh` is provided to process raw parsed eAudit logs end-to-end:

```bash
# Inside the docker container:
./run_pipeline.sh /path/to/my_parsed.txt
```
This script sequentially runs:
1. `preprocess.py` on the input log.
2. `train.py` to train both models.
3. `eval.py` to evaluate metrics.
