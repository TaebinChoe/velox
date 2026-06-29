import os
import sys
import time
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from gensim.models import Word2Vec
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score

# Ensure the root directory of the project is in the path to import pidsmaker
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from pidsmaker.utils.utils import tokenize_label

# Helper function to print logs
def log(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)

class EventClassifierMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def cal_word_weight(n, percentage=30):
    if n == 0:
        return []
    d = -1 / n * percentage / 100
    a_1 = 1 / n - 0.5 * (n - 1) * d
    sequence = []
    for i in range(n):
        a_i = a_1 + i * d
        sequence.append(a_i)
    return sequence

def main():
    train_dir = "/pscratch/sd/s/sgkim/tchoe_home/hpc_sec/vm/shared/data/benign/pg"
    eval_dir = "/pscratch/sd/s/sgkim/tchoe_home/hpc_sec/vm/shared/data/db/pg"
    malicious_csv = "/pscratch/sd/s/sgkim/tchoe_home/hpc_sec/vm/shared/data/db/malicious_events.csv"
    
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    output_parquet = os.path.join(exp_dir, "evaluation_results.parquet")
    output_report = os.path.join(exp_dir, "report.md")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")
    
    start_all_time = time.time()
    
    # -------------------------------------------------------------
    # 1. Split validation in train data & Load Datasets
    # -------------------------------------------------------------
    log("Step 1: Loading benign training events and splitting validation set...")
    train_events_df = pd.read_parquet(os.path.join(train_dir, "events.parquet"))
    
    # Sort chronologically by timestamp_rec
    train_events_df = train_events_df.sort_values(by="timestamp_rec").reset_index(drop=True)
    
    # Chronological Split: 80% train, 20% validation
    split_idx = int(len(train_events_df) * 0.8)
    train_split_df = train_events_df.iloc[:split_idx].reset_index(drop=True)
    val_split_df = train_events_df.iloc[split_idx:].reset_index(drop=True)
    
    log(f"Total benign training events: {len(train_events_df)}")
    log(f"Training split events: {len(train_split_df)}")
    log(f"Validation split events: {len(val_split_df)}")
    
    # Load nodes to build node embeddings mapping
    log("Loading subjects, files, and netflows parquets...")
    train_subjects = pd.read_parquet(os.path.join(train_dir, "subjects.parquet"))
    train_files = pd.read_parquet(os.path.join(train_dir, "files.parquet"))
    train_netflows = pd.read_parquet(os.path.join(train_dir, "netflows.parquet"))
    
    eval_subjects = pd.read_parquet(os.path.join(eval_dir, "subjects.parquet"))
    eval_files = pd.read_parquet(os.path.join(eval_dir, "files.parquet"))
    eval_netflows = pd.read_parquet(os.path.join(eval_dir, "netflows.parquet"))
    
    # Deduplicate nodes to have a single mapping of index_id -> type & label
    subjects = pd.concat([train_subjects, eval_subjects]).drop_duplicates(subset=['index_id'])
    files = pd.concat([train_files, eval_files]).drop_duplicates(subset=['index_id'])
    netflows = pd.concat([train_netflows, eval_netflows]).drop_duplicates(subset=['index_id'])
    
    log("Building global index_id to label mapping...")
    node_id_to_info = {}
    
    # Follow SQL loading order: netflows -> subjects -> files (files overwrite netflows if they overlap)
    for _, row in netflows.iterrows():
        idx = int(row['index_id'])
        dst_addr = str(row['dst_addr']) if row['dst_addr'] is not None else ""
        dst_port = str(row['dst_port']) if row['dst_port'] is not None else ""
        label = f"netflow {dst_addr} {dst_port}"
        node_id_to_info[idx] = ("netflow", label)
        
    for _, row in subjects.iterrows():
        idx = int(row['index_id'])
        path = str(row['path']) if row['path'] is not None else ""
        cmd = str(row['cmd']) if row['cmd'] is not None else ""
        label = f"subject {path} {cmd}"
        node_id_to_info[idx] = ("subject", label)
        
    for _, row in files.iterrows():
        idx = int(row['index_id'])
        path = str(row['path']) if row['path'] is not None else ""
        label = f"file {path}"
        node_id_to_info[idx] = ("file", label)
        
    # Build list of node ids in benign train data (to train Word2Vec only on train nodes)
    train_node_ids = set(train_subjects['index_id']).union(set(train_files['index_id'])).union(set(train_netflows['index_id']))
    
    # -------------------------------------------------------------
    # 2. Train Word2Vec
    # -------------------------------------------------------------
    log("Step 2: Training Word2Vec node embeddings...")
    start_w2v_time = time.time()
    
    # Prepare tokenized corpus for training nodes
    corpus = []
    for idx in train_node_ids:
        if idx in node_id_to_info:
            node_type, label = node_id_to_info[idx]
            try:
                tokens = tokenize_label(label, node_type)
            except Exception as e:
                # simple tokenizer fallback
                tokens = re.sub(r"[\\/:\.]+", " ", label).split()
            corpus.append(tokens)
            
    emb_dim = 128
    w2v_model = Word2Vec(
        corpus,
        alpha=0.025,
        vector_size=emb_dim,
        window=5,
        min_count=1,
        sg=1, # Skip-gram
        workers=4,
        epochs=50,
        seed=0
    )
    
    # Compatibility support for gensim init_sims
    try:
        w2v_model.init_sims(replace=True)
    except AttributeError:
        w2v_model.wv.fill_norms(force=True)
        
    w2v_time = time.time() - start_w2v_time
    log(f"Word2Vec training completed in {w2v_time:.2f} seconds.")
    
    # Compute embeddings for all nodes in the universe
    log("Computing dense node embeddings...")
    zeros = np.zeros((emb_dim,), dtype=np.float32)
    max_node_id = max(node_id_to_info.keys())
    embeddings_matrix = np.zeros((max_node_id + 1, emb_dim), dtype=np.float32)
    
    for idx, (node_type, label) in node_id_to_info.items():
        try:
            tokens = tokenize_label(label, node_type)
        except Exception:
            tokens = re.sub(r"[\\/:\.]+", " ", label).split()
            
        if not tokens:
            embeddings_matrix[idx] = zeros
            continue
            
        weight_list = cal_word_weight(len(tokens), 30)
        word_vectors = [w2v_model.wv[word] if word in w2v_model.wv else zeros for word in tokens]
        weighted_vectors = [weight * word_vec for weight, word_vec in zip(weight_list, word_vectors)]
        sentence_vector = np.mean(weighted_vectors, axis=0)
        norm = np.linalg.norm(sentence_vector)
        normalized_vector = sentence_vector / (norm + 1e-12)
        embeddings_matrix[idx] = normalized_vector

    # -------------------------------------------------------------
    # 3. Train MLP
    # -------------------------------------------------------------
    log("Step 3: Preparing training matrices and training MLP classifier...")
    start_mlp_time = time.time()
    
    # Collect all operations from train and eval sets to build operations vocabulary
    eval_events_df = pd.read_parquet(os.path.join(eval_dir, "events.parquet"))
    all_ops = sorted(list(set(train_events_df['operation'].unique()) | set(eval_events_df['operation'].unique())))
    op_to_idx = {op: i for i, op in enumerate(all_ops)}
    num_classes = len(all_ops)
    log(f"Operations vocabulary: {op_to_idx}")
    
    # Construct train/validation indices (not full matrices to save memory)
    log("Preparing train/validation indices...")
    src_indices_train = train_split_df['src_index_id'].astype(int).values
    dst_indices_train = train_split_df['dst_index_id'].astype(int).values
    y_train = train_split_df['operation'].map(op_to_idx).values
    
    src_indices_val = val_split_df['src_index_id'].astype(int).values
    dst_indices_val = val_split_df['dst_index_id'].astype(int).values
    y_val = val_split_df['operation'].map(op_to_idx).values
    
    # Custom memory-efficient dataset
    class MemoryEfficientDataset(Dataset):
        def __init__(self, src_ids, dst_ids, labels, embeddings):
            self.src_ids = src_ids
            self.dst_ids = dst_ids
            self.labels = labels
            self.embeddings = embeddings
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            src_emb = self.embeddings[self.src_ids[idx]]
            dst_emb = self.embeddings[self.dst_ids[idx]]
            x = np.concatenate([src_emb, dst_emb], axis=0)
            return torch.from_numpy(x).float(), torch.tensor(self.labels[idx]).long()
            
    train_dataset = MemoryEfficientDataset(src_indices_train, dst_indices_train, y_train, embeddings_matrix)
    # Use num_workers=0 to avoid multiprocessing overhead / login node limits
    train_loader = DataLoader(train_dataset, batch_size=8192, shuffle=True, num_workers=0)
    
    # Train PyTorch MLP
    model = EventClassifierMLP(input_dim=2*emb_dim, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    num_epochs = 10
    best_val_loss = float('inf')
    best_model_state = None
    patience = 3
    patience_counter = 0
    
    log(f"Starting MLP training (up to {num_epochs} epochs)...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(batch_x)
            
        epoch_loss /= len(train_split_df)
        
        # Validation Loss (using batching)
        model.eval()
        val_loss = 0.0
        val_dataset = MemoryEfficientDataset(src_indices_val, dst_indices_val, y_val, embeddings_matrix)
        val_loader = DataLoader(val_dataset, batch_size=16384, shuffle=False, num_workers=0)
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * len(batch_x)
            val_loss /= len(val_split_df)
            
        log(f"Epoch {epoch+1:02d} | Train Loss: {epoch_loss:.5f} | Val Loss: {val_loss:.5f} | Time: {time.time() - epoch_start:.1f}s")
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log(f"Early stopping at epoch {epoch+1}")
                break
                
    # Restore best model
    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    
    mlp_time = time.time() - start_mlp_time
    total_training_time = w2v_time + mlp_time
    log(f"MLP training completed in {mlp_time:.2f} seconds.")
    log(f"Total training time: {total_training_time:.2f} seconds.")
    
    # -------------------------------------------------------------
    # 4. Cal anomaly score for validation data & 5. Set threshold
    # -------------------------------------------------------------
    log("Step 4 & 5: Calculating anomaly scores on validation set to determine thresholds...")
    
    def compute_anomaly_scores_efficient(model, src_ids, dst_ids, y, embeddings, device, batch_size=16384):
        model.eval()
        scores = np.zeros(len(src_ids))
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        with torch.no_grad():
            for i in range(0, len(src_ids), batch_size):
                end_idx = min(i + batch_size, len(src_ids))
                batch_src = src_ids[i:end_idx]
                batch_dst = dst_ids[i:end_idx]
                batch_y = torch.from_numpy(y[i:end_idx]).long().to(device)
                
                # Fetch embeddings and concatenate
                emb_src = embeddings[batch_src]
                emb_dst = embeddings[batch_dst]
                batch_x = np.concatenate([emb_src, emb_dst], axis=1)
                
                batch_x_t = torch.from_numpy(batch_x).float().to(device)
                logits = model(batch_x_t)
                loss = loss_fn(logits, batch_y)
                scores[i:end_idx] = loss.cpu().numpy()
        return scores

    val_scores = compute_anomaly_scores_efficient(model, src_indices_val, dst_indices_val, y_val, embeddings_matrix, device)
    
    # Set different percentiles as thresholds
    thr_max = np.max(val_scores)
    thr_99_9 = np.percentile(val_scores, 99.9)
    thr_99 = np.percentile(val_scores, 99.0)
    
    log(f"Validation anomaly score stats: mean={np.mean(val_scores):.4f}, std={np.std(val_scores):.4f}")
    log(f"Threshold (Max): {thr_max:.5f}")
    log(f"Threshold (99.9th percentile): {thr_99_9:.5f}")
    log(f"Threshold (99th percentile): {thr_99:.5f}")
    
    # -------------------------------------------------------------
    # 6. Calc anomaly score for eval data and save to result parquet
    # -------------------------------------------------------------
    log("Step 6: Running inference on evaluation dataset and saving results...")
    start_det_time = time.time()
    
    src_indices_eval = eval_events_df['src_index_id'].astype(int).values
    dst_indices_eval = eval_events_df['dst_index_id'].astype(int).values
    y_eval = eval_events_df['operation'].map(op_to_idx).values
    
    eval_scores = compute_anomaly_scores_efficient(model, src_indices_eval, dst_indices_eval, y_eval, embeddings_matrix, device)
    det_time = time.time() - start_det_time
    
    # Append results to the evaluation DataFrame
    eval_events_df['anomaly_score'] = eval_scores
    eval_events_df['is_anomaly_max'] = (eval_scores > thr_max).astype(int)
    eval_events_df['is_anomaly_999'] = (eval_scores > thr_99_9).astype(int)
    eval_events_df['is_anomaly_99'] = (eval_scores > thr_99).astype(int)
    
    # Save to result parquet file
    eval_events_df.to_parquet(output_parquet)
    log(f"Detection/inference completed in {det_time:.2f} seconds.")
    log(f"Saved evaluation results to: {output_parquet}")
    
    # -------------------------------------------------------------
    # 7. Write report with performance metrics
    # -------------------------------------------------------------
    log("Step 7: Computing performance metrics against ground truth...")
    
    # Load ground truth malicious events
    if os.path.exists(malicious_csv):
        with open(malicious_csv, 'r') as f:
            malicious_uuids = set(line.strip() for line in f if line.strip())
    else:
        log(f"WARNING: Malicious events file not found at {malicious_csv}")
        malicious_uuids = set()
        
    y_true = eval_events_df['event_uuid'].isin(malicious_uuids).astype(int).values
    total_malicious = np.sum(y_true)
    log(f"Found {total_malicious} malicious events in the evaluation set.")
    
    def evaluate_threshold(thr, thr_name):
        y_pred = (eval_scores > thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        auroc = roc_auc_score(y_true, eval_scores)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        return {
            "name": thr_name,
            "threshold": thr,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "auroc": auroc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        
    results = [
        evaluate_threshold(thr_max, "Max Val Loss"),
        evaluate_threshold(thr_99_9, "99.9% Percentile"),
        evaluate_threshold(thr_99, "99% Percentile")
    ]
    
    # Write report file
    with open(output_report, 'w') as f:
        f.write("# Velox Anomaly Detection Pipeline Report\n\n")
        f.write("This report presents the performance of the Word2Vec + MLP anomaly detection model on provenance graph evaluation data.\n\n")
        
        f.write("## 1. Benchmarking Times\n")
        f.write("| Stage | Time (seconds) |\n")
        f.write("| --- | --- |\n")
        f.write(f"| Word2Vec Training | {w2v_time:.2f} s |\n")
        f.write(f"| MLP Training | {mlp_time:.2f} s |\n")
        f.write(f"| **Total Training Time** | **{total_training_time:.2f} s** |\n")
        f.write(f"| **Inference/Detection Time** | **{det_time:.2f} s** |\n\n")
        
        f.write("## 2. Performance Metrics for Different Thresholds\n\n")
        f.write("| Threshold Setting | Value | TP | FP | TN | FN | AUROC | Precision | Recall | F1 Score |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for res in results:
            f.write(f"| {res['name']} | {res['threshold']:.4f} | {res['tp']} | {res['fp']} | {res['tn']} | {res['fn']} | {res['auroc']:.5f} | {res['precision']:.5f} | {res['recall']:.5f} | {res['f1']:.5f} |\n")
            
        f.write("\n")
        f.write("## 3. Dataset Summary\n")
        f.write(f"- Benign Train Events: {len(train_events_df)}\n")
        f.write(f"- Benign Train Split (80%): {len(train_split_df)}\n")
        f.write(f"- Benign Val Split (20%): {len(val_split_df)}\n")
        f.write(f"- Evaluation Events: {len(eval_events_df)}\n")
        f.write(f"- Evaluation Malicious Events: {total_malicious}\n")
        
    log(f"Saved performance report to: {output_report}")
    log(f"Complete pipeline execution finished in {time.time() - start_all_time:.2f} seconds.")

if __name__ == "__main__":
    main()
