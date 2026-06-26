#!/usr/bin/env python3
import re
import os
import sys
import hashlib
import uuid
import argparse
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Edge direction mappings (DARPA TC E3 format)
edge_reversed = {
    "EVENT_EXECUTE",
    "EVENT_OPEN",
    "EVENT_READ",
    "EVENT_RECVFROM",
    "EVENT_RECVMSG",
    "EVENT_READ_SOCKET_PARAMS"
}

# Syscall to relation type mapping
syscall_to_op = {
    "open": "EVENT_OPEN",
    "read": "EVENT_READ",
    "write": "EVENT_WRITE",
    "execve": "EVENT_EXECUTE",
    "clone": "EVENT_CLONE",
    "fork": "EVENT_CLONE",
    "connect": "EVENT_CONNECT",
    "accept": "EVENT_OPEN",
    "sendto": "EVENT_SENDTO",
    "recvfrom": "EVENT_RECVFROM",
}

def stringtomd5(originstr):
    originstr = originstr.encode("utf-8")
    signaturemd5 = hashlib.sha256()
    signaturemd5.update(originstr)
    return signaturemd5.hexdigest()

def get_process_info(pid):
    try:
        exe_path = os.readlink(f"/proc/{pid}/exe")
    except:
        exe_path = "null"
    try:
        with open(f"/proc/{pid}/cmdline", "r") as f:
            cmdline = f.read().replace("\x00", " ").strip()
        if not cmdline:
            cmdline = "null"
    except:
        cmdline = "null"
    return exe_path, cmdline

def parse_args_field(args_str):
    file_path = None
    endpoint = None
    obj_id = None
    
    file_match = re.search(r'file="([^"]+)"', args_str)
    if file_match:
        file_path = file_match.group(1)
        
    endpoint_match = re.search(r'endpoint=([^,\s\)]+)', args_str)
    if endpoint_match:
        endpoint = endpoint_match.group(1)
        
    id_match = re.search(r'id=(\d+)', args_str)
    if id_match:
        obj_id = id_match.group(1)
        
    return file_path, endpoint, obj_id

def extract_cmdline_from_execve(args_str):
    matches = re.findall(r'argv\[(\d+)\]=([^,\)]+)', args_str)
    if matches:
        sorted_args = sorted(matches, key=lambda x: int(x[0]))
        cmdline = " ".join([val.strip('"') for _, val in sorted_args])
        return cmdline
    return None

def yield_stitched_lines(file_path):
    entry_start_pattern = re.compile(r'^\d+\.\d+:\d+:')
    current_entry = []
    with open(file_path, "r") as f:
        for line in f:
            if entry_start_pattern.match(line):
                if current_entry:
                    yield " ".join(current_entry)
                current_entry = [line.strip()]
            else:
                if current_entry:
                    current_entry.append(line.strip())
                else:
                    current_entry = [line.strip()]
    if current_entry:
        yield " ".join(current_entry)

def main():
    parser = argparse.ArgumentParser(description="Preprocess eAudit logs with cgroup info to Parquet format (Memory Efficient).")
    parser.add_argument("--input", default="temp_parsed_serialized.txt", help="Path to parsed eAudit log file.")
    parser.add_argument("--output-dir", default="./data", help="Output directory to store Parquet files.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input file '{args.input}' not found.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Patterns for the new log format
    line_pattern = re.compile(
        r'^([\d\.]+):(\d+):\s+pid=(\d+)\s+cgroup=(\d+)\s+cgroup_path=(\S+)\s+start_time=(\d+):(?:\s+tid=(\d+):)?\s+(\w+)\((.*)\)(?:\s+ret=(\S+))?(?:\s+\[id=(\d+)\])?'
    )
    
    clone_ret_pattern = re.compile(
        r'^([\d\.]+):(\d+):\s+pid=(\d+)\s+cgroup=(\d+)\s+cgroup_path=(\S+)\s+start_time=(\d+):(?:\s+tid=\d+:)?\s+(clone|fork)\s+ret=(\d+)'
    )

    # First pass: Count total events for partitioning shift logic
    print("Counting events in first pass (low memory)...")
    total_events = 0
    for entry in yield_stitched_lines(args.input):
        if not entry:
            continue
        m = line_pattern.match(entry)
        if m:
            syscall = m.group(8)
        else:
            m = clone_ret_pattern.match(entry)
            if m:
                syscall = m.group(7)
            else:
                m = re.match(r'^([\d\.]+):(\d+):\s+(\w+)\((.*)\)(?:\s+ret=(\S+))?(?:\s+\[id=(\d+)\])?', entry)
                if m:
                    syscall = m.group(3)
                else:
                    continue
        if syscall in syscall_to_op:
            total_events += 1

    print(f"Total valid events: {total_events}")

    # Set up incremental Parquet writer for events
    event_schema = pa.schema([
        ("src_node", pa.string()),
        ("src_index_id", pa.string()),
        ("operation", pa.string()),
        ("dst_node", pa.string()),
        ("dst_index_id", pa.string()),
        ("event_uuid", pa.string()),
        ("timestamp_rec", pa.int64()),
        ("_id", pa.int64())
    ])
    events_parquet_path = os.path.join(args.output_dir, "events.parquet")
    writer = pq.ParquetWriter(events_parquet_path, event_schema)

    node_uuid_to_index = {}
    current_index = 0
    
    subjects_to_insert = {} # uuid: (path, cmd, cgroup_id, cgroup_path, index_id)
    files_to_insert = {} # uuid: (path, index_id)
    netflows_to_insert = {} # uuid: (src_addr, src_port, dst_addr, dst_port, index_id)
    
    pid_to_info = {}

    def register_node(node_uuid):
        nonlocal current_index
        if node_uuid not in node_uuid_to_index:
            node_uuid_to_index[node_uuid] = current_index
            current_index += 1
        return node_uuid_to_index[node_uuid]

    print("Processing events and writing to Parquet...")
    idx = 0
    batch_data = []
    batch_size = 200000

    for entry in yield_stitched_lines(args.input):
        if not entry:
            continue
            
        m = line_pattern.match(entry)
        if m:
            ts_str, seq_str, pid, cgroup_id, cgroup_path, start_time, tid, syscall, args_str, ret_str, obj_id = m.groups()
        else:
            m = clone_ret_pattern.match(entry)
            if m:
                ts_str, seq_str, pid, cgroup_id, cgroup_path, start_time, syscall, ret_str = m.groups()
                args_str, obj_id = "", None
            else:
                # Fallback check (no pid prefix)
                m = re.match(r'^([\d\.]+):(\d+):\s+(\w+)\((.*)\)(?:\s+ret=(\S+))?(?:\s+\[id=(\d+)\])?', entry)
                if not m:
                    continue
                ts_str, seq_str, syscall, args_str, ret_str, obj_id = m.groups()
                pid = "0"
                cgroup_id = "null"
                cgroup_path = "null"

        if syscall not in syscall_to_op:
            continue

        op = syscall_to_op[syscall]
        timestamp_ns = int(float(ts_str) * 1e9)
        file_path, endpoint, args_id = parse_args_field(args_str)
        resolved_id = obj_id or args_id

        if idx < int(0.6 * total_events):
            shift_seconds = 0
        elif idx < int(0.8 * total_events):
            shift_seconds = 86400
        else:
            shift_seconds = 172800
        
        shifted_timestamp_ns = timestamp_ns + shift_seconds * 1000000000
        event_uuid = str(uuid.uuid4())

        if pid not in pid_to_info:
            exe_path, cmd = get_process_info(pid)
            pid_to_info[pid] = {
                "path": exe_path, 
                "cmd": cmd,
                "cgroup_id": cgroup_id,
                "cgroup_path": cgroup_path
            }
        else:
            if cgroup_id != "null":
                pid_to_info[pid]["cgroup_id"] = cgroup_id
            if cgroup_path != "null":
                pid_to_info[pid]["cgroup_path"] = cgroup_path

        if syscall == "execve" and file_path:
            cmdline = extract_cmdline_from_execve(args_str) or pid_to_info[pid]["cmd"]
            pid_to_info[pid]["path"] = file_path
            pid_to_info[pid]["cmd"] = cmdline

        subj_uuid = stringtomd5(f"subject_{pid}")
        subj_idx = register_node(subj_uuid)
        subjects_to_insert[subj_uuid] = (
            pid_to_info[pid]["path"], 
            pid_to_info[pid]["cmd"], 
            pid_to_info[pid].get("cgroup_id", "null"),
            pid_to_info[pid].get("cgroup_path", "null"),
            subj_idx
        )

        src_node = None
        dst_node = None

        if syscall in ("clone", "fork"):
            child_pid = ret_str
            if not child_pid or not child_pid.isdigit() or child_pid == "0":
                continue
            child_uuid = stringtomd5(f"subject_{child_pid}")
            child_idx = register_node(child_uuid)
            
            parent_info = pid_to_info.get(pid, {"path": "unknown", "cmd": "unknown", "cgroup_id": "null", "cgroup_path": "null"})
            pid_to_info[child_pid] = parent_info.copy()
            pid_to_info[child_pid]["cgroup_id"] = cgroup_id
            pid_to_info[child_pid]["cgroup_path"] = cgroup_path
            
            subjects_to_insert[child_uuid] = (
                pid_to_info[child_pid]["path"], 
                pid_to_info[child_pid]["cmd"], 
                pid_to_info[child_pid].get("cgroup_id", "null"),
                pid_to_info[child_pid].get("cgroup_path", "null"),
                child_idx
            )

            src_node = (subj_uuid, subj_idx)
            dst_node = (child_uuid, child_idx)

        else:
            obj_name = file_path or endpoint or f"obj_{resolved_id}"
            obj_uuid = resolved_id or stringtomd5(obj_name)
            
            is_netflow = syscall in ("connect", "accept", "sendto", "recvfrom")
            if not is_netflow and obj_name:
                is_netflow = obj_name.startswith(("IP4:", "IP6:", "unix:", "netlink:"))

            obj_idx = register_node(obj_uuid)

            if is_netflow:
                if obj_uuid not in netflows_to_insert:
                    dst_addr, dst_port = "0.0.0.0", "0"
                    if ":" in obj_name:
                        parts = obj_name.split(":")
                        if len(parts) >= 2:
                            dst_addr = parts[-2]
                            dst_port = parts[-1]
                    netflows_to_insert[obj_uuid] = ("0.0.0.0", "0", dst_addr, dst_port, obj_idx)
            else:
                if obj_uuid not in files_to_insert:
                    files_to_insert[obj_uuid] = (obj_name, obj_idx)

            if op in edge_reversed:
                src_node = (obj_uuid, obj_idx)
                dst_node = (subj_uuid, subj_idx)
            else:
                src_node = (subj_uuid, subj_idx)
                dst_node = (obj_uuid, obj_idx)

        if src_node and dst_node:
            batch_data.append((
                src_node[0], str(src_node[1]),
                op,
                dst_node[0], str(dst_node[1]),
                event_uuid, shifted_timestamp_ns, idx
            ))
            idx += 1

            if len(batch_data) >= batch_size:
                cols = list(zip(*batch_data))
                table = pa.Table.from_arrays([pa.array(c) for c in cols], schema=event_schema)
                writer.write_table(table)
                batch_data = []

    if batch_data:
        cols = list(zip(*batch_data))
        table = pa.Table.from_arrays([pa.array(c) for c in cols], schema=event_schema)
        writer.write_table(table)
        batch_data = []

    writer.close()
    print(f"Preprocessed {idx} events.")

    # Convert node dictionaries to pandas DataFrames and save to Parquet
    df_subjects = pd.DataFrame(
        [[node_uuid, node_uuid, path, cmd, cgroup_id, cgroup_path, idx] 
         for node_uuid, (path, cmd, cgroup_id, cgroup_path, idx) in subjects_to_insert.items()],
        columns=["node_uuid", "hash_id", "path", "cmd", "cgroup_id", "cgroup_path", "index_id"]
    )
    df_subjects.to_parquet(os.path.join(args.output_dir, "subjects.parquet"))
    print(f"Saved {len(df_subjects)} subjects to subjects.parquet")

    df_files = pd.DataFrame(
        [[node_uuid, node_uuid, path, idx] for node_uuid, (path, idx) in files_to_insert.items()],
        columns=["node_uuid", "hash_id", "path", "index_id"]
    )
    df_files.to_parquet(os.path.join(args.output_dir, "files.parquet"))
    print(f"Saved {len(df_files)} files to files.parquet")

    df_netflows = pd.DataFrame(
        [[node_uuid, node_uuid, src_addr, src_port, dst_addr, dst_port, idx] 
         for node_uuid, (src_addr, src_port, dst_addr, dst_port, idx) in netflows_to_insert.items()],
        columns=["node_uuid", "hash_id", "src_addr", "src_port", "dst_addr", "dst_port", "index_id"]
    )
    df_netflows.to_parquet(os.path.join(args.output_dir, "netflows.parquet"))
    print(f"Saved {len(df_netflows)} netflows to netflows.parquet")

if __name__ == "__main__":
    main()
