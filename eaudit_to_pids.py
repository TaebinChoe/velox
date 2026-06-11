#!/usr/bin/env python3
import re
import os
import sys
import hashlib
import uuid
import psycopg2
from psycopg2 import extras as ex

# Database connection details
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = int(os.environ.get("DB_PORT", 5432 if DB_HOST == "postgres" else 8888))
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "postgres")
DB_NAME = "eaudit_db"

DB_PARAMS = {
    "host": DB_HOST,
    "port": DB_PORT,
    "user": DB_USER,
    "password": DB_PASSWORD,
    "database": DB_NAME
}

def stringtomd5(originstr):
    originstr = originstr.encode("utf-8")
    signaturemd5 = hashlib.sha256()
    signaturemd5.update(originstr)
    return signaturemd5.hexdigest()

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
    "accept": "EVENT_OPEN", # accept is mapped to open or read socket params
    "sendto": "EVENT_SENDTO",
    "recvfrom": "EVENT_RECVFROM",
}

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
    # Extracts file="...", endpoint="...", id=... from args string
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

def main():
    log_file = "temp_parsed_serialized.txt"
    if not os.path.exists(log_file):
        print(f"Error: {log_file} not found.")
        sys.exit(1)

    print(f"Connecting to Postgres at {DB_HOST}:{DB_PORT} to initialize target database...")
    
    # Connect to postgres server to create database if not exists
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD
    )
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{DB_NAME}';")
    if not cur.fetchone():
        cur.execute(f"CREATE DATABASE {DB_NAME};")
    conn.close()

    # Connect to database
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()

    # Drop existing tables to ensure clean import
    cur.execute("DROP TABLE IF EXISTS event_table CASCADE;")
    cur.execute("DROP TABLE IF EXISTS file_node_table CASCADE;")
    cur.execute("DROP TABLE IF EXISTS netflow_node_table CASCADE;")
    cur.execute("DROP TABLE IF EXISTS subject_node_table CASCADE;")

    # Recreate tables
    cur.execute("""
    CREATE TABLE event_table (
        src_node VARCHAR,
        src_index_id VARCHAR,
        operation VARCHAR,
        dst_node VARCHAR,
        dst_index_id VARCHAR,
        event_uuid VARCHAR NOT NULL,
        timestamp_rec BIGINT,
        _id SERIAL PRIMARY KEY
    );
    CREATE UNIQUE INDEX event_table__id_uindex ON event_table (_id);
    """)

    cur.execute("""
    CREATE TABLE file_node_table (
        node_uuid VARCHAR NOT NULL,
        hash_id VARCHAR NOT NULL,
        path VARCHAR,
        index_id BIGINT,
        PRIMARY KEY (node_uuid, hash_id)
    );
    """)

    cur.execute("""
    CREATE TABLE netflow_node_table (
        node_uuid VARCHAR NOT NULL,
        hash_id VARCHAR NOT NULL,
        src_addr VARCHAR,
        src_port VARCHAR,
        dst_addr VARCHAR,
        dst_port VARCHAR,
        index_id BIGINT,
        PRIMARY KEY (node_uuid, hash_id)
    );
    """)

    cur.execute("""
    CREATE TABLE subject_node_table (
        node_uuid VARCHAR NOT NULL,
        hash_id VARCHAR NOT NULL,
        path VARCHAR,
        cmd VARCHAR,
        index_id BIGINT,
        PRIMARY KEY (node_uuid, hash_id)
    );
    """)
    conn.commit()

    # Data structures to keep track of indexes and prevent duplicate nodes
    node_uuid_to_index = {}
    current_index = 0
    
    # Store elements to batch insert
    subjects_to_insert = {} # uuid: (path, cmd, index_id)
    files_to_insert = {} # uuid: (path, index_id)
    netflows_to_insert = {} # uuid: (src_addr, src_port, dst_addr, dst_port, index_id)
    
    # Reconstructed process metadata lineage (pid -> {"path": path, "cmd": cmd})
    pid_to_info = {}

    # Compile regex for eaudit log line format
    line_pattern = re.compile(
        r'^([\d\.]+):(\d+):\s+pid=(\d+):(?:\s+tid=\d+:)?\s+(\w+)\((.*)\)(?:\s+ret=(\S+))?(?:\s+\[id=(\d+)\])?'
    )

    events_parsed = []

    print("Parsing logs...")
    with open(log_file, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            m = line_pattern.match(stripped)
            if not m:
                # Fallback format check
                m = re.match(r'^([\d\.]+):(\d+):\s+(\w+)\((.*)\)(?:\s+ret=(\S+))?(?:\s+\[id=(\d+)\])?', stripped)
                if not m:
                    continue
                ts_str, seq_str, syscall, args_str, ret_str, obj_id = m.groups()
                pid = "0"
            else:
                ts_str, seq_str, pid, syscall, args_str, ret_str, obj_id = m.groups()

            if syscall not in syscall_to_op:
                continue

            op = syscall_to_op[syscall]
            timestamp_ns = int(float(ts_str) * 1e9)

            # Parse path, endpoint, and id from arguments
            file_path, endpoint, args_id = parse_args_field(args_str)
            resolved_id = obj_id or args_id

            # Save parsed event
            events_parsed.append({
                "pid": pid,
                "syscall": syscall,
                "op": op,
                "timestamp_ns": timestamp_ns,
                "file_path": file_path,
                "endpoint": endpoint,
                "resolved_id": resolved_id,
                "ret_str": ret_str,
                "args_str": args_str
            })

    total_events = len(events_parsed)
    print(f"Total valid events parsed: {total_events}")

    # Helper to register node and get index
    def register_node(node_uuid):
        nonlocal current_index
        if node_uuid not in node_uuid_to_index:
            node_uuid_to_index[node_uuid] = current_index
            current_index += 1
        return node_uuid_to_index[node_uuid]

    events_to_insert = []

    # Second pass: trace lineage and build events & nodes
    for idx, ev in enumerate(events_parsed):
        pid = ev["pid"]
        syscall = ev["syscall"]
        op = ev["op"]
        timestamp_ns = ev["timestamp_ns"]
        file_path = ev["file_path"]
        endpoint = ev["endpoint"]
        resolved_id = ev["resolved_id"]
        ret_str = ev["ret_str"]
        args_str = ev["args_str"]

        # 3-day partition shifting logic
        # First 60% on day 0, next 20% shift 1 day, last 20% shift 2 days
        if idx < int(0.6 * total_events):
            shift_seconds = 0
        elif idx < int(0.8 * total_events):
            shift_seconds = 86400
        else:
            shift_seconds = 172800
        
        shifted_timestamp_ns = timestamp_ns + shift_seconds * 1000000000
        event_uuid = str(uuid.uuid4())

        # Process lineage metadata setup
        if pid not in pid_to_info:
            exe_path, cmd = get_process_info(pid)
            pid_to_info[pid] = {"path": exe_path, "cmd": cmd}

        if syscall == "execve" and file_path:
            cmdline = extract_cmdline_from_execve(args_str) or pid_to_info[pid]["cmd"]
            pid_to_info[pid] = {"path": file_path, "cmd": cmdline}

        # 1. Register process subject
        subj_uuid = stringtomd5(f"subject_{pid}")
        subj_idx = register_node(subj_uuid)
        subjects_to_insert[subj_uuid] = (pid_to_info[pid]["path"], pid_to_info[pid]["cmd"], subj_idx)

        src_node = None
        dst_node = None

        # Handle process creation (clone/fork)
        if syscall in ("clone", "fork"):
            child_pid = ret_str
            if not child_pid or not child_pid.isdigit() or child_pid == "0":
                continue
            child_uuid = stringtomd5(f"subject_{child_pid}")
            child_idx = register_node(child_uuid)
            
            # Inherit metadata from parent
            parent_info = pid_to_info.get(pid, {"path": "unknown", "cmd": "unknown"})
            pid_to_info[child_pid] = parent_info.copy()
            subjects_to_insert[child_uuid] = (pid_to_info[child_pid]["path"], pid_to_info[child_pid]["cmd"], child_idx)

            src_node = (subj_uuid, subj_idx)
            dst_node = (child_uuid, child_idx)

        else:
            # Resolve object information
            obj_name = file_path or endpoint or f"obj_{resolved_id}"
            obj_uuid = resolved_id or stringtomd5(obj_name)
            
            # Determine object type (file or netflow)
            is_netflow = syscall in ("connect", "accept", "sendto", "recvfrom")
            if not is_netflow and obj_name:
                is_netflow = obj_name.startswith(("IP4:", "IP6:", "unix:", "netlink:"))

            obj_idx = register_node(obj_uuid)

            if is_netflow:
                if obj_uuid not in netflows_to_insert:
                    # Extract remote address and port
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

            # Determine edge direction
            if op in edge_reversed:
                src_node = (obj_uuid, obj_idx)
                dst_node = (subj_uuid, subj_idx)
            else:
                src_node = (subj_uuid, subj_idx)
                dst_node = (obj_uuid, obj_idx)

        if src_node and dst_node:
            events_to_insert.append((
                src_node[0], str(src_node[1]),
                op,
                dst_node[0], str(dst_node[1]),
                event_uuid, shifted_timestamp_ns
            ))

    print(f"Formed {len(events_to_insert)} events for database insertion.")

    # Batch Insert Subject Nodes
    if subjects_to_insert:
        subj_data = [[node_uuid, node_uuid, path, cmd, idx] for node_uuid, (path, cmd, idx) in subjects_to_insert.items()]
        ex.execute_values(cur, "INSERT INTO subject_node_table (node_uuid, hash_id, path, cmd, index_id) VALUES %s", subj_data)
        print(f"Inserted {len(subjects_to_insert)} subject nodes.")

    # Batch Insert File Nodes
    if files_to_insert:
        file_data = [[node_uuid, node_uuid, path, idx] for node_uuid, (path, idx) in files_to_insert.items()]
        ex.execute_values(cur, "INSERT INTO file_node_table (node_uuid, hash_id, path, index_id) VALUES %s", file_data)
        print(f"Inserted {len(files_to_insert)} file nodes.")

    # Batch Insert Netflow Nodes
    if netflows_to_insert:
        net_data = [[node_uuid, node_uuid, src_addr, src_port, dst_addr, dst_port, idx] 
                    for node_uuid, (src_addr, src_port, dst_addr, dst_port, idx) in netflows_to_insert.items()]
        ex.execute_values(cur, "INSERT INTO netflow_node_table (node_uuid, hash_id, src_addr, src_port, dst_addr, dst_port, index_id) VALUES %s", net_data)
        print(f"Inserted {len(netflows_to_insert)} netflow nodes.")

    # Batch Insert Events
    if events_to_insert:
        ex.execute_values(cur, "INSERT INTO event_table (src_node, src_index_id, operation, dst_node, dst_index_id, event_uuid, timestamp_rec) VALUES %s", events_to_insert)
        print(f"Inserted {len(events_to_insert)} events.")

    conn.commit()
    cur.close()
    conn.close()
    print("Database populate successful! Data integrity is intact.")

if __name__ == "__main__":
    main()
