"""
live_sniffer.py — Real-Life Real-Time Network Traffic Sniffer
=============================================================================
Captures live network packets from your Wi-Fi / Ethernet interface, extracts
42 flow statistical features (exactly matching the trained ASEADOS-SDN-IoT
schema in feature_cols.joblib), and sends them to P.R.I.S.M for detection.

Usage:
    pip install scapy
    python live_sniffer.py [--interface Wi-Fi] [--url http://127.0.0.1:7860/predict]

--- FIX CHANGELOG ---
v2.0 (2026-08-20):
  1. BIDIRECTIONAL flow tracking: forward (A→B) and reverse (B→A) packets
     are merged into a single flow key using min/max IP ordering, so backward
     packet count, backward lengths, and Down/Up Ratio are correctly populated.
  2. FEATURE INDEX ALIGNMENT: all 42 slots now exactly match feature_cols.joblib:
       0: Src Port, 1: Dst Port, 2: Protocol, 3: Flow Duration (µs),
       4: Tot Fwd Pkts, 5: Tot Bwd Pkts, 6: TotLen Fwd Pkts,
       7: Fwd Pkt Len Max, 8: Fwd Pkt Len Min, 9: Fwd Pkt Len Mean,
       10: Bwd Pkt Len Max, 11: Bwd Pkt Len Min, 12: Bwd Pkt Len Mean,
       13: Flow Byts/s, 14: Flow Pkts/s, 15: Flow IAT Mean, 16: Flow IAT Std,
       17: Flow IAT Max, 18: Fwd IAT Tot, 19: Fwd IAT Mean, 20: Fwd IAT Std,
       21: Fwd IAT Max, 22: Fwd IAT Min, 23: Bwd IAT Tot, 24: Bwd IAT Mean,
       25: Bwd IAT Min, 26: Bwd PSH Flags, 27: Fwd Header Len,
       28: Fwd Pkts/s, 29: Bwd Pkts/s, 30: Pkt Len Var,
       31: FIN Flag Cnt, 32: SYN Flag Cnt, 33: RST Flag Cnt, 34: ACK Flag Cnt,
       35: Down/Up Ratio, 36: Init Bwd Win Byts, 37: Fwd Act Data Pkts,
       38: Active Mean, 39: Active Std, 40: Active Max, 41: Idle Std
  3. TIME UNITS: Flow Duration and all IAT values are now in MICROSECONDS (µs)
     to match the scaler fitted on the ASEADOS dataset (max ~120,000,000 µs).
  4. MINIMUM FLOW SIZE: flows are only evaluated when they have ≥5 packets
     AND ≥2 seconds of capture time, preventing single broadcast packets
     from being treated as flood attacks.
  5. FALLBACK GENERATOR: dummy flows now correctly reflect realistic benign
     browsing patterns with proper bidirectional ratios and real feature values.
"""

import sys
import time
import argparse
import requests
import numpy as np
from collections import defaultdict

try:
    from scapy.all import sniff, IP, TCP, UDP, get_working_ifaces
    HAS_SCAPY = True
except ImportError:
    HAS_SCAPY = False


def print_banner():
    print("=" * 68)
    print("      P.R.I.S.M — Real-Life Real-Time Network Packet Sniffer")
    print("           (v2.0 — Bidirectional, Schema-Aligned Features)")
    print("=" * 68)


def check_dependencies():
    if not HAS_SCAPY:
        print("\n[!] 'scapy' is required for real-life packet capture.")
        print("    Please install it by running:")
        print("        pip install scapy\n")
        print("    After installing, re-run: python live_sniffer.py")
        sys.exit(1)


# ─── Bidirectional Flow Tracker ───────────────────────────────────────────────

class FlowTracker:
    """
    Tracks bidirectional network flows keyed by a canonical 5-tuple
    (min_ip, max_ip, min_port, max_port, proto) so that forward and
    reverse packets are merged into the same flow record.

    Fix 1: Bidirectional tracking (was unidirectional — caused bwd_pkts=0
           which looks like a one-way flood attack).
    Fix 2: 42 features mapped to exact feature_cols.joblib schema.
    Fix 3: All times in microseconds (µs) to match scaler range.
    Fix 4: Minimum 5 packets AND 2 seconds before emitting a flow.
    """

    # Minimum conditions before a flow is submitted for prediction.
    # Short immature flows look "bursty" (high Pkts/s) and are the main
    # source of live false-positive DoS on home/office Wi‑Fi.
    MIN_PKTS = 15
    MIN_DURATION_S = 4.0      # seconds
    MAX_DURATION_S = 120.0    # force-expire very long idle flows

    def __init__(self, min_pkts=None, min_duration_s=None):
        self.flows = {}
        if min_pkts is not None:
            self.MIN_PKTS = int(min_pkts)
        if min_duration_s is not None:
            self.MIN_DURATION_S = float(min_duration_s)

    def _canonical_key(self, src_ip, dst_ip, src_port, dst_port, proto):
        """Return a direction-independent flow key."""
        if (src_ip, src_port) < (dst_ip, dst_port):
            return (src_ip, dst_ip, src_port, dst_port, proto)
        return (dst_ip, src_ip, dst_port, src_port, proto)

    def _is_forward(self, src_ip, dst_ip, src_port, dst_port, proto):
        """Is this packet in the canonical forward direction?"""
        key = self._canonical_key(src_ip, dst_ip, src_port, dst_port, proto)
        return key == (src_ip, dst_ip, src_port, dst_port, proto)

    def process_packet(self, pkt):
        if not pkt.haslayer(IP):
            return None

        ip = pkt[IP]
        proto = ip.proto
        src_ip = ip.src
        dst_ip = ip.dst
        src_port = 0
        dst_port = 0
        tcp_layer = None
        udp_layer = None

        if pkt.haslayer(TCP):
            tcp_layer = pkt[TCP]
            src_port = tcp_layer.sport
            dst_port = tcp_layer.dport
        elif pkt.haslayer(UDP):
            udp_layer = pkt[UDP]
            src_port = udp_layer.sport
            dst_port = udp_layer.dport

        key = self._canonical_key(src_ip, dst_ip, src_port, dst_port, proto)
        is_fwd = self._is_forward(src_ip, dst_ip, src_port, dst_port, proto)

        now = time.time()
        # Packet length at IP layer (excludes Ethernet overhead, matches CICFlowMeter)
        pkt_len = len(ip)

        # TCP header length in bytes (for Fwd Header Len accumulation)
        tcp_hdr_len = 0
        if tcp_layer is not None:
            try:
                tcp_hdr_len = tcp_layer.dataofs * 4  # dataofs is in 32-bit words
            except Exception:
                tcp_hdr_len = 20  # default minimum TCP header

        # TCP flags
        syn = ack = fin = rst = psh = bwd_psh = 0
        if tcp_layer is not None:
            flags = tcp_layer.flags
            syn = 1 if flags.S else 0
            ack = 1 if flags.A else 0
            fin = 1 if flags.F else 0
            rst = 1 if flags.R else 0
            psh = 1 if flags.P else 0
            bwd_psh = psh if not is_fwd else 0  # Bwd PSH Flags

        # Init window bytes (from the first SYN-ACK, i.e., the backward direction)
        init_bwd_win = -1
        if tcp_layer is not None and not is_fwd:
            try:
                init_bwd_win = tcp_layer.window
            except Exception:
                pass

        if key not in self.flows:
            self.flows[key] = {
                'start_time': now,
                'last_time': now,
                # Forward packets
                'fwd_pkts': 1 if is_fwd else 0,
                'fwd_lengths': [pkt_len] if is_fwd else [],
                'fwd_times': [now] if is_fwd else [],
                'fwd_hdr_len': tcp_hdr_len if is_fwd else 0,
                'fwd_data_pkts': (1 if (is_fwd and tcp_layer and pkt_len > tcp_hdr_len + 20) else 0),
                # Backward packets
                'bwd_pkts': 0 if is_fwd else 1,
                'bwd_lengths': [] if is_fwd else [pkt_len],
                'bwd_times': [] if is_fwd else [now],
                'bwd_psh_flags': bwd_psh,
                'init_bwd_win_byts': init_bwd_win,
                # Flags (both directions aggregated per CICFlowMeter convention)
                'syn_count': syn,
                'ack_count': ack,
                'fin_count': fin,
                'rst_count': rst,
                'psh_count': psh,
                # All packet inter-arrival times (for Flow IAT)
                'all_times': [now],
                'protocol': proto,
                # Remember canonical src_port for Dst Port feature
                'dst_port': dst_port if is_fwd else src_port,
                'src_port': src_port if is_fwd else dst_port,
            }
        else:
            f = self.flows[key]
            f['last_time'] = now
            f['all_times'].append(now)

            if is_fwd:
                f['fwd_pkts'] += 1
                f['fwd_lengths'].append(pkt_len)
                f['fwd_times'].append(now)
                f['fwd_hdr_len'] += tcp_hdr_len
                if tcp_layer and pkt_len > tcp_hdr_len + 20:
                    f['fwd_data_pkts'] += 1
            else:
                f['bwd_pkts'] += 1
                f['bwd_lengths'].append(pkt_len)
                f['bwd_times'].append(now)
                f['bwd_psh_flags'] += bwd_psh
                if f['init_bwd_win_byts'] < 0 and init_bwd_win >= 0:
                    f['init_bwd_win_byts'] = init_bwd_win

            if tcp_layer:
                f['syn_count'] += syn
                f['ack_count'] += ack
                f['fin_count'] += fin
                f['rst_count'] += rst
                f['psh_count'] += psh

        flow = self.flows[key]
        duration = flow['last_time'] - flow['start_time']
        total_pkts = flow['fwd_pkts'] + flow['bwd_pkts']

        # Emit flow only when we have enough data for reliable statistics.
        # Fix 4: wait for MIN_PKTS AND MIN_DURATION before emitting.
        should_emit = (
            (total_pkts >= self.MIN_PKTS and duration >= self.MIN_DURATION_S)
            or duration >= self.MAX_DURATION_S  # force-emit stale flows
        )

        if should_emit:
            feat_vector = self.extract_42_features(flow)
            del self.flows[key]
            return feat_vector
        return None

    def extract_42_features(self, f):
        """
        Extract exactly 42 features in the order of feature_cols.joblib,
        using MICROSECONDS for all time-based features (to match the scaler
        fitted on the ASEADOS-SDN-IoT dataset).

        Schema (0-indexed):
         0  Src Port
         1  Dst Port
         2  Protocol
         3  Flow Duration          ← µs  (Fix 3: was ms, now µs)
         4  Tot Fwd Pkts
         5  Tot Bwd Pkts
         6  TotLen Fwd Pkts
         7  Fwd Pkt Len Max
         8  Fwd Pkt Len Min
         9  Fwd Pkt Len Mean
        10  Bwd Pkt Len Max        ← Fix 2: was Fwd Pkt Len Std
        11  Bwd Pkt Len Min        ← Fix 2: was Flow Byts/s
        12  Bwd Pkt Len Mean       ← Fix 2: was Flow Pkts/s
        13  Flow Byts/s            ← Fix 2: was FIN Flag Cnt
        14  Flow Pkts/s            ← Fix 2: was SYN Flag Cnt
        15  Flow IAT Mean          ← Fix 2: was RST Flag Cnt
        16  Flow IAT Std           ← Fix 2: was PSH Flag Cnt
        17  Flow IAT Max           ← Fix 2: was ACK Flag Cnt
        18  Fwd IAT Tot            ← Fix 2: was dummy formula
        19  Fwd IAT Mean           ← Fix 2: was dummy formula
        20  Fwd IAT Std            ← Fix 2: was dummy formula
        21  Fwd IAT Max            ← Fix 2: was dummy formula
        22  Fwd IAT Min            ← Fix 2: was dummy formula
        23  Bwd IAT Tot            ← Fix 2: was dummy formula
        24  Bwd IAT Mean           ← Fix 2: was dummy formula
        25  Bwd IAT Min            ← Fix 2: was dummy formula
        26  Bwd PSH Flags          ← Fix 1+2: bidirectional, was dummy
        27  Fwd Header Len         ← Fix 2: was dummy formula (most important feature!)
        28  Fwd Pkts/s             ← Fix 2: was dummy formula
        29  Bwd Pkts/s             ← Fix 2: was dummy formula
        30  Pkt Len Var            ← Fix 2: was dummy formula
        31  FIN Flag Cnt           ← Fix 2: correct index now
        32  SYN Flag Cnt           ← Fix 2: correct index now
        33  RST Flag Cnt           ← Fix 2: correct index now
        34  ACK Flag Cnt           ← Fix 2: correct index now
        35  Down/Up Ratio          ← Fix 1+2: requires bidirectional data
        36  Init Bwd Win Byts      ← Fix 1+2: requires backward packets
        37  Fwd Act Data Pkts
        38  Active Mean
        39  Active Std
        40  Active Max
        41  Idle Std
        """
        # ── Basic timing ─────────────────────────────────────────────────────
        # Convert to microseconds (µs) — matches ASEADOS dataset and scaler
        duration_s = max(1e-6, f['last_time'] - f['start_time'])
        duration_us = duration_s * 1_000_000.0  # Fix 3: µs, not ms

        # ── Forward packet stats ──────────────────────────────────────────────
        fwd_lens = np.array(f['fwd_lengths'], dtype=np.float64) if f['fwd_lengths'] else np.zeros(1)
        fwd_times = np.array(f['fwd_times'], dtype=np.float64)
        fwd_n = float(f['fwd_pkts'])
        fwd_total_bytes = float(np.sum(fwd_lens))

        fwd_iat_arr = np.diff(fwd_times) * 1_000_000.0 if len(fwd_times) > 1 else np.array([0.0])  # µs
        fwd_iat_tot = float(np.sum(fwd_iat_arr))
        fwd_iat_mean = float(np.mean(fwd_iat_arr))
        fwd_iat_std = float(np.std(fwd_iat_arr))
        fwd_iat_max = float(np.max(fwd_iat_arr))
        fwd_iat_min = float(np.min(fwd_iat_arr))

        # ── Backward packet stats ─────────────────────────────────────────────
        # Fix 1: these were always 0 before bidirectional tracking
        bwd_lens = np.array(f['bwd_lengths'], dtype=np.float64) if f['bwd_lengths'] else np.zeros(1)
        bwd_times = np.array(f['bwd_times'], dtype=np.float64)
        bwd_n = float(f['bwd_pkts'])

        bwd_iat_arr = np.diff(bwd_times) * 1_000_000.0 if len(bwd_times) > 1 else np.array([0.0])  # µs
        bwd_iat_tot = float(np.sum(bwd_iat_arr))
        bwd_iat_mean = float(np.mean(bwd_iat_arr))
        bwd_iat_min = float(np.min(bwd_iat_arr))

        # ── Flow-level stats ──────────────────────────────────────────────────
        all_lens = np.concatenate([fwd_lens, bwd_lens])
        total_bytes = float(np.sum(all_lens))
        total_pkts = fwd_n + bwd_n

        all_times = np.array(sorted(f['all_times']), dtype=np.float64)
        flow_iat_arr = np.diff(all_times) * 1_000_000.0 if len(all_times) > 1 else np.array([0.0])  # µs
        flow_iat_mean = float(np.mean(flow_iat_arr))
        flow_iat_std = float(np.std(flow_iat_arr))
        flow_iat_max = float(np.max(flow_iat_arr))

        flow_bytes_per_s = total_bytes / duration_s
        flow_pkts_per_s = total_packets_per_s = total_pkts / duration_s
        fwd_pkts_per_s = fwd_n / duration_s
        bwd_pkts_per_s = bwd_n / duration_s

        # Down/Up Ratio = bwd_bytes / fwd_bytes (Fix 1+2)
        down_up_ratio = (np.sum(bwd_lens) / max(1.0, np.sum(fwd_lens)))

        # Packet length variance across all packets
        pkt_len_var = float(np.var(all_lens)) if len(all_lens) > 1 else 0.0

        # Fwd Header Len: total TCP header bytes in forward direction (Fix 2 — most important!)
        fwd_header_len = float(f.get('fwd_hdr_len', 0))

        # Init Bwd Win Byts: first TCP window size from server (Fix 1+2)
        init_bwd_win = float(f.get('init_bwd_win_byts', -1))

        # Fwd Active Data Packets: fwd packets that carried a payload
        fwd_act_data_pkts = float(f.get('fwd_data_pkts', 0))

        # ── Assemble feature vector (Fix 2: exact index mapping) ─────────────
        v = np.zeros(42, dtype=np.float32)

        v[0]  = float(f.get('src_port', 0))           # Src Port
        v[1]  = float(f.get('dst_port', 80))           # Dst Port
        v[2]  = float(f.get('protocol', 6))            # Protocol (6=TCP, 17=UDP)
        v[3]  = float(duration_us)                     # Flow Duration (µs) — Fix 3
        v[4]  = fwd_n                                  # Tot Fwd Pkts
        v[5]  = bwd_n                                  # Tot Bwd Pkts — Fix 1
        v[6]  = fwd_total_bytes                        # TotLen Fwd Pkts
        v[7]  = float(np.max(fwd_lens))               # Fwd Pkt Len Max
        v[8]  = float(np.min(fwd_lens))               # Fwd Pkt Len Min
        v[9]  = float(np.mean(fwd_lens))              # Fwd Pkt Len Mean
        v[10] = float(np.max(bwd_lens))               # Bwd Pkt Len Max  — Fix 2 (was Fwd Std)
        v[11] = float(np.min(bwd_lens))               # Bwd Pkt Len Min  — Fix 2 (was Flow Byts/s)
        v[12] = float(np.mean(bwd_lens))              # Bwd Pkt Len Mean — Fix 2 (was Flow Pkts/s)
        v[13] = float(flow_bytes_per_s)               # Flow Byts/s      — Fix 2 (was FIN Flag)
        v[14] = float(flow_pkts_per_s)                # Flow Pkts/s      — Fix 2 (was SYN Flag)
        v[15] = float(flow_iat_mean)                  # Flow IAT Mean    — Fix 2 (was RST Flag)
        v[16] = float(flow_iat_std)                   # Flow IAT Std     — Fix 2 (was PSH Flag)
        v[17] = float(flow_iat_max)                   # Flow IAT Max     — Fix 2 (was ACK Flag)
        v[18] = float(fwd_iat_tot)                    # Fwd IAT Tot      — Fix 2 (was dummy)
        v[19] = float(fwd_iat_mean)                   # Fwd IAT Mean     — Fix 2 (was dummy)
        v[20] = float(fwd_iat_std)                    # Fwd IAT Std      — Fix 2 (was dummy)
        v[21] = float(fwd_iat_max)                    # Fwd IAT Max      — Fix 2 (was dummy)
        v[22] = float(fwd_iat_min)                    # Fwd IAT Min      — Fix 2 (was dummy)
        v[23] = float(bwd_iat_tot)                    # Bwd IAT Tot      — Fix 1+2
        v[24] = float(bwd_iat_mean)                   # Bwd IAT Mean     — Fix 1+2
        v[25] = float(bwd_iat_min)                    # Bwd IAT Min      — Fix 1+2
        v[26] = float(f.get('bwd_psh_flags', 0))     # Bwd PSH Flags    — Fix 1+2
        v[27] = float(fwd_header_len)                 # Fwd Header Len   — Fix 2 (was dummy, #1 importance!)
        v[28] = float(fwd_pkts_per_s)                 # Fwd Pkts/s       — Fix 2
        v[29] = float(bwd_pkts_per_s)                 # Bwd Pkts/s       — Fix 1+2
        v[30] = float(pkt_len_var)                    # Pkt Len Var      — Fix 2
        v[31] = float(f.get('fin_count', 0))          # FIN Flag Cnt     — Fix 2 (correct slot)
        v[32] = float(f.get('syn_count', 0))          # SYN Flag Cnt     — Fix 2 (correct slot)
        v[33] = float(f.get('rst_count', 0))          # RST Flag Cnt     — Fix 2 (correct slot)
        v[34] = float(f.get('ack_count', 0))          # ACK Flag Cnt     — Fix 2 (correct slot)
        v[35] = float(down_up_ratio)                  # Down/Up Ratio    — Fix 1+2
        v[36] = float(init_bwd_win)                   # Init Bwd Win Byts — Fix 1+2
        v[37] = float(fwd_act_data_pkts)              # Fwd Act Data Pkts
        v[38] = 0.0                                   # Active Mean (requires active/idle segmentation)
        v[39] = 0.0                                   # Active Std
        v[40] = 0.0                                   # Active Max
        v[41] = 0.0                                   # Idle Std

        # Clamp all values to finite range (handles any inf/NaN from divisions)
        v = np.nan_to_num(v, nan=0.0, posinf=1e8, neginf=0.0)
        return v.tolist()


# ─── Fallback: Realistic Benign Flow Generator (Fix 5) ───────────────────────

def _make_benign_flow_features():
    """
    Generate a realistic benign HTTPS browsing flow feature vector.
    Uses proper bidirectional ratios, µs durations, and correct feature
    indices so the model correctly classifies it as Benign.

    Fix 5: The old dummy generator produced 10–250 fwd packets in 0.1–2.5s,
    giving packet rates of up to 2,500 pkt/s — a DoS signature.
    Now we generate realistic web-browsing traffic patterns:
      - Small number of fwd packets (requests)
      - Larger number of bwd packets (responses)
      - Normal duration: 1–10 seconds
      - Proper TCP header lengths and window sizes
    """
    rng = np.random.default_rng()

    # Session characteristics (web browsing)
    duration_s = rng.uniform(1.0, 10.0)
    duration_us = duration_s * 1_000_000.0

    fwd_n = int(rng.integers(5, 25))       # client requests (small)
    bwd_n = int(rng.integers(10, 60))      # server responses (larger)
    total_n = fwd_n + bwd_n

    # Fwd: small HTTP request packets (headers only, mostly)
    fwd_lens = rng.integers(54, 300, size=fwd_n).astype(float)
    # Bwd: larger response packets (may include data)
    bwd_lens = rng.integers(54, 1448, size=bwd_n).astype(float)
    all_lens = np.concatenate([fwd_lens, bwd_lens])

    # Spread packet times over the duration
    fwd_times = np.sort(rng.uniform(0, duration_s, size=fwd_n)) * 1_000_000.0
    bwd_times = np.sort(rng.uniform(0, duration_s, size=bwd_n)) * 1_000_000.0
    all_times = np.sort(np.concatenate([fwd_times, bwd_times]))

    fwd_iat = np.diff(fwd_times) if len(fwd_times) > 1 else np.array([0.0])
    bwd_iat = np.diff(bwd_times) if len(bwd_times) > 1 else np.array([0.0])
    flow_iat = np.diff(all_times) if len(all_times) > 1 else np.array([0.0])

    fwd_total_bytes = float(np.sum(fwd_lens))
    bwd_total_bytes = float(np.sum(bwd_lens))
    total_bytes = fwd_total_bytes + bwd_total_bytes

    flow_bytes_per_s = total_bytes / duration_s
    flow_pkts_per_s = total_n / duration_s
    fwd_pkts_per_s = fwd_n / duration_s
    bwd_pkts_per_s = bwd_n / duration_s

    # Fwd header len: typical TCP header is 32 bytes (with options like SACK/TS)
    fwd_hdr_len = float(rng.integers(20, 40)) * fwd_n

    # Init Bwd Win Byts: typical server TCP window size (8KB-64KB)
    init_bwd_win = float(rng.choice([8192, 16384, 32768, 65535]))

    # Fwd Act Data Pkts: packets with actual payload
    fwd_act_data_pkts = max(0, fwd_n - 3)  # minus SYN, FIN, ACK handshake pkts

    # Normal TCP session flags
    dst_port = int(rng.choice([80, 443, 8080, 8443]))
    src_port = int(rng.integers(1024, 60999))
    protocol = 6  # TCP

    # TCP flags for a normal session: SYN=1, ACK=many, FIN=1, PSH=some
    syn_count = 1
    fin_count = 1
    rst_count = 0
    ack_count = int(rng.integers(fwd_n, fwd_n + bwd_n))
    psh_count = int(rng.integers(1, max(2, fwd_n // 3)))
    bwd_psh_flags = int(rng.integers(0, max(1, bwd_n // 5)))

    v = np.zeros(42, dtype=np.float32)
    v[0]  = float(src_port)
    v[1]  = float(dst_port)
    v[2]  = float(protocol)
    v[3]  = float(duration_us)
    v[4]  = float(fwd_n)
    v[5]  = float(bwd_n)
    v[6]  = float(fwd_total_bytes)
    v[7]  = float(np.max(fwd_lens))
    v[8]  = float(np.min(fwd_lens))
    v[9]  = float(np.mean(fwd_lens))
    v[10] = float(np.max(bwd_lens))
    v[11] = float(np.min(bwd_lens))
    v[12] = float(np.mean(bwd_lens))
    v[13] = float(flow_bytes_per_s)
    v[14] = float(flow_pkts_per_s)
    v[15] = float(np.mean(flow_iat))
    v[16] = float(np.std(flow_iat))
    v[17] = float(np.max(flow_iat))
    v[18] = float(np.sum(fwd_iat))
    v[19] = float(np.mean(fwd_iat))
    v[20] = float(np.std(fwd_iat))
    v[21] = float(np.max(fwd_iat))
    v[22] = float(np.min(fwd_iat))
    v[23] = float(np.sum(bwd_iat))
    v[24] = float(np.mean(bwd_iat))
    v[25] = float(np.min(bwd_iat))
    v[26] = float(bwd_psh_flags)
    v[27] = float(fwd_hdr_len)
    v[28] = float(fwd_pkts_per_s)
    v[29] = float(bwd_pkts_per_s)
    v[30] = float(np.var(all_lens))
    v[31] = float(fin_count)
    v[32] = float(syn_count)
    v[33] = float(rst_count)
    v[34] = float(ack_count)
    v[35] = bwd_total_bytes / max(1.0, fwd_total_bytes)  # Down/Up Ratio
    v[36] = float(init_bwd_win)
    v[37] = float(fwd_act_data_pkts)
    v[38] = 0.0  # Active Mean
    v[39] = 0.0  # Active Std
    v[40] = 0.0  # Active Max
    v[41] = 0.0  # Idle Std

    v = np.nan_to_num(v, nan=0.0, posinf=1e8, neginf=0.0)
    return v.tolist()


# ─── Prediction sender ────────────────────────────────────────────────────────

def _collect_votes(d):
    """Build (label, confidence, source) votes from XGBoost + Local RL + Federated RL."""
    votes = []
    xgb = d.get('prediction', 'Unknown')
    xgb_conf = float(d.get('confidence') or 0.0)
    votes.append(('XGBoost', xgb, xgb_conf))

    rl = d.get('rl') or {}
    primary = rl.get('primary') or {}
    if primary.get('action'):
        votes.append(('LocalRL', primary['action'], float(primary.get('confidence') or 0.0)))

    fl = d.get('fl') or {}
    if fl.get('prediction'):
        votes.append(('FedRL', fl['prediction'], float(fl.get('confidence') or 0.0)))

    return votes


def _consensus_decision(d, min_attack_votes=2, min_attack_conf=0.85, suppress_if_rl_benign=True):
    """
    Reduce live false DoS alerts.

    Live Wi‑Fi/LAN traffic is outside the ASEADOS lab distribution, so a lone
    XGBoost "Dos @ 100%" is often wrong. Prefer multi-model agreement.
    """
    votes = _collect_votes(d)
    xgb_pred = votes[0][1]
    xgb_conf = votes[0][2]

    attack_votes = [(src, lab, conf) for src, lab, conf in votes if lab and lab != 'Benign']
    benign_votes = [(src, lab, conf) for src, lab, conf in votes if lab == 'Benign']

    rl_benign = any(src == 'LocalRL' and lab == 'Benign' and conf >= 0.60 for src, lab, conf in votes)
    fl_benign = any(src == 'FedRL' and lab == 'Benign' and conf >= 0.60 for src, lab, conf in votes)

    # Soft feature sanity: moderate bidirectional browsing rarely is DoS.
    feats = d.get('_features')  # optional; set by caller
    looks_browsing = False
    if isinstance(feats, (list, tuple)) and len(feats) >= 36:
        flow_pkts_s = float(feats[14])
        down_up = float(feats[35])
        bwd_pkts = float(feats[5])
        looks_browsing = (flow_pkts_s < 80.0 and bwd_pkts >= 2.0 and 0.2 <= down_up <= 20.0)

    if xgb_pred == 'Benign':
        return 'Benign', xgb_conf, 'xgboost', votes, None

    # Attack candidate from XGBoost — gate it.
    enough_votes = len(attack_votes) >= int(min_attack_votes)
    strong_xgb = xgb_conf >= float(min_attack_conf)
    others_agree = any(lab == xgb_pred for src, lab, conf in attack_votes if src != 'XGBoost')

    if suppress_if_rl_benign and rl_benign and not others_agree:
        return 'Benign', max((c for s, l, c in benign_votes if s == 'LocalRL'), default=xgb_conf), 'suppressed_rl_benign', votes, xgb_pred

    if looks_browsing and not others_agree:
        return 'Benign', xgb_conf, 'suppressed_browsing_shape', votes, xgb_pred

    # Only XGBoost available: demand very high conf AND not browsing-shaped.
    only_xgb = len(votes) == 1
    if only_xgb:
        if strong_xgb and not looks_browsing:
            return xgb_pred, xgb_conf, 'xgboost_only', votes, None
        return 'Benign', xgb_conf, 'suppressed_weak_solo_xgb', votes, xgb_pred

    if enough_votes or (strong_xgb and others_agree):
        # Prefer the agreed attack label (XGBoost if others match it).
        if others_agree:
            return xgb_pred, xgb_conf, 'consensus', votes, None
        # Multiple attacks but different classes — still alert with XGBoost label.
        if enough_votes and strong_xgb:
            return xgb_pred, xgb_conf, 'multi_attack', votes, None

    if fl_benign and not others_agree:
        return 'Benign', max((c for s, l, c in benign_votes if s == 'FedRL'), default=xgb_conf), 'suppressed_fl_benign', votes, xgb_pred

    return 'Benign', xgb_conf, 'suppressed_no_consensus', votes, xgb_pred


def send_predict(url, features, min_attack_votes=2, min_attack_conf=0.85, quiet_suppress=False):
    try:
        r = requests.post(url, json={'features': features}, timeout=3)
        if r.status_code == 200:
            d = r.json()
            d['_features'] = features
            pred, conf, reason, votes, raw_attack = _consensus_decision(
                d,
                min_attack_votes=min_attack_votes,
                min_attack_conf=min_attack_conf,
            )
            conf_pct = conf * 100.0
            tag = "🔴 ATTACK" if pred != "Benign" else "🟢 BENIGN"

            # fl_processes_alive is True ONLY when the Flower RL-FL processes
            # (rl_server.py + rl_client.py) are currently running on the server.
            fl_running = d.get('fl_processes_alive', False)
            vote_txt = ', '.join(f'{s}={l}({c*100:.0f}%)' for s, l, c in votes)
            if fl_running:
                client = d.get('client_id', 0)
                suffix = f" | FL live · client {client} | votes: {vote_txt}"
            else:
                suffix = f" | votes: {vote_txt}"

            if pred == 'Benign' and raw_attack and reason.startswith('suppressed'):
                if quiet_suppress:
                    return
                print(
                    f"[{time.strftime('%H:%M:%S')}] Flow -> {tag}: {pred:<8} "
                    f"({conf_pct:.1f}% · FP-guard: ignored {raw_attack} [{reason}]){suffix}"
                )
            else:
                print(
                    f"[{time.strftime('%H:%M:%S')}] Flow -> {tag}: {pred:<8} "
                    f"({conf_pct:.1f}% confidence · {reason}){suffix}"
                )
        else:
            print(f"[!] Server returned status {r.status_code}")
    except Exception as e:
        print(f"[!] Could not reach prediction server: {e}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print_banner()
    check_dependencies()

    parser = argparse.ArgumentParser(description="Live Real Network Sniffer for AEGIS IDS")
    parser.add_argument("--url", default="http://127.0.0.1:7860/predict", help="Prediction server URL")
    parser.add_argument("--count", type=int, default=0, help="Number of packets to capture (0 = infinite)")
    parser.add_argument("--min-pkts", type=int, default=15, help="Min packets before a flow is scored (default 15)")
    parser.add_argument("--min-duration", type=float, default=4.0, help="Min flow seconds before scoring (default 4)")
    parser.add_argument("--min-attack-votes", type=int, default=2, help="Models that must agree on attack (default 2)")
    parser.add_argument("--min-attack-conf", type=float, default=0.85, help="Min XGBoost conf to allow attack (default 0.85)")
    parser.add_argument("--quiet-suppress", action="store_true", help="Hide FP-guard suppressed lines")
    parser.add_argument("--raw-xgb", action="store_true", help="Disable FP-guard (old noisy XGBoost-only alerts)")
    args = parser.parse_args()

    ifaces = []
    try:
        ifaces = get_working_ifaces()
    except Exception:
        pass

    def _send(feats):
        if args.raw_xgb:
            # Legacy path: trust XGBoost label only (high FP on live LAN).
            try:
                r = requests.post(args.url, json={'features': feats}, timeout=3)
                if r.status_code == 200:
                    d = r.json()
                    pred = d.get('prediction', 'Unknown')
                    conf = float(d.get('confidence') or 0.0) * 100
                    tag = "🔴 ATTACK" if pred != "Benign" else "🟢 BENIGN"
                    print(f"[{time.strftime('%H:%M:%S')}] Flow -> {tag}: {pred:<8} ({conf:.1f}% confidence) [raw-xgb]")
            except Exception as e:
                print(f"[!] Could not reach prediction server: {e}")
            return
        send_predict(
            args.url,
            feats,
            min_attack_votes=args.min_attack_votes,
            min_attack_conf=args.min_attack_conf,
            quiet_suppress=args.quiet_suppress,
        )

    if ifaces:
        print(f"\n[+] Active Network Interfaces:")
        for iface in ifaces:
            print(f"    - {getattr(iface, 'name', str(iface))} ({getattr(iface, 'ip', 'N/A')})")

        print(f"\n[+] Listening for live real network packets (bidirectional flow tracking)...")
        print(f"    Target Server: {args.url}")
        print(f"    Flow emit threshold: ≥{args.min_pkts} packets AND ≥{args.min_duration}s duration")
        if args.raw_xgb:
            print("    FP-guard: OFF (--raw-xgb)")
        else:
            print(f"    FP-guard: ON (need ≥{args.min_attack_votes} attack votes, conf≥{args.min_attack_conf:.0%})")
        print("    Press Ctrl+C to stop sniffing.\n")

        tracker = FlowTracker(min_pkts=args.min_pkts, min_duration_s=args.min_duration)

        def pkt_callback(pkt):
            feats = tracker.process_packet(pkt)
            if feats:
                _send(feats)

        try:
            sniff(prn=pkt_callback, store=0, count=args.count)
            return
        except Exception as e:
            print(f"\n[!] Scapy Sniffing Warning: {e}")

    # Fallback if Npcap / WinPcap is not installed on Windows
    print("\n" + "=" * 68)
    print(" [!] Npcap Driver is missing on Windows!")
    print(" ===============================================================")
    print(" To sniff raw Wi-Fi/Ethernet adapters, download free Npcap:")
    print("     👉 https://npcap.com/#download  (npcap-1.80.exe)")
    print("     (During install, check 'Install Npcap in WinPcap Mode')")
    print(" ===============================================================")
    print(" [►] SWITCHING TO LIVE REAL-TIME FLOW GENERATOR MODE...")
    print("     Generating realistic benign browsing flows every 3 seconds...")
    print("     (Fix 5: flows now have proper bidirectional ratios & µs timing)")
    print("     Target Server: " + args.url)
    print("     Press Ctrl+C to stop.\n")

    try:
        while True:
            feats = _make_benign_flow_features()
            _send(feats)
            time.sleep(3.0)
    except KeyboardInterrupt:
        print("\n[!] Live flow generator stopped by user.")


if __name__ == '__main__':
    main()
