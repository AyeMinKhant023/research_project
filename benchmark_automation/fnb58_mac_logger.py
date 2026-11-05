#!/usr/bin/env python3
"""
High-Speed Cross-Platform Power Logger for FNIRSI FNB58 on macOS/Linux.
Directly captures 64-byte HID packets (Report ID 0xaa), sends periodic 0x83 keep-alive
every 0.8s so streaming never stops after 1.5s, decodes V & I, and writes to output file.
"""

import sys
import os
import time
import argparse
import signal

# Add logger dir to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGGER_DIR = os.path.join(SCRIPT_DIR, "..", "flower_project", "fnirsi-usb-power-data-logger")
sys.path.append(LOGGER_DIR)
import fnirsi_logger


def decode_packet_to_file(data: bytes, calculate_crc, time_interval: float, alpha: float, state: dict, out_fp) -> int:
    """Decodes a 64-byte FNB58 packet (4 samples) and writes directly to out_fp."""
    if len(data) < 64 or data[1] != 0x04:
        return 0

    if calculate_crc:
        actual_checksum = data[-1]
        expected_checksum = calculate_crc(bytearray(data[1:-1]))
        if actual_checksum != expected_checksum:
            return 0

    t0 = time.time() - 4 * time_interval
    samples_written = 0
    for i in range(4):
        offset = 2 + 15 * i
        voltage = (
            data[offset + 3] * 16777216
            + data[offset + 2] * 65536
            + data[offset + 1] * 256
            + data[offset + 0]
        ) / 100000.0
        current = (
            data[offset + 7] * 16777216
            + data[offset + 6] * 65536
            + data[offset + 5] * 256
            + data[offset + 4]
        ) / 100000.0
        dp = (data[offset + 8] + data[offset + 9] * 256) / 1000.0
        dn = (data[offset + 10] + data[offset + 11] * 256) / 1000.0
        temp_C = (data[offset + 13] + data[offset + 14] * 256) / 10.0

        if state["temp_ema"] is not None:
            state["temp_ema"] = temp_C * (1.0 - alpha) + state["temp_ema"] * alpha
        else:
            state["temp_ema"] = temp_C

        power = voltage * current
        state["energy"] += power * time_interval
        state["capacity"] += current * time_interval
        t = t0 + i * time_interval

        out_fp.write(
            f"{t:.3f} {i} {voltage:7.5f} {current:7.5f} {dp:5.3f} {dn:5.3f} "
            f"{state['temp_ema']:6.3f} {state['energy']:.6f} {state['capacity']:.6f}\n"
        )
        samples_written += 1

    return samples_written


def send_init_handshake(h, pkt_81: bytes, pkt_82: bytes):
    """Send 3-packet FNB58 start-streaming handshake with safe inter-packet spacing."""
    h.write(pkt_81)
    time.sleep(0.02)
    h.write(pkt_82)
    time.sleep(0.02)
    h.write(pkt_82)
    time.sleep(0.02)


def main():
    parser = argparse.ArgumentParser(description="FNIRSI FNB58 Logger for Mac")
    parser.add_argument("--output", default="mydata.txt", help="Output file path (default: mydata.txt)")
    parser.add_argument("--duration", type=float, default=0.0, help="Optional duration in seconds (0 = until SIGINT/SIGTERM)")
    args = parser.parse_args()

    import hid

    devices = hid.enumerate(0x2E3C, 0x5558)
    if not devices:
        print("Error: FNB58 device (0x2E3C:0x5558) not found via HID.", file=sys.stderr)
        sys.exit(1)

    h = hid.device()
    h.open_path(devices[0]["path"])

    # FNB58 uses HID vendor constant 0xaa as byte 0 of the 64-byte output report
    pkt_81 = bytes([0xaa, 0x81] + [0x00] * 61 + [0x8e])
    pkt_82 = bytes([0xaa, 0x82] + [0x00] * 61 + [0x96])
    pkt_83 = bytes([0xaa, 0x83] + [0x00] * 61 + [0x9e])

    send_init_handshake(h, pkt_81, pkt_82)

    crc_fun = fnirsi_logger.setup_crc()

    running = True
    def sig_handler(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out_fp = open(args.output, "w")
    out_fp.write("\ntimestamp sample_in_packet voltage_V current_A dp_V dn_V temp_C_ema energy_Ws capacity_As\n")
    out_fp.flush()

    time_interval = 0.01
    alpha = 0.9
    refresh_interval = 0.8  # FNB58 requires 0x83 keep-alive every <= 1.0s
    next_keepalive = time.time() + refresh_interval
    start_wall = time.time()
    last_data_time = start_wall

    state = {"energy": 0.0, "capacity": 0.0, "temp_ema": None}
    packets_since_flush = 0

    try:
        while running:
            if args.duration > 0 and (time.time() - start_wall) >= args.duration:
                break

            now = time.time()
            if now >= next_keepalive:
                h.write(pkt_83)
                next_keepalive = now + refresh_interval

            data = bytes(h.read(64, timeout_ms=50))
            if data:
                n = decode_packet_to_file(data, crc_fun, time_interval, alpha, state, out_fp)
                if n > 0:
                    last_data_time = time.time()
                    packets_since_flush += 1
                    if packets_since_flush >= 10:
                        out_fp.flush()
                        packets_since_flush = 0
            else:
                if time.time() - last_data_time > 1.6:
                    send_init_handshake(h, pkt_81, pkt_82)
                    last_data_time = time.time()
                    next_keepalive = last_data_time + refresh_interval
    finally:
        out_fp.flush()
        out_fp.close()
        h.close()


if __name__ == "__main__":
    main()
