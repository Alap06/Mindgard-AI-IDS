"""
Mindgard — Live Network Packet Sniffer (Proxy for eBPF)
Simulates a live high-speed PCAP parsing tap for real-time inference.
"""
import time
import json
import logging
import threading
from collections import deque
import requests

try:
    from scapy.all import sniff, IP, TCP, UDP
    HAS_SCAPY = True
except ImportError:
    HAS_SCAPY = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

API_ENDPOINT = "http://localhost:8000/predict"

class LiveNetworkTap:
    """
    Live Stateful Packet Sniffer.
    Acts as a proxy for eBPF by reading raw packets, maintaining a stateful window,
    and pushing calculated network heuristics directly to the AI Engine.
    """
    def __init__(self, interface=None, pcap_file=None):
        self.interface = interface
        self.pcap_file = pcap_file
        self.packet_count = 0
        self.state_window = deque(maxlen=5000) # Memory for stateful features
    
    def process_packet(self, packet):
        """Extract L3/L4 features on the fly."""
        self.packet_count += 1
        
        if IP not in packet:
            return
            
        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
        protocol = "tcp" if TCP in packet else "udp" if UDP in packet else "icmp"
        
        # Calculate raw payload
        src_bytes = len(packet[IP].payload)
        
        # Basic stateful tracking
        now = time.time()
        self.state_window.append({"time": now, "src": src_ip, "dst": dst_ip})
        
        # Approximate behavioral 'count' (connections to same host in window)
        count = sum(1 for p in self.state_window if p["src"] == src_ip and (now - p["time"] < 2.0))
        
        payload = {
            "src_ip": src_ip,
            "timestamp": now,
            "protocol_type": protocol,
            "service": "http" if packet.haslayer(TCP) and (packet[TCP].dport == 80 or packet[TCP].sport == 80) else "other",
            "src_bytes": src_bytes,
            "dst_bytes": 0,
            "count": count,
            "serror_rate": 0.0,
            "duration": 0.0 
        }
        
        # Forward to the Fast API Endpoint
        try:
            # We don't block the sniffing thread! In prod, this goes to Kafka.
            threading.Thread(target=self._send_payload, args=(payload,)).start()
        except Exception as e:
            logging.error(f"Post error: {e}")

    def _send_payload(self, payload):
        try:
            res = requests.post(API_ENDPOINT, json=payload, timeout=2.0)
            if res.status_code != 200:
                logging.error(f"API Error ({res.status_code}): {res.text}")
                return
            data = res.json()
            if data.get("risk_score", 0) > 80:
                logging.warning(f"🚨 CRITICAL ALERT: IP {payload['src_ip']} Risk {data['risk_score']} - [{data.get('alert_level', 'UNKNOWN')}]")
            else:
                logging.info(f"Traffic OK: IP {payload['src_ip']} Risk {data.get('risk_score', 0)}")
        except requests.exceptions.RequestException:
            pass

    def run(self):
        if not HAS_SCAPY:
            logging.error("Scapy is not installed. To run the live tap: pip install scapy")
            return
            
        if self.pcap_file:
            logging.info(f"Starting Offline PCAP Replay on {self.pcap_file}...")
            sniff(offline=self.pcap_file, prn=self.process_packet, store=0)
        else:
            logging.info(f"Starting Live eBPF Proxied Sniffer on {self.interface or 'default'}...")
            sniff(iface=self.interface, prn=self.process_packet, store=0, filter="ip")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Mindgard Live Network Tap")
    parser.add_argument("--interface", type=str, help="Network interface (e.g., eth0)")
    parser.add_argument("--pcap", type=str, help="Path to offline PCAP file")
    args = parser.parse_args()
    
    tap = LiveNetworkTap(interface=args.interface, pcap_file=args.pcap)
    tap.run()