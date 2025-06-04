import argparse
import base58
import binascii
import concurrent.futures
import hashlib
import json
import logging
import os
import pickle
import signal
import string
import sys
import threading
import time
from datetime import datetime
from time import sleep

import numpy as np
import pyopencl as cl
from tqdm import tqdm

# Import wallet libraries only if available
try:
    from hdwallet import HDWallet
    from hdwallet.symbols import BTC
except ImportError:
    HDWallet = None

try:
    from btc_com import explorer as btc_explorer
except ImportError:
    btc_explorer = None

# ----------------------------
# Configuration
# ----------------------------
DEFAULT_STATE_FILE = "recovery_state.bin"
SAVE_INTERVAL = 300  # 5 minutes between state saves
GPU_BATCH_SIZE = 500000  # Keys per GPU batch
CPU_THREADS = 8
LOG_FILE = "recovery.log"
MAX_KEY_LENGTH = 256  # Maximum supported key length

def convert_masked_wif_to_hex(masked_wif):
    import base58
    import binascii

    known_part = masked_wif.rstrip('*')
    masked_length = len(masked_wif) - len(known_part)

    try:
        fake_full = known_part + '1' * masked_length
        decoded = base58.b58decode_check(fake_full)
    except Exception:
        return None

    if decoded[0] != 0x80:
        return None

    priv_bytes = decoded[1:33]
    hex_key = binascii.hexlify(priv_bytes).decode()

    chars_to_mask = int((masked_length / len(masked_wif)) * len(hex_key))
    return hex_key[:-chars_to_mask] + '*' * chars_to_mask

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
A = 0
B = 7
Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

class KeyRecoveryEngine:
    def __init__(self, args):
        self.args = args
        self.masked_key = args.maskedkey
        self.target_address = args.address
        self.fetch_balances = args.fetchbalances
        self.mode = args.mode
        self.resume = args.resume
        self.state_file = args.statefile
        self.device = args.device
        self.cl_context = None
        self.found_flag = threading.Event()
        self.lock = threading.Lock()
        self.start_time = time.time()
        self.start_index = 0
        self.total_combinations = 0
        self.processed_count = 0
        self.match_info = {"index": None, "key": None, "address": None}

        self.validate_arguments()
        self.determine_key_type()
        self.init_opencl()
        self.load_state()

    def validate_arguments(self):
        if '*' not in self.masked_key:
            raise ValueError("Masked key must contain asterisks (*) for unknown characters")

        if not self.target_address:
            logger.warning("No target address provided. Will check all valid addresses")

        if self.device == 'gpu' and not self.cl_context:
            logger.warning("GPU not available. Falling back to CPU")
            self.device = 'cpu'

    def determine_key_type(self):
        """Determine key type and convert WIF to hex if needed"""
        self.missing_length = self.masked_key.count('*')
        key_length = len(self.masked_key)

        if key_length in (51, 52):  # Possibly WIF
            converted = convert_masked_wif_to_hex(self.masked_key)
            if converted:
                logger.info("\U0001f501 Converting masked WIF key to masked hex key for GPU support.")
                self.masked_key = converted
                self.secret_type = 'classic'
                self.allowed_chars = string.hexdigits.upper()[:-6]  # 0-9A-F
                self.missing_length = self.masked_key.count('*')
            else:
                raise ValueError("Invalid or too-short WIF key to convert to hex.")
        elif key_length == 64:
            self.secret_type = 'classic'
            self.allowed_chars = string.hexdigits.upper()[:-6]  # 0-9A-F
        elif key_length == 111:
            self.secret_type = 'extended'
            self.allowed_chars = string.ascii_letters + string.digits
        else:
            raise ValueError(f"Unsupported key length: {key_length}")
