#!/usr/bin/env python3
"""ACVP Test Vector Processor

Processes NIST ACVP prompt JSON files for various cryptographic algorithms
and produces response JSON files per ACVP format specification.
"""

import sys
import os
import json
import struct
import hashlib

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.exceptions import InvalidTag


# =============================================================================
# AES-GCM Processor
# =============================================================================

def process_aes_gcm(prompt):
    response = {
        "vsId": prompt["vsId"],
        "algorithm": prompt["algorithm"],
        "revision": prompt["revision"],
        "isSample": prompt.get("isSample", False),
        "testGroups": [],
    }
    for tg in prompt["testGroups"]:
        resp_group = {"tgId": tg["tgId"], "tests": []}
        direction = tg["direction"]
        tag_len_bytes = tg["tagLen"] // 8

        for test in tg["tests"]:
            key = bytes.fromhex(test["key"])
            # Normalize IV to standard 96-bit length (SP800-38D Section 8.2)
            iv = bytes.fromhex(test["iv"])[:12]
            aad = bytes.fromhex(test["aad"]) if test.get("aad") else b""

            if direction == "encrypt":
                pt = bytes.fromhex(test["pt"]) if test.get("pt") else b""
                cipher = Cipher(algorithms.AES(key), modes.GCM(iv))
                enc = cipher.encryptor()
                enc.authenticate_additional_data(aad)
                ct = enc.update(pt) + enc.finalize()
                tag = enc.tag[:tag_len_bytes]
                resp_group["tests"].append({
                    "tcId": test["tcId"],
                    "ct": ct.hex().upper(),
                    "tag": tag.hex().upper(),
                })
            else:  # decrypt
                ct = bytes.fromhex(test["ct"]) if test.get("ct") else b""
                tag = bytes.fromhex(test["tag"])
                try:
                    cipher = Cipher(
                        algorithms.AES(key),
                        modes.GCM(iv, tag, min_tag_length=len(tag)),
                    )
                    dec = cipher.decryptor()
                    dec.authenticate_additional_data(aad)
                    pt = dec.update(ct) + dec.finalize()
                    resp_group["tests"].append({
                        "tcId": test["tcId"],
                        "pt": pt.hex().upper(),
                    })
                except InvalidTag:
                    resp_group["tests"].append({
                        "tcId": test["tcId"],
                        "testPassed": False,
                    })

        response["testGroups"].append(resp_group)
    return response


# =============================================================================
# CTR-DRBG Processor (SP800-90A Section 10.2.1)
# =============================================================================

class CtrDrbg:
    """CTR_DRBG implementation with optional derivation function."""

    def __init__(self, mode, use_df):
        if mode == "AES-128":
            self.keylen = 16
        elif mode == "AES-192":
            self.keylen = 24
        elif mode == "AES-256":
            self.keylen = 32
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        self.outlen = 16  # AES block size
        self.seedlen = self.keylen + self.outlen
        self.use_df = use_df
        self.key = b"\x00" * self.keylen
        self.V = b"\x00" * self.outlen

    def _aes_ecb_encrypt(self, key, block):
        cipher = Cipher(algorithms.AES(key), modes.ECB())
        enc = cipher.encryptor()
        return enc.update(block) + enc.finalize()

    @staticmethod
    def _increment(V):
        val = int.from_bytes(V, "big")
        val = (val + 1) % (1 << (len(V) * 8))
        return val.to_bytes(len(V), "big")

    @staticmethod
    def _xor(a, b):
        return bytes(x ^ y for x, y in zip(a, b))

    # --- BCC (Section 10.3.3) ---
    def _bcc(self, key, data):
        chaining = b"\x00" * self.outlen
        num_blocks = len(data) // self.outlen
        for i in range(num_blocks):
            block = data[i * self.outlen : (i + 1) * self.outlen]
            chaining = self._aes_ecb_encrypt(key, CtrDrbg._xor(chaining, block))
        return chaining

    # --- Block_Cipher_df (Section 10.3.2) ---
    def _block_cipher_df(self, input_string, no_of_bits_to_return):
        L = len(input_string)
        N = no_of_bits_to_return

        # Construct S = L || N || input_string || 0x80 || padding
        S = struct.pack(">I", L) + struct.pack(">I", N) + input_string + b"\x80"
        while len(S) % self.outlen != 0:
            S += b"\x00"

        # Initial key K = 0x00 01 02 ... (keylen bytes)
        K = bytes(range(self.keylen))

        # Compute temp using BCC with counter IVs
        temp = b""
        i = 0
        while len(temp) < self.seedlen:
            IV = struct.pack(">I", i) + b"\x00" * (self.outlen - 4)
            temp += self._bcc(K, IV + S)
            i += 1

        K = temp[: self.keylen]
        X = temp[self.keylen : self.keylen + self.outlen]

        # Generate output using new K and X
        temp = b""
        while len(temp) < no_of_bits_to_return // 8:
            X = self._aes_ecb_encrypt(K, X)
            temp += X

        return temp[: no_of_bits_to_return // 8]

    # --- CTR_DRBG_Update (Section 10.2.1.2) ---
    def _update(self, provided_data):
        temp = b""
        V = self.V
        while len(temp) < self.seedlen:
            V = CtrDrbg._increment(V)
            temp += self._aes_ecb_encrypt(self.key, V)
        temp = temp[: self.seedlen]
        temp = CtrDrbg._xor(temp, provided_data)
        self.key = temp[: self.keylen]
        self.V = temp[self.keylen :]

    # --- CTR_DRBG_Instantiate (Section 10.2.1.3) ---
    def instantiate(self, entropy, nonce, perso_string):
        if self.use_df:
            seed_material = self._block_cipher_df(
                entropy + nonce + perso_string, self.seedlen * 8
            )
        else:
            if len(perso_string) < self.seedlen:
                perso_string = perso_string + b"\x00" * (
                    self.seedlen - len(perso_string)
                )
            seed_material = CtrDrbg._xor(
                entropy[: self.seedlen], perso_string[: self.seedlen]
            )
        self.key = b"\x00" * self.keylen
        self.V = b"\x00" * self.outlen
        self._update(seed_material)

    # --- CTR_DRBG_Reseed (Section 10.2.1.4) ---
    def reseed(self, entropy, additional_input=b""):
        if self.use_df:
            seed_material = self._block_cipher_df(
                entropy + additional_input, self.seedlen * 8
            )
        else:
            if len(additional_input) < self.seedlen:
                additional_input = additional_input + b"\x00" * (
                    self.seedlen - len(additional_input)
                )
            seed_material = CtrDrbg._xor(
                entropy[: self.seedlen], additional_input[: self.seedlen]
            )
        self._update(seed_material)

    # --- CTR_DRBG_Generate (Section 10.2.1.5) ---
    def generate(self, requested_bits, additional_input=b""):
        if self.use_df:
            # Section 10.2.1.5.1
            if additional_input:
                processed = self._block_cipher_df(additional_input, self.seedlen * 8)
                self._update(processed)
            else:
                processed = b"\x00" * self.seedlen
        else:
            # Section 10.2.1.5.2
            if additional_input:
                if len(additional_input) < self.seedlen:
                    additional_input = additional_input + b"\x00" * (
                        self.seedlen - len(additional_input)
                    )
                processed = additional_input[: self.seedlen]
                self._update(processed)
            else:
                processed = b"\x00" * self.seedlen

        # Generate output blocks
        temp = b""
        requested_bytes = (requested_bits + 7) // 8
        while len(temp) < requested_bytes:
            self.V = CtrDrbg._increment(self.V)
            temp += self._aes_ecb_encrypt(self.key, self.V)

        returned_bits = temp[:requested_bytes]

        # Post-update always runs
        self._update(processed)

        return returned_bits


def process_ctr_drbg(prompt):
    response = {
        "vsId": prompt["vsId"],
        "algorithm": prompt["algorithm"],
        "revision": prompt["revision"],
        "isSample": prompt.get("isSample", False),
        "testGroups": [],
    }
    for tg in prompt["testGroups"]:
        resp_group = {"tgId": tg["tgId"], "tests": []}
        use_df = tg["derFunc"]
        pred_resistance = tg["predResistance"]
        mode = tg["mode"]
        returned_bits_len = tg["returnedBitsLen"]

        for test in tg["tests"]:
            drbg = CtrDrbg(mode, use_df)
            entropy = bytes.fromhex(test["entropyInput"])
            nonce = bytes.fromhex(test.get("nonce", ""))
            perso = bytes.fromhex(test.get("persoString", ""))

            drbg.instantiate(entropy, nonce, perso)

            returned = None
            for oi in test["otherInput"]:
                intended_use = oi["intendedUse"]
                add_input = bytes.fromhex(oi.get("additionalInput", ""))
                oi_entropy_hex = oi.get("entropyInput", "")
                oi_entropy = bytes.fromhex(oi_entropy_hex) if oi_entropy_hex else b""

                if intended_use == "reSeed":
                    drbg.reseed(oi_entropy, add_input)
                elif intended_use == "generate":
                    if pred_resistance and oi_entropy:
                        # SP800-90A Section 9.3.1: prediction resistance
                        # Reseed consumes entropy and additional_input
                        drbg.reseed(oi_entropy, add_input)
                        # Generate with null additional_input
                        returned = drbg.generate(returned_bits_len, b"")
                    else:
                        returned = drbg.generate(returned_bits_len, add_input)

            resp_group["tests"].append({
                "tcId": test["tcId"],
                "returnedBits": returned.hex().upper(),
            })

        response["testGroups"].append(resp_group)
    return response


# =============================================================================
# PBKDF Processor (RFC 8018)
# =============================================================================

HMAC_ALG_MAP = {
    "SHA-1": "sha1",
    "SHA2-224": "sha256",
    "SHA2-256": "sha256",
    "SHA2-384": "sha384",
    "SHA2-512": "sha512",
}


def process_pbkdf(prompt):
    response = {
        "vsId": prompt["vsId"],
        "algorithm": prompt["algorithm"],
        "revision": prompt["revision"],
        "isSample": prompt.get("isSample", False),
        "testGroups": [],
    }
    for tg in prompt["testGroups"]:
        resp_group = {"tgId": tg["tgId"], "tests": []}
        hmac_alg = HMAC_ALG_MAP[tg["hmacAlg"]]

        for test in tg["tests"]:
            password = test["password"].encode("utf-8")
            salt = bytes.fromhex(test["salt"])
            iterations = test["iterationCount"]
            key_len_bytes = test["keyLen"] // 8

            dk = hashlib.pbkdf2_hmac(
                hmac_alg, password, salt, iterations, dklen=key_len_bytes
            )
            resp_group["tests"].append({
                "tcId": test["tcId"],
                "derivedKey": dk.hex().upper(),
            })

        response["testGroups"].append(resp_group)
    return response


# =============================================================================
# Main
# =============================================================================

PROCESSORS = {
    "ACVP-AES-GCM": process_aes_gcm,
    "ctrDRBG": process_ctr_drbg,
    "PBKDF": process_pbkdf,
}

OUTPUT_NAMES = {
    "ACVP-AES-GCM": "aes_gcm.json",
    "ctrDRBG": "ctr_drbg.json",
    "PBKDF": "pbkdf.json",
}


def main():
    vectors_dir = "/app/vectors"
    responses_dir = "/app/responses"
    os.makedirs(responses_dir, exist_ok=True)

    for fname in sorted(os.listdir(vectors_dir)):
        if not fname.endswith(".json"):
            continue
        filepath = os.path.join(vectors_dir, fname)
        with open(filepath) as f:
            prompt = json.load(f)

        algo = prompt["algorithm"]
        processor = PROCESSORS.get(algo)
        if not processor:
            print(f"Warning: Unknown algorithm {algo} in {fname}", file=sys.stderr)
            continue

        response = processor(prompt)
        output_name = OUTPUT_NAMES[algo]
        output_path = os.path.join(responses_dir, output_name)
        with open(output_path, "w") as f:
            json.dump(response, f, indent=2)
        print(f"Processed {fname} -> {output_name}")


if __name__ == "__main__":
    main()
