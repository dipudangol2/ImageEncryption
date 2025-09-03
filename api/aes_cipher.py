# aes_cipher.py  (drop-in replacement)
import numpy as np


class AESCipher:
    """
    Faster pure-Python AES-128 (ECB) for educational use.
    API-compatible with your previous AESCipher: encrypt(bytes)->bytes, decrypt(bytes)->bytes
    """

    # S-Box and inverse S-Box as tuples (fast indexing)
    SBOX = (
        0x63,
        0x7C,
        0x77,
        0x7B,
        0xF2,
        0x6B,
        0x6F,
        0xC5,
        0x30,
        0x01,
        0x67,
        0x2B,
        0xFE,
        0xD7,
        0xAB,
        0x76,
        0xCA,
        0x82,
        0xC9,
        0x7D,
        0xFA,
        0x59,
        0x47,
        0xF0,
        0xAD,
        0xD4,
        0xA2,
        0xAF,
        0x9C,
        0xA4,
        0x72,
        0xC0,
        0xB7,
        0xFD,
        0x93,
        0x26,
        0x36,
        0x3F,
        0xF7,
        0xCC,
        0x34,
        0xA5,
        0xE5,
        0xF1,
        0x71,
        0xD8,
        0x31,
        0x15,
        0x04,
        0xC7,
        0x23,
        0xC3,
        0x18,
        0x96,
        0x05,
        0x9A,
        0x07,
        0x12,
        0x80,
        0xE2,
        0xEB,
        0x27,
        0xB2,
        0x75,
        0x09,
        0x83,
        0x2C,
        0x1A,
        0x1B,
        0x6E,
        0x5A,
        0xA0,
        0x52,
        0x3B,
        0xD6,
        0xB3,
        0x29,
        0xE3,
        0x2F,
        0x84,
        0x53,
        0xD1,
        0x00,
        0xED,
        0x20,
        0xFC,
        0xB1,
        0x5B,
        0x6A,
        0xCB,
        0xBE,
        0x39,
        0x4A,
        0x4C,
        0x58,
        0xCF,
        0xD0,
        0xEF,
        0xAA,
        0xFB,
        0x43,
        0x4D,
        0x33,
        0x85,
        0x45,
        0xF9,
        0x02,
        0x7F,
        0x50,
        0x3C,
        0x9F,
        0xA8,
        0x51,
        0xA3,
        0x40,
        0x8F,
        0x92,
        0x9D,
        0x38,
        0xF5,
        0xBC,
        0xB6,
        0xDA,
        0x21,
        0x10,
        0xFF,
        0xF3,
        0xD2,
        0xCD,
        0x0C,
        0x13,
        0xEC,
        0x5F,
        0x97,
        0x44,
        0x17,
        0xC4,
        0xA7,
        0x7E,
        0x3D,
        0x64,
        0x5D,
        0x19,
        0x73,
        0x60,
        0x81,
        0x4F,
        0xDC,
        0x22,
        0x2A,
        0x90,
        0x88,
        0x46,
        0xEE,
        0xB8,
        0x14,
        0xDE,
        0x5E,
        0x0B,
        0xDB,
        0xE0,
        0x32,
        0x3A,
        0x0A,
        0x49,
        0x06,
        0x24,
        0x5C,
        0xC2,
        0xD3,
        0xAC,
        0x62,
        0x91,
        0x95,
        0xE4,
        0x79,
        0xE7,
        0xC8,
        0x37,
        0x6D,
        0x8D,
        0xD5,
        0x4E,
        0xA9,
        0x6C,
        0x56,
        0xF4,
        0xEA,
        0x65,
        0x7A,
        0xAE,
        0x08,
        0xBA,
        0x78,
        0x25,
        0x2E,
        0x1C,
        0xA6,
        0xB4,
        0xC6,
        0xE8,
        0xDD,
        0x74,
        0x1F,
        0x4B,
        0xBD,
        0x8B,
        0x8A,
        0x70,
        0x3E,
        0xB5,
        0x66,
        0x48,
        0x03,
        0xF6,
        0x0E,
        0x61,
        0x35,
        0x57,
        0xB9,
        0x86,
        0xC1,
        0x1D,
        0x9E,
        0xE1,
        0xF8,
        0x98,
        0x11,
        0x69,
        0xD9,
        0x8E,
        0x94,
        0x9B,
        0x1E,
        0x87,
        0xE9,
        0xCE,
        0x55,
        0x28,
        0xDF,
        0x8C,
        0xA1,
        0x89,
        0x0D,
        0xBF,
        0xE6,
        0x42,
        0x68,
        0x41,
        0x99,
        0x2D,
        0x0F,
        0xB0,
        0x54,
        0xBB,
        0x16,
    )

    INV_SBOX = (
        0x52,
        0x09,
        0x6A,
        0xD5,
        0x30,
        0x36,
        0xA5,
        0x38,
        0xBF,
        0x40,
        0xA3,
        0x9E,
        0x81,
        0xF3,
        0xD7,
        0xFB,
        0x7C,
        0xE3,
        0x39,
        0x82,
        0x9B,
        0x2F,
        0xFF,
        0x87,
        0x34,
        0x8E,
        0x43,
        0x44,
        0xC4,
        0xDE,
        0xE9,
        0xCB,
        0x54,
        0x7B,
        0x94,
        0x32,
        0xA6,
        0xC2,
        0x23,
        0x3D,
        0xEE,
        0x4C,
        0x95,
        0x0B,
        0x42,
        0xFA,
        0xC3,
        0x4E,
        0x08,
        0x2E,
        0xA1,
        0x66,
        0x28,
        0xD9,
        0x24,
        0xB2,
        0x76,
        0x5B,
        0xA2,
        0x49,
        0x6D,
        0x8B,
        0xD1,
        0x25,
        0x72,
        0xF8,
        0xF6,
        0x64,
        0x86,
        0x68,
        0x98,
        0x16,
        0xD4,
        0xA4,
        0x5C,
        0xCC,
        0x5D,
        0x65,
        0xB6,
        0x92,
        0x6C,
        0x70,
        0x48,
        0x50,
        0xFD,
        0xED,
        0xB9,
        0xDA,
        0x5E,
        0x15,
        0x46,
        0x57,
        0xA7,
        0x8D,
        0x9D,
        0x84,
        0x90,
        0xD8,
        0xAB,
        0x00,
        0x8C,
        0xBC,
        0xD3,
        0x0A,
        0xF7,
        0xE4,
        0x58,
        0x05,
        0xB8,
        0xB3,
        0x45,
        0x06,
        0xD0,
        0x2C,
        0x1E,
        0x8F,
        0xCA,
        0x3F,
        0x0F,
        0x02,
        0xC1,
        0xAF,
        0xBD,
        0x03,
        0x01,
        0x13,
        0x8A,
        0x6B,
        0x3A,
        0x91,
        0x11,
        0x41,
        0x4F,
        0x67,
        0xDC,
        0xEA,
        0x97,
        0xF2,
        0xCF,
        0xCE,
        0xF0,
        0xB4,
        0xE6,
        0x73,
        0x96,
        0xAC,
        0x74,
        0x22,
        0xE7,
        0xAD,
        0x35,
        0x85,
        0xE2,
        0xF9,
        0x37,
        0xE8,
        0x1C,
        0x75,
        0xDF,
        0x6E,
        0x47,
        0xF1,
        0x1A,
        0x71,
        0x1D,
        0x29,
        0xC5,
        0x89,
        0x6F,
        0xB7,
        0x62,
        0x0E,
        0xAA,
        0x18,
        0xBE,
        0x1B,
        0xFC,
        0x56,
        0x3E,
        0x4B,
        0xC6,
        0xD2,
        0x79,
        0x20,
        0x9A,
        0xDB,
        0xC0,
        0xFE,
        0x78,
        0xCD,
        0x5A,
        0xF4,
        0x1F,
        0xDD,
        0xA8,
        0x33,
        0x88,
        0x07,
        0xC7,
        0x31,
        0xB1,
        0x12,
        0x10,
        0x59,
        0x27,
        0x80,
        0xEC,
        0x5F,
        0x60,
        0x51,
        0x7F,
        0xA9,
        0x19,
        0xB5,
        0x4A,
        0x0D,
        0x2D,
        0xE5,
        0x7A,
        0x9F,
        0x93,
        0xC9,
        0x9C,
        0xEF,
        0xA0,
        0xE0,
        0x3B,
        0x4D,
        0xAE,
        0x2A,
        0xF5,
        0xB0,
        0xC8,
        0xEB,
        0xBB,
        0x3C,
        0x83,
        0x53,
        0x99,
        0x61,
        0x17,
        0x2B,
        0x04,
        0x7E,
        0xBA,
        0x77,
        0xD6,
        0x26,
        0xE1,
        0x69,
        0x14,
        0x63,
        0x55,
        0x21,
        0x0C,
        0x7D,
    )

    # Precompute GF(2^8) mul tables for speed
    _GF2 = [0] * 256
    _GF3 = [0] * 256
    _GF9 = [0] * 256
    _GF11 = [0] * 256
    _GF13 = [0] * 256
    _GF14 = [0] * 256

    @staticmethod
    def _xtime(a):  # multiply by 2 in GF(2^8)
        a <<= 1
        if a & 0x100:
            a ^= 0x11B
        return a & 0xFF

    @classmethod
    def _init_tables(cls):
        # fills GF mul tables once
        if cls._GF2[1] != 0:  # already inited (1*2 == 2)
            return
        for x in range(256):
            g2 = cls._xtime(x)
            g4 = cls._xtime(g2)
            g8 = cls._xtime(g4)
            cls._GF2[x] = g2
            cls._GF3[x] = g2 ^ x
            cls._GF9[x] = g8 ^ x
            cls._GF11[x] = g8 ^ g2 ^ x
            cls._GF13[x] = g8 ^ g4 ^ x
            cls._GF14[x] = g8 ^ g4 ^ g2

    def __init__(self, key: bytes):
        if len(key) != 16:
            raise ValueError("Key must be exactly 16 bytes for AES-128")
        self._init_tables()
        self.round_keys = self._key_expansion(key)

    # ---------- Padding ----------
    @staticmethod
    def _pad(data: bytes) -> bytes:
        padlen = 16 - (len(data) % 16)
        return data + bytes([padlen]) * padlen

    @staticmethod
    def _unpad(data: bytes) -> bytes:
        padlen = data[-1]
        if padlen < 1 or padlen > 16:
            raise ValueError("Invalid padding")
        return data[:-padlen]

    # ---------- State helpers (column-major AES layout) ----------
    @staticmethod
    def _bytes_to_state(block: bytes):
        # 16-byte -> list of 16 ints, column-major (s[0..3]=col0, [4..7]=col1, etc.)
        return [block[i] for i in range(16)]

    @staticmethod
    def _state_to_bytes(s):
        return bytes(s)

    # ---------- Core transforms (in-place on 16-int list) ----------
    def _add_round_key(self, s, rk):
        for i in range(16):
            s[i] ^= rk[i]

    def _sub_bytes(self, s):
        S = self.SBOX
        for i in range(16):
            s[i] = S[s[i]]

    def _inv_sub_bytes(self, s):
        S = self.INV_SBOX
        for i in range(16):
            s[i] = S[s[i]]

    def _shift_rows(self, s):
        # rows are (r, c): index r + 4*c
        s[1], s[5], s[9], s[13] = s[5], s[9], s[13], s[1]  # row1 left by 1
        s[2], s[6], s[10], s[14] = s[10], s[14], s[2], s[6]  # row2 left by 2
        s[3], s[7], s[11], s[15] = s[15], s[3], s[7], s[11]  # row3 left by 3

    def _inv_shift_rows(self, s):
        s[1], s[5], s[9], s[13] = s[13], s[1], s[5], s[9]  # right by 1
        s[2], s[6], s[10], s[14] = s[10], s[14], s[2], s[6]  # right by 2
        s[3], s[7], s[11], s[15] = s[7], s[11], s[15], s[3]  # right by 3

    def _mix_columns(self, s):
        g2, g3 = self._GF2, self._GF3
        for c in (0, 4, 8, 12):
            a0, a1, a2, a3 = s[c], s[c + 1], s[c + 2], s[c + 3]
            s[c] = g2[a0] ^ g3[a1] ^ a2 ^ a3
            s[c + 1] = a0 ^ g2[a1] ^ g3[a2] ^ a3
            s[c + 2] = a0 ^ a1 ^ g2[a2] ^ g3[a3]
            s[c + 3] = g3[a0] ^ a1 ^ a2 ^ g2[a3]

    def _inv_mix_columns(self, s):
        g9, g11, g13, g14 = self._GF9, self._GF11, self._GF13, self._GF14
        for c in (0, 4, 8, 12):
            a0, a1, a2, a3 = s[c], s[c + 1], s[c + 2], s[c + 3]
            s[c] = g14[a0] ^ g11[a1] ^ g13[a2] ^ g9[a3]
            s[c + 1] = g9[a0] ^ g14[a1] ^ g11[a2] ^ g13[a3]
            s[c + 2] = g13[a0] ^ g9[a1] ^ g14[a2] ^ g11[a3]
            s[c + 3] = g11[a0] ^ g13[a1] ^ g9[a2] ^ g14[a3]

    # ---------- Key schedule ----------
    def _key_expansion(self, key_bytes):
        S = self.SBOX
        RCON = (0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36)

        # start with original key as 16 bytes
        w = [b for b in key_bytes]  # expanded key bytes
        # we need 11 round keys * 16 = 176 bytes (current 16 -> need +160)
        for i in range(10):
            # last 4 bytes of current key block
            t0, t1, t2, t3 = w[-4], w[-3], w[-2], w[-1]
            # rotword + subword + rcon
            t0, t1, t2, t3 = S[t1], S[t2], S[t3], S[t0]
            t0 ^= RCON[i]
            # next 16 bytes are generated in 4 words
            # word 0
            base = len(w) - 16
            w.append(w[base] ^ t0)
            w.append(w[base + 1] ^ t1)
            w.append(w[base + 2] ^ t2)
            w.append(w[base + 3] ^ t3)
            # words 1..3
            for j in range(1, 4):
                base2 = len(w) - 16
                w.append(w[base2 + 4 * j - 4] ^ w[base2 + 4 * j])
                w.append(w[base2 + 4 * j - 3] ^ w[base2 + 4 * j + 1])
                w.append(w[base2 + 4 * j - 2] ^ w[base2 + 4 * j + 2])
                w.append(w[base2 + 4 * j - 1] ^ w[base2 + 4 * j + 3])

        # split into 11 round keys of 16 bytes
        rks = [w[i * 16 : (i + 1) * 16] for i in range(11)]
        return rks

    # ---------- Public API ----------
    def encrypt(self, data):
        # accept numpy arrays too
        if isinstance(data, np.ndarray):
            data = data.tobytes()

        data = self._pad(data)
        out = bytearray(len(data))

        rk = self.round_keys
        sub_bytes = self._sub_bytes
        shift_rows = self._shift_rows
        mix_columns = self._mix_columns
        add_round_key = self._add_round_key
        bytes_to_state = self._bytes_to_state
        state_to_bytes = self._state_to_bytes

        mv_in = memoryview(data)
        mv_out = memoryview(out)

        for off in range(0, len(data), 16):
            block = mv_in[off : off + 16]
            s = bytes_to_state(block)

            add_round_key(s, rk[0])
            for rnd in range(1, 10):
                sub_bytes(s)
                shift_rows(s)
                mix_columns(s)
                add_round_key(s, rk[rnd])
            sub_bytes(s)
            shift_rows(s)
            add_round_key(s, rk[10])

            mv_out[off : off + 16] = state_to_bytes(s)

        return bytes(out)

    def decrypt(self, data):
        if len(data) % 16 != 0:
            raise ValueError("Ciphertext length must be multiple of 16")

        out = bytearray(len(data))

        rk = self.round_keys
        inv_sub_bytes = self._inv_sub_bytes
        inv_shift_rows = self._inv_shift_rows
        inv_mix_columns = self._inv_mix_columns
        add_round_key = self._add_round_key
        bytes_to_state = self._bytes_to_state
        state_to_bytes = self._state_to_bytes

        mv_in = memoryview(data)
        mv_out = memoryview(out)

        for off in range(0, len(data), 16):
            block = mv_in[off : off + 16]
            s = bytes_to_state(block)

            add_round_key(s, rk[10])
            for rnd in range(9, 0, -1):
                inv_shift_rows(s)
                inv_sub_bytes(s)
                add_round_key(s, rk[rnd])
                inv_mix_columns(s)
            inv_shift_rows(s)
            inv_sub_bytes(s)
            add_round_key(s, rk[0])

            mv_out[off : off + 16] = state_to_bytes(s)

        return self._unpad(bytes(out))
