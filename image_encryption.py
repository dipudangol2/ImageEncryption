import numpy as np

class AESCipher:
    """
    AES-128 encryption/decryption implementation from scratch.
    Uses a 128-bit (16-byte) key for encryption and decryption.
    """
    
    def __init__(self, key):
        """
        Initialize AES cipher with a 16-byte key.
        
        Args:
            key: 16-byte key for AES-128 encryption
        """
        if len(key) != 16:
            raise ValueError("Key must be exactly 16 bytes for AES-128")
        
        self.key = key
        self.round_keys = self._key_expansion(key)
    
    def _pad(self, data):
        """
        Implement PKCS7 padding to ensure data length is multiple of 16 bytes.
        
        Args:
            data: Input data (bytes)
        
        Returns:
            bytes: Padded data
        """
        padding_length = 16 - (len(data) % 16)
        padding = bytes([padding_length] * padding_length)
        return data + padding
    
    def _unpad(self, data):
        """
        Remove PKCS7 padding after decryption.
        
        Args:
            data: Padded data (bytes)
        
        Returns:
            bytes: Unpadded data
        """
        padding_length = data[-1]
        return data[:-padding_length]
    
    def _sub_bytes(self, state):
        """
        Apply SubBytes transformation using S-Box substitution.
        
        Args:
            state: 4x4 state matrix
        
        Returns:
            4x4 state matrix after SubBytes transformation
        """
        # AES S-Box
        s_box = [
            0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
            0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
            0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
            0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
            0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
            0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
            0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
            0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
            0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
            0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
            0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
            0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
            0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
            0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
            0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
            0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
        ]
        
        new_state = [[0 for _ in range(4)] for _ in range(4)]
        for i in range(4):
            for j in range(4):
                new_state[i][j] = s_box[state[i][j]]
        
        return new_state
    
    def _inv_sub_bytes(self, state):
        """
        Apply inverse SubBytes transformation using inverse S-Box.
        
        Args:
            state: 4x4 state matrix
        
        Returns:
            4x4 state matrix after inverse SubBytes transformation
        """
        # AES inverse S-Box
        inv_s_box = [
            0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
            0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
            0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
            0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
            0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
            0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
            0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
            0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
            0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
            0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
            0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
            0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
            0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
            0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
            0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
            0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
        ]
        
        new_state = [[0 for _ in range(4)] for _ in range(4)]
        for i in range(4):
            for j in range(4):
                new_state[i][j] = inv_s_box[state[i][j]]
        
        return new_state
    
    def _shift_rows(self, state):
        """
        Apply ShiftRows transformation - rotate each row left by its row number.
        
        Args:
            state: 4x4 state matrix
        
        Returns:
            4x4 state matrix after ShiftRows transformation
        """
        new_state = [[0 for _ in range(4)] for _ in range(4)]
        
        # Row 0: no shift
        new_state[0] = state[0][:]
        
        # Row 1: shift left by 1
        new_state[1] = state[1][1:] + state[1][:1]
        
        # Row 2: shift left by 2
        new_state[2] = state[2][2:] + state[2][:2]
        
        # Row 3: shift left by 3
        new_state[3] = state[3][3:] + state[3][:3]
        
        return new_state
    
    def _inv_shift_rows(self, state):
        """
        Apply inverse ShiftRows transformation - rotate each row right by its row number.
        
        Args:
            state: 4x4 state matrix
        
        Returns:
            4x4 state matrix after inverse ShiftRows transformation
        """
        new_state = [[0 for _ in range(4)] for _ in range(4)]
        
        # Row 0: no shift
        new_state[0] = state[0][:]
        
        # Row 1: shift right by 1
        new_state[1] = state[1][-1:] + state[1][:-1]
        
        # Row 2: shift right by 2
        new_state[2] = state[2][-2:] + state[2][:-2]
        
        # Row 3: shift right by 3
        new_state[3] = state[3][-3:] + state[3][:-3]
        
        return new_state
    
    def _mix_columns(self, state):
        """
        Apply MixColumns transformation using Galois field multiplication.
        
        Args:
            state: 4x4 state matrix
        
        Returns:
            4x4 state matrix after MixColumns transformation
        """
        new_state = [[0 for _ in range(4)] for _ in range(4)]
        
        for c in range(4):
            # MixColumns matrix multiplication in GF(2^8)
            new_state[0][c] = self._gf_mult(2, state[0][c]) ^ self._gf_mult(3, state[1][c]) ^ state[2][c] ^ state[3][c]
            new_state[1][c] = state[0][c] ^ self._gf_mult(2, state[1][c]) ^ self._gf_mult(3, state[2][c]) ^ state[3][c]
            new_state[2][c] = state[0][c] ^ state[1][c] ^ self._gf_mult(2, state[2][c]) ^ self._gf_mult(3, state[3][c])
            new_state[3][c] = self._gf_mult(3, state[0][c]) ^ state[1][c] ^ state[2][c] ^ self._gf_mult(2, state[3][c])
        
        return new_state
    
    def _inv_mix_columns(self, state):
        """
        Apply inverse MixColumns transformation.
        
        Args:
            state: 4x4 state matrix
        
        Returns:
            4x4 state matrix after inverse MixColumns transformation
        """
        new_state = [[0 for _ in range(4)] for _ in range(4)]
        
        for c in range(4):
            # Inverse MixColumns matrix multiplication in GF(2^8)
            new_state[0][c] = self._gf_mult(14, state[0][c]) ^ self._gf_mult(11, state[1][c]) ^ self._gf_mult(13, state[2][c]) ^ self._gf_mult(9, state[3][c])
            new_state[1][c] = self._gf_mult(9, state[0][c]) ^ self._gf_mult(14, state[1][c]) ^ self._gf_mult(11, state[2][c]) ^ self._gf_mult(13, state[3][c])
            new_state[2][c] = self._gf_mult(13, state[0][c]) ^ self._gf_mult(9, state[1][c]) ^ self._gf_mult(14, state[2][c]) ^ self._gf_mult(11, state[3][c])
            new_state[3][c] = self._gf_mult(11, state[0][c]) ^ self._gf_mult(13, state[1][c]) ^ self._gf_mult(9, state[2][c]) ^ self._gf_mult(14, state[3][c])
        
        return new_state
    
    def _gf_mult(self, a, b):
        """
        Perform multiplication in Galois Field GF(2^8).
        
        Args:
            a, b: Integers to multiply in GF(2^8)
        
        Returns:
            Result of multiplication in GF(2^8)
        """
        result = 0
        for _ in range(8):
            if b & 1:
                result ^= a
            hi_bit_set = a & 0x80
            a <<= 1
            if hi_bit_set:
                a ^= 0x1b  # AES irreducible polynomial
            b >>= 1
        return result & 0xFF
    
    def _add_round_key(self, state, round_key):
        """
        Apply AddRoundKey transformation by XORing state with round key.
        
        Args:
            state: 4x4 state matrix
            round_key: 16-byte round key
        
        Returns:
            4x4 state matrix after AddRoundKey transformation
        """
        new_state = [[0 for _ in range(4)] for _ in range(4)]
        
        for i in range(4):
            for j in range(4):
                new_state[i][j] = state[i][j] ^ round_key[i * 4 + j]
        
        return new_state
    
    def _key_expansion(self, key):
        """
        Generate round keys from the original key using AES key schedule.
        
        Args:
            key: 16-byte original key
        
        Returns:
            List of 11 round keys (176 bytes total)
        """
        # Round constants for key expansion
        rcon = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36]
        
        # S-Box for key expansion
        s_box = [
            0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
            0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
            0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
            0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
            0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
            0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
            0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
            0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
            0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
            0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
            0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
            0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
            0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
            0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
            0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
            0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
        ]
        
        # Initialize expanded key with original key
        expanded_key = list(key)
        
        # Generate 10 additional round keys (40 more words)
        for i in range(1, 11):
            # Get the last 4 bytes of the previous round key
            temp = expanded_key[-4:]
            
            # Rotate temp
            temp = temp[1:] + temp[:1]
            
            # Apply S-Box substitution
            temp = [s_box[b] for b in temp]
            
            # XOR with round constant
            temp[0] ^= rcon[i-1]
            
            # XOR with the word 4 positions back
            prev_word = expanded_key[i*16-16:i*16-12]
            new_word = [temp[j] ^ prev_word[j] for j in range(4)]
            expanded_key.extend(new_word)
            
            # Generate the remaining 3 words for this round
            for j in range(3):
                prev_word = expanded_key[-8:-4]
                last_word = expanded_key[-4:]
                new_word = [last_word[k] ^ prev_word[k] for k in range(4)]
                expanded_key.extend(new_word)
        
        # Split into 11 round keys of 16 bytes each
        round_keys = []
        for i in range(11):
            round_keys.append(expanded_key[i*16:(i+1)*16])
        
        return round_keys
    
    def encrypt(self, data):
        """
        Encrypt data using AES-128.
        
        Args:
            data: Input data (bytes)
        
        Returns:
            bytes: Encrypted ciphertext
        """
        # Convert data to bytes if it's a numpy array
        if isinstance(data, np.ndarray):
            data = data.tobytes()
        
        # Pad the data
        padded_data = self._pad(data)
        
        encrypted_blocks = []
        
        # Process data in 16-byte blocks
        for i in range(0, len(padded_data), 16):
            block = padded_data[i:i+16]
            
            # Convert block to 4x4 state matrix
            state = [[block[r*4 + c] for c in range(4)] for r in range(4)]
            
            # Initial AddRoundKey
            state = self._add_round_key(state, self.round_keys[0])
            
            # 9 main rounds
            for round_num in range(1, 10):
                state = self._sub_bytes(state)
                state = self._shift_rows(state)
                state = self._mix_columns(state)
                state = self._add_round_key(state, self.round_keys[round_num])
            
            # Final round (no MixColumns)
            state = self._sub_bytes(state)
            state = self._shift_rows(state)
            state = self._add_round_key(state, self.round_keys[10])
            
            # Convert state back to bytes
            encrypted_block = bytes([state[r][c] for r in range(4) for c in range(4)])
            encrypted_blocks.append(encrypted_block)
        
        return b''.join(encrypted_blocks)
    
    def decrypt(self, data):
        """
        Decrypt data using AES-128.
        
        Args:
            data: Encrypted ciphertext (bytes)
        
        Returns:
            bytes: Decrypted plaintext
        """
        decrypted_blocks = []
        
        # Process data in 16-byte blocks
        for i in range(0, len(data), 16):
            block = data[i:i+16]
            
            # Convert block to 4x4 state matrix
            state = [[block[r*4 + c] for c in range(4)] for r in range(4)]
            
            # Initial AddRoundKey
            state = self._add_round_key(state, self.round_keys[10])
            
            # 9 main rounds (in reverse)
            for round_num in range(9, 0, -1):
                state = self._inv_shift_rows(state)
                state = self._inv_sub_bytes(state)
                state = self._add_round_key(state, self.round_keys[round_num])
                state = self._inv_mix_columns(state)
            
            # Final round (no InvMixColumns)
            state = self._inv_shift_rows(state)
            state = self._inv_sub_bytes(state)
            state = self._add_round_key(state, self.round_keys[0])
            
            # Convert state back to bytes
            decrypted_block = bytes([state[r][c] for r in range(4) for c in range(4)])
            decrypted_blocks.append(decrypted_block)
        
        decrypted_data = b''.join(decrypted_blocks)
        
        # Remove padding
        return self._unpad(decrypted_data)