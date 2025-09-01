# Project Diagrams

This document contains the class and sequence diagrams for the Image Encryption project, rendered using Mermaid.

## Class Diagram

This diagram shows the static structure of the system, including classes and their relationships.

```mermaid
classDiagram
    class FastAPI {
        +add_middleware()
        +mount()
        +post(path)
        +get(path)
    }

    class ImageEncryptionAPI {
        <<FastAPI App>>
        -compressor: ImageCompressor
        -file_manager: SecureFileManager
        +encrypt_image(image, key, quality)
        +decrypt_image(file, key)
    }

    class AESEncryptor {
        -round_keys: list
        +encrypt(data: bytes) bytes
        +decrypt(data: bytes) bytes
    }

    class ImageCompressor {
        +compress(image, quality, use_color) dict
        +decompress(compressed_data) ndarray
        +get_compression_stats(original, compressed) dict
    }

    class ImprovedDCTCompressor {
        -zigzag_indices: ndarray
        -luminance_quant: ndarray
        -chrominance_quant: ndarray
        +compress(image, quality) dict
        +decompress(compressed_data) ndarray
    }

    class IndexPage {
        <<React Component>>
        -selectedImage: File
        -isLoading: boolean
        -result: ProcessResult
        +handleImageSelect(file)
        +handleEncryption(key, operation)
        +handleReset()
    }

    class ImageUpload {
        <<React Component>>
        +onImageSelect(file)
        +onImageRemove()
    }

    class EncryptionForm {
        <<React Component>>
        +onSubmit(key, operation)
    }

    class ResultDisplay {
        <<React Component>>
        -result: ProcessResult
        +onDownload()
        +onReset()
    }

    FastAPI --* ImageEncryptionAPI
    ImageEncryptionAPI --> AESEncryptor : uses
    ImageEncryptionAPI --> ImageCompressor : uses
    ImageCompressor --* ImprovedDCTCompressor : uses

    IndexPage --* ImageUpload
    IndexPage --* EncryptionForm
    IndexPage --* ResultDisplay
    IndexPage --> ImageEncryptionAPI : "makes API calls to"
```

## Sequence Diagram

This diagram illustrates the dynamic interaction between objects for the encryption process.

```mermaid
sequenceDiagram
    actor User
    participant IndexPage as "Index Page (React)"
    participant FastAPI_Backend as "FastAPI Backend"
    participant ImageCompressor as "Image Compressor"
    participant AESEncryptor as "AES Encryptor"

    User->>IndexPage: Selects image file
    activate IndexPage
    IndexPage->>IndexPage: Updates state with selected file

    User->>IndexPage: Enters 16-char key and clicks "Encrypt"
    IndexPage->>+FastAPI_Backend: POST /api/encrypt (image, key)
    
    FastAPI_Backend->>+ImageCompressor: compress(image_data)
    ImageCompressor-->>-FastAPI_Backend: compressed_data
    
    FastAPI_Backend->>+AESEncryptor: encrypt(compressed_data)
    AESEncryptor-->>-FastAPI_Backend: encrypted_data
    
    FastAPI_Backend-->>-IndexPage: { success: true, files: {...}, stats: {...} }
    
    IndexPage->>IndexPage: Updates state with result
    IndexPage->>User: Displays encrypted visualization and download links
    deactivate IndexPage

```