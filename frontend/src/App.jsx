import  { useState } from "react";
import axios from "axios";


function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [resultImage, setResultImage] = useState(null);
  const [encryptionKey, setEncryptionKey] = useState("");
  const [loading, setLoading] = useState(false);

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setResultImage(null);
    }
  };

  const sendToAPI = async (endpoint) => {
    if (!image || !encryptionKey) {
      alert("Please upload an image and enter a key.");
      return;
    }

    const formData = new FormData();
    formData.append("file", image);
    formData.append("key", encryptionKey);

    try {
      setLoading(true);
      const response = await axios.post(
        `http://localhost:8000/${endpoint}`,
        formData,
        {
          responseType: "blob",
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      const blob = new Blob([response.data]);
      setResultImage(URL.createObjectURL(blob));
    } catch (err) {
      console.error("API Error:", err);
      alert("Failed to process image.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-r from-sky-200 to-indigo-200 flex items-center justify-center p-6">
      <div className="bg-white p-8 rounded-2xl shadow-lg w-full max-w-2xl">
        <h1 className="text-3xl font-bold text-center mb-6 text-gray-800">
          🖼️ Image Encryption
        </h1>

        <div className="mb-4">
          <label className="block mb-2 font-medium text-gray-700">
            Upload Image
          </label>
          <input
            type="file"
            accept="image/*"
            onChange={handleImageUpload}
            className="w-full border rounded-lg p-2"
          />
        </div>

        <div className="mb-6">
          <label className="block mb-2 font-medium text-gray-700">
            Enter Encryption Key
          </label>
          <input
            type="password"
            value={encryptionKey}
            onChange={(e) => setEncryptionKey(e.target.value)}
            placeholder="Enter a secure key"
            className="w-full border rounded-lg p-2"
          />
        </div>

        <div className="flex justify-center gap-4 mb-6">
          <button
            onClick={() => sendToAPI("encrypt")}
            disabled={loading}
            className="bg-blue-500 hover:bg-blue-600 text-white px-5 py-2 rounded-lg transition"
          >
            {loading ? "Encrypting..." : "Encrypt"}
          </button>
          <button
            onClick={() => sendToAPI("decrypt")}
            disabled={loading}
            className="bg-purple-500 hover:bg-purple-600 text-white px-5 py-2 rounded-lg transition"
          >
            {loading ? "Decrypting..." : "Decrypt"}
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {preview && (
            <div>
              <h3 className="text-center font-semibold text-gray-700">Original</h3>
              <img
                src={preview}
                alt="Original"
                className="rounded-lg border w-full object-cover"
              />
            </div>
          )}
          {resultImage && (
            <div>
              <h3 className="text-center font-semibold text-gray-700">Processed</h3>
              <img
                src={resultImage}
                alt="Processed"
                className="rounded-lg border w-full object-cover"
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
