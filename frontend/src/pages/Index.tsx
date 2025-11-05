import React, { useState } from 'react';
import { ImageUpload } from '@/components/ImageUpload';
import { EncryptionForm } from '@/components/EncryptionForm';
import { ResultDisplay } from '@/components/ResultDisplay';
import { Card } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Shield, Lock, AlertTriangle } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

interface ProcessResult {
  success: boolean;
  files?: {
    encrypted_bin?: string;
    visualization?: string;
    decrypted_image?: string;
  };
  stats?: {
    original_size?: string;
    compressed_size?: string;
    compression_ratio?: number;
    space_saved_percent?: number;
    mse?: number;
    psnr?: number;
    npcr?: number;
    uaci?: number;
    encryption_strength?: string;
    bits_per_pixel?: number;
    compression_efficiency?: number;
    total_processing_time?: number;
    compression_time?: number;
    encryption_time?: number;
    decryption_time?: number;
    decompression_time?: number;
    original_entropy?: number;
    encrypted_entropy?: number;
    overall_quality?: string;
    encryption_security?: string;
    compression_efficiency_rating?: string;
    output_size?: number;
    output_size_display?: string;
    image_shape?: number[];
    original_format?: string;
    quality?: number;
  };
  message: string;
  operation: 'encrypt' | 'decrypt';
}

const Index = () => {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<ProcessResult | null>(null);
  const { toast } = useToast();
  const baseUrl = import.meta.env.VITE_SERVER_URL ?? "";

  const handleImageSelect = (file: File) => {
    setSelectedImage(file);
    setResult(null); // Clear previous results
  };

  const handleImageRemove = () => {
    setSelectedImage(null);
    setResult(null);
  };

  const handleEncryption = async (password: string, operation: 'encrypt' | 'decrypt') => {
    if (!selectedImage) {
      toast({
        title: "Error",
        description: "Please select a file first",
        variant: "destructive",
      });
      return;
    }

    // Validate file type for operation
    const isImageFile = selectedImage.type.startsWith('image/');
    const isBinFile = selectedImage.name.endsWith('.bin');

    if (operation === 'encrypt' && !isImageFile) {
      toast({
        title: "Error",
        description: "Encryption requires an image file (.jpg, .jpeg, .png)",
        variant: "destructive",
      });
      return;
    }

    if (operation === 'decrypt' && !isBinFile) {
      toast({
        title: "Error",
        description: "Decryption requires an encrypted .bin file",
        variant: "destructive",
      });
      return;
    }

    setIsLoading(true);

    try {
      // Create FormData for file upload
      const formData = new FormData();
      formData.append(operation === 'encrypt' ? 'image' : 'file', selectedImage);
      formData.append('key', password);
      if (operation === 'encrypt') {
        formData.append('quality', '75'); // Default quality
      }

      // Call appropriate endpoint
      const endpoint = operation === 'encrypt' ? '/api/encrypt' : '/api/decrypt';
      const response = await fetch(`${baseUrl}${endpoint}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Server error: ${response.status}`);
      }

      const data = await response.json();

      if (data.success) {
        setResult({
          success: true,
          files: data.files,
          stats: data.stats,
          message: `${operation === 'encrypt' ? 'Encryption' : 'Decryption'} completed successfully!`,
          operation,
        });

        toast({
          title: "Success",
          description: `${operation === 'encrypt' ? 'Image encrypted' : 'File decrypted'} successfully!`,
        });
      } else {
        throw new Error(data.message || 'Processing failed');
      }

    } catch (error) {
      console.error('Processing error:', error);

      const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred';

      setResult({
        success: false,
        message: errorMessage,
        operation,
      });

      toast({
        title: "Error",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedImage(null);
    setResult(null);
  };

  const handleDownload = () => {
    toast({
      title: "Download Started",
      description: "Your file download has begun",
    });
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <div className="border-b border-border/50 bg-gradient-secondary">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center space-x-4">
            <div className="p-3 rounded-xl bg-primary/10 border border-primary/20">
              <Shield className="h-8 w-8 text-primary" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-foreground">
                Image Encryption Tool
              </h1>
              <p className="text-muted-foreground mt-1">
                Secure your images with AES encryption
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-6xl mx-auto space-y-8">



          {/* Main Grid */}
          <div className="grid lg:grid-cols-2 gap-8">

            {/* Left Column */}
            <div className="space-y-6">
              <ImageUpload
                onImageSelect={handleImageSelect}
                selectedImage={selectedImage}
                onImageRemove={handleImageRemove}
              />

              <EncryptionForm
                onSubmit={handleEncryption}
                isLoading={isLoading}
                hasImage={!!selectedImage}
                selectedFile={selectedImage}
              />
            </div>

            {/* Right Column */}
            <div className="space-y-8">
              {result ? (
                <ResultDisplay
                  result={result}
                  onDownload={handleDownload}
                  onReset={handleReset}
                />
              ) : (
                <Card className="p-8 bg-gradient-secondary border-border/50">
                  <div className="text-center space-y-4">
                    <div className="p-4 rounded-full bg-muted/50 mx-auto w-fit">
                      <Shield className="h-12 w-12 text-muted-foreground" />
                    </div>
                    <div className="space-y-2">
                      <h3 className="text-lg font-semibold text-foreground">
                        Ready to Process
                      </h3>
                      <p className="text-sm text-muted-foreground">
                        Upload an image and enter your password to get started
                      </p>
                    </div>
                  </div>
                </Card>
              )}

            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Index;