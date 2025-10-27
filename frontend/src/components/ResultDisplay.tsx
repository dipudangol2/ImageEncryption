import React from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Download, CheckCircle, XCircle, Shield, Image as ImageIcon, BarChart3, Activity } from 'lucide-react';
import { HistogramPanel } from './HistogramPanel';

interface ResultDisplayProps {
  result: {
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
      visualization_time?: number;
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
      // New comprehensive metrics
      image_width?: number;
      image_height?: number;
      image_dimensions?: string;
      image_channels?: number;
      total_pixels?: number;
      image_format?: string;
      original_size_bytes?: number;
      compressed_size_bytes?: number;
      array_size_bytes?: number;
      array_compression_ratio?: number;
      array_space_saved_percent?: number;
      detailed_metrics?: {
        histogram_analysis?: {
          original?: {
            type?: string;
            channels?: any;
            overall?: any;
            entropy?: number;
            mean_intensity?: number;
            std_intensity?: number;
          };
          encrypted?: {
            type?: string;
            channels?: any;
            overall?: any;
            entropy?: number;
            mean_intensity?: number;
            std_intensity?: number;
          };
        };
        timing_analysis?: any;
        encryption_metrics?: any;
        quality_metrics?: any;
        compression_metrics?: any;
        summary?: any;
      };
    };
    message: string;
    operation: 'encrypt' | 'decrypt';
  } | null;
  onDownload?: () => void;
  onReset: () => void;
}

export const ResultDisplay: React.FC<ResultDisplayProps> = ({
  result,
  onDownload,
  onReset,
}) => {
  if (!result) return null;

  const handleDownload = (fileUrl: string, filename: string) => {
    // Create download link
    const file = fileUrl.split('/').pop();
    const link = document.createElement('a');
    link.href = `/api/download/${file}`;
    link.setAttribute('download', filename);
    link.setAttribute('target', '_blank');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    // if (onDownload) onDownload();
  };

  return (
    <Card className="p-6 bg-gradient-secondary border-border/50">
      <div className="space-y-6">
        <div className="flex items-center space-x-3">
          <div className={`p-2 rounded-lg ${result.success ? 'bg-success/10' : 'bg-destructive/10'}`}>
            <Shield className={`h-5 w-5 ${result.success ? 'text-success' : 'text-destructive'}`} />
          </div>
          <h3 className="text-lg font-semibold text-foreground">
            {result.operation === 'encrypt' ? 'Encryption' : 'Decryption'} Result
          </h3>
        </div>

        <Alert className={`${result.success ? 'border-success/50 bg-success/10' : 'border-destructive/50 bg-destructive/10'}`}>
          {result.success ? (
            <CheckCircle className="h-4 w-4 text-success" />
          ) : (
            <XCircle className="h-4 w-4 text-destructive" />
          )}
          <AlertDescription className={result.success ? 'text-success-foreground' : 'text-destructive-foreground'}>
            {result.message}
          </AlertDescription>
        </Alert>

        {result.success && result.files && (
          <div className="space-y-4">
            {/* File Downloads */}
            <div className="space-y-3">
              {/* Encrypted Binary File */}
              {result.files.encrypted_bin && (
                <div className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
                  <div className="flex items-center space-x-3">
                    <Shield className="h-5 w-5 text-primary" />
                    <div>
                      <p className="text-sm font-medium text-foreground">Encrypted Binary</p>
                      <p className="text-xs text-muted-foreground">For secure storage/transfer</p>
                    </div>
                  </div>
                  <Button
                    variant="secure"
                    size="sm"
                    onClick={() => handleDownload(result.files!.encrypted_bin!, 'encrypted_data.bin')}
                  >
                    <Download className="h-4 w-4" />
                    Download
                  </Button>
                </div>
              )}

              {/* Visualization Image */}
              {result.files.visualization && (
                <div className="space-y-2">
                  <div className="relative group rounded-lg overflow-hidden border border-border/50">
                    <img
                      src={`${import.meta.env.VITE_SERVER_URL}${result.files.visualization}`}
                      alt="Encrypted data visualization"
                      className="w-full h-32 object-cover"
                    />
                    <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center z-50">
                      <Button
                        variant="default"
                        onClick={() => handleDownload(result.files!.visualization!, 'encrypted_visualization.png')}
                        className="bg-background/90 text-foreground hover:bg-background"
                      >
                        <Download className="h-4 w-4" />
                        Download
                      </Button>
                    </div>
                  </div>
                  <div className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
                    <div className="flex items-center space-x-3">
                      <ImageIcon className="h-5 w-5 text-accent" />
                      <div>
                        <p>{result.files!.visualization!}</p>
                        <p className="text-sm font-medium text-foreground">Visualization</p>
                        <p className="text-xs text-muted-foreground">For histogram analysis</p>
                      </div>
                    </div>
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={() => handleDownload(result.files!.visualization!, 'encrypted_visualization.png')}
                    >
                      <Download className="h-4 w-4" />
                      Download
                    </Button>
                  </div>
                </div>
              )}

              {/* Decrypted Image */}
              {result.files.decrypted_image && (
                <div className="space-y-2">
                  <div className="relative group rounded-lg overflow-hidden border border-border/50">
                    <img
                      src={`${import.meta.env.VITE_SERVER_URL}${result.files.decrypted_image}`}
                      alt="Decrypted image"
                      className="w-full h-48 object-cover"
                    />
                    <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center">
                      <Button
                        variant="default"
                        onClick={() => handleDownload(result.files!.decrypted_image!, 'decrypted_image.png')}
                        className="bg-background/90 text-foreground hover:bg-background"
                      >
                        <Download className="h-4 w-4" />
                        Download
                      </Button>
                    </div>
                  </div>
                  <div className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
                    <div className="flex items-center space-x-3">
                      <ImageIcon className="h-5 w-5 text-success" />
                      <div>
                        <p className="text-sm font-medium text-foreground">Decrypted Image</p>
                        <p className="text-xs text-muted-foreground">Original image restored</p>
                      </div>
                    </div>
                    <Button
                      variant="secure"
                      size="sm"
                      onClick={() => handleDownload(result.files!.decrypted_image!, 'decrypted_image.png')}
                    >
                      <Download className="h-4 w-4" />
                      Download
                    </Button>
                  </div>
                </div>
              )}
            </div>

            {/* Original Image Histogram */}
            {result.stats?.detailed_metrics?.histogram_analysis?.original && (
              <div className="p-4 rounded-lg bg-muted/30 space-y-4">
                <div className="flex items-center space-x-3">
                  <div className="p-2 rounded-lg bg-accent/10">
                    <BarChart3 className="h-5 w-5 text-accent" />
                  </div>
                  <h3 className="text-lg font-semibold text-foreground">Original Image Histogram</h3>
                </div>

                <div className="w-full">
                  <HistogramPanel
                    title=""
                    data={result.stats.detailed_metrics.histogram_analysis.original}
                    overallEntropy={result.stats.original_entropy}
                  />
                </div>
              </div>
            )}

            {/* Encrypted Image Histogram */}
            {result.stats?.detailed_metrics?.histogram_analysis?.encrypted && (
              <div className="p-4 rounded-lg bg-muted/30 space-y-4">
                <div className="flex items-center space-x-3">
                  <div className="p-2 rounded-lg bg-accent/10">
                    <BarChart3 className="h-5 w-5 text-accent" />
                  </div>
                  <h3 className="text-lg font-semibold text-foreground">Encrypted Image Histogram</h3>
                </div>

                <div className="w-full">
                  <HistogramPanel
                    title=""
                    data={result.stats.detailed_metrics.histogram_analysis.encrypted}
                    overallEntropy={result.stats.encrypted_entropy}
                  />
                </div>

                <div className="p-3 text-xs text-muted-foreground bg-success/10 rounded border-l-2 border-success">
                  <strong className="text-success">Security Indicator:</strong><br />
                  Higher entropy values (closer to 8.0) indicate better encryption randomness and security.
                </div>
              </div>
            )}

            {/* Fallback for when encrypted histogram is not available */}
            {result.stats?.detailed_metrics?.histogram_analysis && !result.stats.detailed_metrics.histogram_analysis.encrypted && result.stats.detailed_metrics.histogram_analysis.original && (
              <div className="p-4 rounded-lg bg-muted/30 space-y-3">
                <h4 className="text-sm font-semibold text-foreground">Encrypted Image Histogram</h4>
                <div className="p-3 text-xs text-muted-foreground bg-muted/20 rounded">
                  Encrypted histogram analysis will be available after visualization processing completes.
                </div>
              </div>
            )}

            {/* Analytics Metrics */}
            {result.stats && (result.stats.psnr || result.stats.npcr || result.stats.uaci) && (
              <div className="p-4 rounded-lg bg-muted/30 space-y-3">
                <h4 className="text-sm font-semibold text-foreground">Analytics</h4>
                <div className="grid grid-cols-2 gap-3 text-xs">
                  {result.stats.psnr && (
                    <div>
                      <span className="text-muted-foreground">PSNR:</span>
                      <span className="ml-1 font-medium">{result.stats.psnr} dB</span>
                    </div>
                  )}
                  {result.stats.npcr && (
                    <div>
                      <span className="text-muted-foreground">NPCR:</span>
                      <span className="ml-1 font-medium">{result.stats.npcr}%</span>
                    </div>
                  )}
                  {result.stats.uaci && (
                    <div>
                      <span className="text-muted-foreground">UACI:</span>
                      <span className="ml-1 font-medium">{result.stats.uaci}%</span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Performance Timing */}
            {result.stats && (
              <div className="p-4 rounded-lg bg-muted/30 space-y-3">
                <h4 className="text-sm font-semibold text-foreground">Performance Timing</h4>
                <div className="grid grid-cols-2 gap-3 text-xs">
                  {result.stats.total_processing_time && result.stats.total_processing_time > 0 && (
                    <div>
                      <span className="text-muted-foreground">Total Time:</span>
                      <span className="ml-1 font-medium">{result.stats.total_processing_time}s</span>
                    </div>
                  )}

                  {/* Show encryption-specific timing only for encrypt operation */}
                  {result.operation === 'encrypt' && result.stats.compression_time && result.stats.compression_time > 0 && (
                    <div>
                      <span className="text-muted-foreground">Compression:</span>
                      <span className="ml-1 font-medium">{result.stats.compression_time}s</span>
                    </div>
                  )}
                  {result.operation === 'encrypt' && result.stats.encryption_time && result.stats.encryption_time > 0 && (
                    <div>
                      <span className="text-muted-foreground">Encryption:</span>
                      <span className="ml-1 font-medium">{result.stats.encryption_time}s</span>
                    </div>
                  )}
                  {result.operation === 'encrypt' && result.stats.visualization_time && result.stats.visualization_time > 0 && (
                    <div>
                      <span className="text-muted-foreground">Visualization:</span>
                      <span className="ml-1 font-medium">{result.stats.visualization_time}s</span>
                    </div>
                  )}

                  {/* Show decryption-specific timing only for decrypt operation */}
                  {result.operation === 'decrypt' && result.stats.decryption_time && result.stats.decryption_time > 0 && (
                    <div>
                      <span className="text-muted-foreground">Decryption:</span>
                      <span className="ml-1 font-medium">{result.stats.decryption_time}s</span>
                    </div>
                  )}
                  {result.operation === 'decrypt' && result.stats.decompression_time && result.stats.decompression_time > 0 && (
                    <div>
                      <span className="text-muted-foreground">Decompression:</span>
                      <span className="ml-1 font-medium">{result.stats.decompression_time}s</span>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        <div className="flex gap-3">
          <Button
            variant="secondary"
            onClick={onReset}
            className="flex-1"
          >
            Process Another File
          </Button>
        </div>
      </div>
    </Card>
  );
};