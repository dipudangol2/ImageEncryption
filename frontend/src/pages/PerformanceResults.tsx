import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ArrowLeft, Download, BarChart3, Clock, Shield, Zap, Image as ImageIcon, Activity, CheckCircle } from 'lucide-react';
import { HistogramPanel } from '@/components/HistogramPanel';

interface HistogramData {
  type?: string;
  entropy?: number;
  mean?: number;
  std_dev?: number;
  dynamic_range?: number;
  unique_values?: number;
  histogram?: number[];
  channels?: {
    red?: {
      histogram?: number[];
      entropy?: number;
      mean?: number;
      std_dev?: number;
    };
    green?: {
      histogram?: number[];
      entropy?: number;
      mean?: number;
      std_dev?: number;
    };
    blue?: {
      histogram?: number[];
      entropy?: number;
      mean?: number;
      std_dev?: number;
    };
  };
  overall?: {
    mean_intensity?: number;
    std_intensity?: number;
  };
}

interface ComprehensiveAnalysis {
  histogram_data: {
    original: HistogramData;
    encrypted: HistogramData;
    decrypted: HistogramData;
  };
  comparison_metrics: {
    quality_comparison: {
      psnr_original_vs_decrypted: number;
      mse_original_vs_decrypted: number;
    };
    encryption_security: {
      npcr_original_vs_encrypted: number;
      uaci_original_vs_encrypted: number;
      encryption_strength: string;
    };
    timing_breakdown: {
      total_time: number;
      decryption_time: number;
      decompression_time: number;
      analysis_time: number;
      compression_time: number;
      encryption_time: number;
      visualization_time: number;
      total_encryption_time: number;
    };
    compression_analysis: any;
    file_size_comparison?: {
      original_file_size: number;
      original_file_size_display: string;
      decrypted_file_size: number;
      decrypted_file_size_display: string;
      size_difference: number;
      size_difference_display: string;
      size_difference_percent: number;
      is_larger: boolean;
      is_smaller: boolean;
      is_same: boolean;
    };
  };
  detailed_metrics: any;
  summary: {
    overall_quality: string;
    encryption_security: string;
    compression_efficiency: string;
  };
}

interface PerformanceResultsState {
  files: {
    original: string;
    encrypted_visualization: string;
    decrypted: string;
  };
  comprehensive_analysis: ComprehensiveAnalysis;
  session_ids: {
    original: string;
    decrypted: string;
  };
}

const MetricCard: React.FC<{
  title: string;
  value: number | string;
  unit?: string;
  description?: string;
  variant?: 'default' | 'success' | 'warning' | 'destructive';
  review?: string;
  reviewType?: 'excellent' | 'good' | 'fair' | 'poor';
}> = ({ title, value, unit = '', description, variant = 'default', review, reviewType }) => {
  const colors = {
    default: 'bg-muted/30 border-border',
    success: 'bg-success/10 border-success/30',
    warning: 'bg-warning/10 border-warning/30',
    destructive: 'bg-destructive/10 border-destructive/30',
  };

  const reviewColors = {
    excellent: 'bg-success/20 text-success border-success/40',
    good: 'bg-primary/20 text-primary border-primary/40',
    fair: 'bg-warning/20 text-warning border-warning/40',
    poor: 'bg-destructive/20 text-destructive border-destructive/40',
  };

  const reviewIcons = {
    excellent: '✓',
    good: '↗',
    fair: '~',
    poor: '✗',
  };

  return (
    <div className={`p-4 rounded-lg border ${colors[variant]}`}>
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-medium text-muted-foreground">{title}</h3>
          {review && reviewType && (
            <Badge className={`text-xs px-2 py-0.5 ${reviewColors[reviewType]}`}>
              {reviewIcons[reviewType]} {review}
            </Badge>
          )}
        </div>
        <div className="text-2xl font-bold text-foreground">
          {typeof value === 'number' ? value.toFixed(2) : value}
          {unit && <span className="text-sm text-muted-foreground ml-1">{unit}</span>}
        </div>
        {description && (
          <p className="text-xs text-muted-foreground">{description}</p>
        )}
      </div>
    </div>
  );
};

const ImageColumn: React.FC<{
  title: string;
  imageUrl: string;
  histogramData: HistogramData;
  icon: React.ReactNode;
  variant: 'original' | 'encrypted' | 'decrypted';
  fileSize?: string;
}> = ({ title, imageUrl, histogramData, icon, variant, fileSize }) => {
  const baseUrl = import.meta.env.VITE_SERVER_URL ?? "";

  const variantStyles = {
    original: 'border-accent/50 bg-accent/5',
    encrypted: 'border-warning/50 bg-warning/5',
    decrypted: 'border-success/50 bg-success/5',
  };

  return (
    <Card className={`${variantStyles[variant]} transition-all duration-300 hover:shadow-lg`}>
      <CardHeader className="pb-4">
        <div className="flex items-center space-x-2">
          {icon}
          <CardTitle className="text-lg">{title}</CardTitle>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Image Display */}
        <div className="relative group rounded-lg overflow-hidden border border-border/50">
          <img
            src={`${baseUrl}${imageUrl}`}
            alt={title}
            className="w-full h-48 object-cover transition-transform duration-300 group-hover:scale-105"
          />
          <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center">
            <Button
              variant="default"
              size="sm"
              onClick={() => {
                const link = document.createElement('a');
                link.href = `${baseUrl}/api/download/${imageUrl.split('/').pop()}`;
                link.setAttribute('download', `${variant}_image.png`);
                link.setAttribute('target', '_blank');
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
              }}
              className="bg-background/90 text-foreground hover:bg-background"
            >
              <Download className="h-4 w-4 mr-2" />
              Download
            </Button>
          </div>
        </div>

        {/* Statistics */}
        <div className="grid grid-cols-2 gap-3">
          {fileSize && (
            <MetricCard
              title="File Size"
              value={fileSize}
              description="Storage size"
            />
          )}
        </div>

        {/* Histogram */}
        <div className="space-y-3">
          <h4 className="text-sm font-semibold text-foreground flex items-center">
            <BarChart3 className="h-4 w-4 mr-2" />
            Histogram Analysis
          </h4>
          <div className="p-3 rounded-lg bg-muted/30">
            <HistogramPanel
              data={histogramData}
              title=""
              overallEntropy={histogramData.entropy}
            />
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

// Helper functions to evaluate metrics
const evaluatePSNR = (psnr: number): { review: string; type: 'excellent' | 'good' | 'fair' | 'poor' } => {
  if (psnr >= 40) return { review: 'Excellent', type: 'excellent' };
  if (psnr >= 30) return { review: 'Good', type: 'good' };
  if (psnr >= 20) return { review: 'Fair', type: 'fair' };
  return { review: 'Poor', type: 'poor' };
};

const evaluateNPCR = (npcr: number): { review: string; type: 'excellent' | 'good' | 'fair' | 'poor' } => {
  if (npcr >= 99.5) return { review: 'Excellent', type: 'excellent' };
  if (npcr >= 99) return { review: 'Good', type: 'good' };
  if (npcr >= 95) return { review: 'Fair', type: 'fair' };
  return { review: 'Poor', type: 'poor' };
};

const evaluateUACI = (uaci: number): { review: string; type: 'excellent' | 'good' | 'fair' | 'poor' } => {
  const ideal = 33.4;
  const diff = Math.abs(uaci - ideal);
  if (diff <= 1) return { review: 'Excellent', type: 'excellent' };
  if (diff <= 3) return { review: 'Good', type: 'good' };
  if (diff <= 5) return { review: 'Fair', type: 'fair' };
  return { review: 'Poor', type: 'poor' };
};

const evaluateCompressionRatio = (ratio: number): { review: string; type: 'excellent' | 'good' | 'fair' | 'poor' } => {
  if (ratio >= 8) return { review: 'Excellent', type: 'excellent' };
  if (ratio >= 5) return { review: 'Good', type: 'good' };
  if (ratio >= 3) return { review: 'Fair', type: 'fair' };
  return { review: 'Poor', type: 'poor' };
};

const evaluateProcessingTime = (time: number, type: 'compression' | 'encryption'): { review: string; type: 'excellent' | 'good' | 'fair' | 'poor' } => {
  const thresholds = type === 'compression' ? [5, 10, 20] : [15, 30, 60];
  if (time <= thresholds[0]) return { review: 'Fast', type: 'excellent' };
  if (time <= thresholds[1]) return { review: 'Good', type: 'good' };
  if (time <= thresholds[2]) return { review: 'Slow', type: 'fair' };
  return { review: 'Very Slow', type: 'poor' };
};

export default function PerformanceResults() {
  const navigate = useNavigate();
  const location = useLocation();
  const [data, setData] = useState<PerformanceResultsState | null>(null);
  const baseUrl = import.meta.env.VITE_SERVER_URL ?? "";

  useEffect(() => {
    // Check if we have data in location state
    if (!location.state || !location.state.files || !location.state.comprehensive_analysis) {
      // Redirect back to home if no data
      navigate('/');
      return;
    }

    // Use data directly from location state
    setData({
      files: location.state.files,
      comprehensive_analysis: location.state.comprehensive_analysis,
      session_ids: location.state.session_ids
    });
  }, [location.state, navigate]);

  if (!data) {
    return null;
  }

  const { files, comprehensive_analysis } = data;

  return (
    <div className="min-h-screen bg-gradient-to-br from-background to-muted/20 p-4">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <Card className="p-6 bg-gradient-secondary !text-white border-accent/30">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Button
                variant="ghost"
                onClick={() => navigate('/')}
                className="text-white hover:bg-primary-foreground/10"
              >
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back
              </Button>
              <div>
                <h1 className="text-2xl font-bold text-white">
                  Performance Analysis Results
                </h1>
                <p className="text-white/70 text-sm">
                  Comprehensive comparison of original, encrypted, and decrypted images
                </p>
              </div>
            </div>
          </div>
        </Card>

        {/* Summary Metrics */}
        <Card className="p-6">
          <CardHeader className="pb-4">
            <CardTitle className="flex items-center">
              <Activity className="h-5 w-5 mr-2" />
              Performance Summary
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <MetricCard
                title="PSNR Quality"
                value={comprehensive_analysis.comparison_metrics.quality_comparison.psnr_original_vs_decrypted}
                unit="dB"
                variant={
                  comprehensive_analysis.comparison_metrics.quality_comparison.psnr_original_vs_decrypted > 30
                    ? 'success'
                    : comprehensive_analysis.comparison_metrics.quality_comparison.psnr_original_vs_decrypted > 20
                      ? 'warning'
                      : 'destructive'
                }
              />
              <MetricCard
                title="NPCR Security"
                value={comprehensive_analysis.comparison_metrics.encryption_security.npcr_original_vs_encrypted}
                unit="%"
                variant={
                  comprehensive_analysis.comparison_metrics.encryption_security.npcr_original_vs_encrypted > 99
                    ? 'success'
                    : 'warning'
                }
              />
              <MetricCard
                title="UACI Security"
                value={comprehensive_analysis.comparison_metrics.encryption_security.uaci_original_vs_encrypted}
                unit="%"
                variant={
                  comprehensive_analysis.comparison_metrics.encryption_security.uaci_original_vs_encrypted > 33
                    ? 'success'
                    : 'warning'
                }
              />
              <MetricCard
                title="Total Time"
                value={comprehensive_analysis.comparison_metrics.timing_breakdown.total_time}
                unit="s"
              />
            </div>
          </CardContent>
        </Card>

        {/* Three-Column Comparison */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <ImageColumn
            title="Original Image"
            imageUrl={files.original}
            histogramData={comprehensive_analysis.histogram_data.original}
            icon={<ImageIcon className="h-5 w-5 text-accent" />}
            variant="original"
            fileSize={comprehensive_analysis.comparison_metrics.file_size_comparison?.original_file_size_display}
          />
          <ImageColumn
            title="Encrypted Visualization"
            imageUrl={files.encrypted_visualization}
            histogramData={comprehensive_analysis.histogram_data.encrypted}
            icon={<Shield className="h-5 w-5 text-warning" />}
            variant="encrypted"
          />
          <ImageColumn
            title="Decrypted Image"
            imageUrl={files.decrypted}
            histogramData={comprehensive_analysis.histogram_data.decrypted}
            icon={<CheckCircle className="h-5 w-5 text-success" />}
            variant="decrypted"
            fileSize={comprehensive_analysis.comparison_metrics.file_size_comparison?.decrypted_file_size_display}
          />
        </div>

        {/* Detailed Analysis Tabs */}
        <Card className="p-6">
          <Tabs defaultValue="timing" className="w-full">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="timing" className="flex items-center">
                <Clock className="h-4 w-4 mr-2" />
                Timing
              </TabsTrigger>
              <TabsTrigger value="quality" className="flex items-center">
                <BarChart3 className="h-4 w-4 mr-2" />
                Quality
              </TabsTrigger>
              <TabsTrigger value="security" className="flex items-center">
                <Shield className="h-4 w-4 mr-2" />
                Security
              </TabsTrigger>
              <TabsTrigger value="compression" className="flex items-center">
                <Zap className="h-4 w-4 mr-2" />
                Compression
              </TabsTrigger>
            </TabsList>

            <TabsContent value="timing" className="space-y-6">
              <CardHeader>
                <CardTitle>Processing Time Breakdown</CardTitle>
                <CardDescription>Time taken for each operation step</CardDescription>
              </CardHeader>

              {/* Encryption Phase */}
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-foreground flex items-center">
                    <Zap className="h-5 w-5 mr-2 text-warning" />
                    Encryption Phase
                  </h3>
                  <Badge variant="secondary" className="bg-warning/10 text-warning">
                    Total: {comprehensive_analysis.comparison_metrics.timing_breakdown.total_encryption_time.toFixed(3)}s
                  </Badge>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  <MetricCard
                    title="Compression"
                    value={comprehensive_analysis.comparison_metrics.timing_breakdown.compression_time}
                    unit="s"
                    description="DCT compression"
                    review={evaluateProcessingTime(comprehensive_analysis.comparison_metrics.timing_breakdown.compression_time, 'compression').review}
                    reviewType={evaluateProcessingTime(comprehensive_analysis.comparison_metrics.timing_breakdown.compression_time, 'compression').type}
                  />
                  <MetricCard
                    title="Encryption"
                    value={comprehensive_analysis.comparison_metrics.timing_breakdown.encryption_time}
                    unit="s"
                    description="AES-128 encryption"
                    review={evaluateProcessingTime(comprehensive_analysis.comparison_metrics.timing_breakdown.encryption_time, 'encryption').review}
                    reviewType={evaluateProcessingTime(comprehensive_analysis.comparison_metrics.timing_breakdown.encryption_time, 'encryption').type}
                  />
                  <MetricCard
                    title="Visualization"
                    value={comprehensive_analysis.comparison_metrics.timing_breakdown.visualization_time}
                    unit="s"
                    description="PNG generation"
                    review={comprehensive_analysis.comparison_metrics.timing_breakdown.visualization_time <= 1 ? 'Fast' :
                      comprehensive_analysis.comparison_metrics.timing_breakdown.visualization_time <= 3 ? 'Good' :
                        comprehensive_analysis.comparison_metrics.timing_breakdown.visualization_time <= 5 ? 'Slow' : 'Very Slow'}
                    reviewType={comprehensive_analysis.comparison_metrics.timing_breakdown.visualization_time <= 1 ? 'excellent' :
                      comprehensive_analysis.comparison_metrics.timing_breakdown.visualization_time <= 3 ? 'good' :
                        comprehensive_analysis.comparison_metrics.timing_breakdown.visualization_time <= 5 ? 'fair' : 'poor'}
                  />
                </div>
              </div>

              <Separator />

              {/* Decryption Phase */}
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-foreground flex items-center">
                    <Shield className="h-5 w-5 mr-2 text-success" />
                    Decryption & Analysis Phase
                  </h3>
                  <Badge variant="secondary" className="bg-success/10 text-success">
                    Total: {comprehensive_analysis.comparison_metrics.timing_breakdown.total_time.toFixed(3)}s
                  </Badge>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  <MetricCard
                    title="Decryption"
                    value={comprehensive_analysis.comparison_metrics.timing_breakdown.decryption_time}
                    unit="s"
                    description="AES-128 decryption"
                    review={evaluateProcessingTime(comprehensive_analysis.comparison_metrics.timing_breakdown.decryption_time, 'encryption').review}
                    reviewType={evaluateProcessingTime(comprehensive_analysis.comparison_metrics.timing_breakdown.decryption_time, 'encryption').type}
                  />
                  <MetricCard
                    title="Decompression"
                    value={comprehensive_analysis.comparison_metrics.timing_breakdown.decompression_time}
                    unit="s"
                    description="DCT decompression"
                    review={evaluateProcessingTime(comprehensive_analysis.comparison_metrics.timing_breakdown.decompression_time, 'compression').review}
                    reviewType={evaluateProcessingTime(comprehensive_analysis.comparison_metrics.timing_breakdown.decompression_time, 'compression').type}
                  />
                  <MetricCard
                    title="Analysis"
                    value={comprehensive_analysis.comparison_metrics.timing_breakdown.analysis_time}
                    unit="s"
                    description="Metrics calculation"
                    review={comprehensive_analysis.comparison_metrics.timing_breakdown.analysis_time <= 5 ? 'Fast' :
                      comprehensive_analysis.comparison_metrics.timing_breakdown.analysis_time <= 10 ? 'Good' :
                        comprehensive_analysis.comparison_metrics.timing_breakdown.analysis_time <= 15 ? 'Slow' : 'Very Slow'}
                    reviewType={comprehensive_analysis.comparison_metrics.timing_breakdown.analysis_time <= 5 ? 'excellent' :
                      comprehensive_analysis.comparison_metrics.timing_breakdown.analysis_time <= 10 ? 'good' :
                        comprehensive_analysis.comparison_metrics.timing_breakdown.analysis_time <= 15 ? 'fair' : 'poor'}
                  />
                </div>
              </div>

              {/* Total Round-Trip Time */}
              <div className="p-4 rounded-lg bg-primary/10 border border-primary/30">
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="text-sm font-medium text-muted-foreground">Complete Round-Trip Time</h4>
                    <p className="text-xs text-muted-foreground mt-1">Encryption + Decryption + Analysis</p>
                  </div>
                  <div className="text-3xl font-bold text-primary">
                    {(comprehensive_analysis.comparison_metrics.timing_breakdown.total_encryption_time +
                      comprehensive_analysis.comparison_metrics.timing_breakdown.total_time).toFixed(3)}
                    <span className="text-lg text-muted-foreground ml-1">s</span>
                  </div>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="quality" className="space-y-4">
              <CardHeader>
                <CardTitle>Image Quality Metrics</CardTitle>
                <CardDescription>Quality preservation analysis</CardDescription>
              </CardHeader>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <MetricCard
                  title="Mean Square Error"
                  value={comprehensive_analysis.comparison_metrics.quality_comparison.mse_original_vs_decrypted}
                  description="Lower is better (deviation from original)"
                  review={comprehensive_analysis.comparison_metrics.quality_comparison.mse_original_vs_decrypted < 50 ? 'Excellent' :
                    comprehensive_analysis.comparison_metrics.quality_comparison.mse_original_vs_decrypted < 100 ? 'Good' :
                      comprehensive_analysis.comparison_metrics.quality_comparison.mse_original_vs_decrypted < 200 ? 'Fair' : 'Poor'}
                  reviewType={comprehensive_analysis.comparison_metrics.quality_comparison.mse_original_vs_decrypted < 50 ? 'excellent' :
                    comprehensive_analysis.comparison_metrics.quality_comparison.mse_original_vs_decrypted < 100 ? 'good' :
                      comprehensive_analysis.comparison_metrics.quality_comparison.mse_original_vs_decrypted < 200 ? 'fair' : 'poor'}
                />
                <MetricCard
                  title="Peak Signal-to-Noise Ratio"
                  value={comprehensive_analysis.comparison_metrics.quality_comparison.psnr_original_vs_decrypted}
                  unit="dB"
                  description="Higher is better (quality preservation)"
                  review={evaluatePSNR(comprehensive_analysis.comparison_metrics.quality_comparison.psnr_original_vs_decrypted).review}
                  reviewType={evaluatePSNR(comprehensive_analysis.comparison_metrics.quality_comparison.psnr_original_vs_decrypted).type}
                />
              </div>
            </TabsContent>

            <TabsContent value="security" className="space-y-4">
              <CardHeader>
                <CardTitle>Encryption Security Analysis</CardTitle>
                <CardDescription>Security strength evaluation</CardDescription>
              </CardHeader>
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <MetricCard
                    title="NPCR"
                    value={comprehensive_analysis.comparison_metrics.encryption_security.npcr_original_vs_encrypted}
                    unit="%"
                    description="Number of Pixels Change Rate"
                    review={evaluateNPCR(comprehensive_analysis.comparison_metrics.encryption_security.npcr_original_vs_encrypted).review}
                    reviewType={evaluateNPCR(comprehensive_analysis.comparison_metrics.encryption_security.npcr_original_vs_encrypted).type}
                  />
                  <MetricCard
                    title="UACI"
                    value={comprehensive_analysis.comparison_metrics.encryption_security.uaci_original_vs_encrypted}
                    unit="%"
                    description="Unified Average Changing Intensity"
                    review={evaluateUACI(comprehensive_analysis.comparison_metrics.encryption_security.uaci_original_vs_encrypted).review}
                    reviewType={evaluateUACI(comprehensive_analysis.comparison_metrics.encryption_security.uaci_original_vs_encrypted).type}
                  />
                  <div className="p-4 rounded-lg border bg-muted/30">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="text-sm font-medium text-muted-foreground">Encryption Strength</h3>
                      <Badge className={`text-xs px-2 py-0.5 ${comprehensive_analysis.comparison_metrics.encryption_security.encryption_strength === 'Excellent' ? 'bg-success/20 text-success border-success/40' :
                          comprehensive_analysis.comparison_metrics.encryption_security.encryption_strength === 'Good' ? 'bg-primary/20 text-primary border-primary/40' :
                            comprehensive_analysis.comparison_metrics.encryption_security.encryption_strength === 'Fair' ? 'bg-warning/20 text-warning border-warning/40' :
                              'bg-destructive/20 text-destructive border-destructive/40'
                        }`}>
                        {comprehensive_analysis.comparison_metrics.encryption_security.encryption_strength === 'Excellent' ? '✓ Secure' :
                          comprehensive_analysis.comparison_metrics.encryption_security.encryption_strength === 'Good' ? '↗ Good' :
                            comprehensive_analysis.comparison_metrics.encryption_security.encryption_strength === 'Fair' ? '~ Fair' : '✗ Weak'}
                      </Badge>
                    </div>
                    <div className="text-2xl font-bold text-foreground mt-2">
                      {comprehensive_analysis.comparison_metrics.encryption_security.encryption_strength}
                    </div>
                    <p className="text-xs text-muted-foreground">Overall security level</p>
                  </div>
                </div>
                <div className="p-4 rounded-lg bg-success/10 border border-success/30">
                  <p className="text-sm text-success-foreground">
                    <strong>Security Guidelines:</strong> NPCR should be close to 99.6% and UACI should be around 33.4% for optimal encryption security.
                  </p>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="compression" className="space-y-4">
              <CardHeader>
                <CardTitle>Compression Analysis</CardTitle>
                <CardDescription>Efficiency and space savings</CardDescription>
              </CardHeader>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {comprehensive_analysis.comparison_metrics.compression_analysis && (
                  <>
                    <MetricCard
                      title="Compression Ratio"
                      value={comprehensive_analysis.comparison_metrics.compression_analysis.compression_ratio || 0}
                      unit=":1"
                      description="Original to compressed size ratio"
                      review={evaluateCompressionRatio(comprehensive_analysis.comparison_metrics.compression_analysis.compression_ratio || 0).review}
                      reviewType={evaluateCompressionRatio(comprehensive_analysis.comparison_metrics.compression_analysis.compression_ratio || 0).type}
                    />
                    <MetricCard
                      title="Efficiency"
                      value={comprehensive_analysis.comparison_metrics.compression_analysis.compression_efficiency || 0}
                      unit="%"
                      description="Space saved percentage"
                      review={(comprehensive_analysis.comparison_metrics.compression_analysis.compression_efficiency || 0) >= 80 ? 'Excellent' :
                        (comprehensive_analysis.comparison_metrics.compression_analysis.compression_efficiency || 0) >= 60 ? 'Good' :
                          (comprehensive_analysis.comparison_metrics.compression_analysis.compression_efficiency || 0) >= 40 ? 'Fair' : 'Poor'}
                      reviewType={(comprehensive_analysis.comparison_metrics.compression_analysis.compression_efficiency || 0) >= 80 ? 'excellent' :
                        (comprehensive_analysis.comparison_metrics.compression_analysis.compression_efficiency || 0) >= 60 ? 'good' :
                          (comprehensive_analysis.comparison_metrics.compression_analysis.compression_efficiency || 0) >= 40 ? 'fair' : 'poor'}
                    />
                    <div className="p-4 rounded-lg border bg-muted/30">
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="text-sm font-medium text-muted-foreground">Rating</h3>
                        <Badge className={`text-xs px-2 py-0.5 ${comprehensive_analysis.summary.compression_efficiency === 'High' ? 'bg-success/20 text-success border-success/40' :
                            comprehensive_analysis.summary.compression_efficiency === 'Medium' ? 'bg-primary/20 text-primary border-primary/40' :
                              'bg-warning/20 text-warning border-warning/40'
                          }`}>
                          {comprehensive_analysis.summary.compression_efficiency === 'High' ? '✓ High' :
                            comprehensive_analysis.summary.compression_efficiency === 'Medium' ? '↗ Medium' : '~ Low'}
                        </Badge>
                      </div>
                      <div className="text-2xl font-bold text-foreground mt-2">
                        {comprehensive_analysis.summary.compression_efficiency}
                      </div>
                      <p className="text-xs text-muted-foreground mt-1">Compression rating</p>
                    </div>
                  </>
                )}
              </div>

              {/* File Size Comparison Section */}
              {comprehensive_analysis.comparison_metrics.file_size_comparison && (
                <div className="mt-6 space-y-4">
                  <Separator />
                  <div>
                    <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center">
                      <Activity className="h-5 w-5 mr-2" />
                      File Size Comparison
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div className="p-4 rounded-lg border bg-accent/10 border-accent/30">
                        <h4 className="text-sm font-medium text-muted-foreground mb-2">Original Image</h4>
                        <div className="text-2xl font-bold text-foreground">
                          {comprehensive_analysis.comparison_metrics.file_size_comparison.original_file_size_display}
                        </div>
                        <p className="text-xs text-muted-foreground mt-1">Source file size</p>
                      </div>

                      <div className="p-4 rounded-lg border bg-success/10 border-success/30">
                        <h4 className="text-sm font-medium text-muted-foreground mb-2">Decrypted Image</h4>
                        <div className="text-2xl font-bold text-foreground">
                          {comprehensive_analysis.comparison_metrics.file_size_comparison.decrypted_file_size_display}
                        </div>
                        <p className="text-xs text-muted-foreground mt-1">Recovered file size</p>
                      </div>


                    </div>

                    <div className="mt-4 p-4 rounded-lg bg-muted/30 border border-border/50">
                      <p className="text-sm text-muted-foreground">
                        <strong>Note:</strong> File size differences may occur due to image format encoding, compression settings, and metadata.
                        The encryption/decryption process preserves image data while the final file size depends on the output format.
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </TabsContent>
          </Tabs>
        </Card>
      </div>
    </div>
  );
}