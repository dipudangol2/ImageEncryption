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
    };
    compression_analysis: any;
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
}> = ({ title, value, unit = '', description, variant = 'default' }) => {
  const colors = {
    default: 'bg-muted/30 border-border',
    success: 'bg-success/10 border-success/30',
    warning: 'bg-warning/10 border-warning/30',
    destructive: 'bg-destructive/10 border-destructive/30',
  };

  return (
    <div className={`p-4 rounded-lg border ${colors[variant]}`}>
      <div className="space-y-2">
        <h3 className="text-sm font-medium text-muted-foreground">{title}</h3>
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
}> = ({ title, imageUrl, histogramData, icon, variant }) => {
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
                link.href = `/api/download/${imageUrl.split('/').pop()}`;
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
          <MetricCard
            title="Entropy"
            value={histogramData.entropy}
            unit="bits"
            description="Information density"
          />
          <MetricCard
            title="Mean"
            value={histogramData.mean}
            description="Average intensity"
          />
          <MetricCard
            title="Std Dev"
            value={histogramData.std_dev}
            description="Intensity variation"
          />
          <MetricCard
            title="Unique Values"
            value={histogramData.unique_values}
            description="Color diversity"
          />
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

export default function PerformanceResults() {
  const navigate = useNavigate();
  const location = useLocation();
  const [data, setData] = useState<PerformanceResultsState | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (location.state && location.state.results) {
      setData(location.state.results);
      setLoading(false);
    } else {
      // Redirect back to home if no data
      navigate('/');
    }
  }, [location.state, navigate]);

  if (loading || !data) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-background to-muted/20 flex items-center justify-center">
        <Card className="p-8 text-center">
          <div className="space-y-4">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto"></div>
            <p className="text-muted-foreground">Loading performance analysis...</p>
          </div>
        </Card>
      </div>
    );
  }

  const { files, comprehensive_analysis } = data;

  return (
    <div className="min-h-screen bg-gradient-to-br from-background to-muted/20 p-4">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <Card className="p-6 bg-gradient-primary border-accent/30">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Button
                variant="ghost"
                onClick={() => navigate('/')}
                className="text-primary-foreground hover:bg-primary-foreground/10"
              >
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back
              </Button>
              <div>
                <h1 className="text-2xl font-bold text-primary-foreground">
                  Performance Analysis Results
                </h1>
                <p className="text-primary-foreground/70 text-sm">
                  Comprehensive comparison of original, encrypted, and decrypted images
                </p>
              </div>
            </div>
            <div className="flex space-x-2">
              <Badge variant="secondary" className="bg-primary-foreground/10 text-primary-foreground">
                {comprehensive_analysis.summary.overall_quality}
              </Badge>
              <Badge variant="secondary" className="bg-primary-foreground/10 text-primary-foreground">
                {comprehensive_analysis.summary.encryption_security}
              </Badge>
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

            <TabsContent value="timing" className="space-y-4">
              <CardHeader>
                <CardTitle>Processing Time Breakdown</CardTitle>
                <CardDescription>Time taken for each operation step</CardDescription>
              </CardHeader>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                <MetricCard
                  title="Decryption"
                  value={comprehensive_analysis.comparison_metrics.timing_breakdown.decryption_time}
                  unit="s"
                />
                <MetricCard
                  title="Decompression"
                  value={comprehensive_analysis.comparison_metrics.timing_breakdown.decompression_time}
                  unit="s"
                />
                <MetricCard
                  title="Analysis"
                  value={comprehensive_analysis.comparison_metrics.timing_breakdown.analysis_time}
                  unit="s"
                />
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
                  description="Lower is better"
                />
                <MetricCard
                  title="Peak Signal-to-Noise Ratio"
                  value={comprehensive_analysis.comparison_metrics.quality_comparison.psnr_original_vs_decrypted}
                  unit="dB"
                  description="Higher is better"
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
                  />
                  <MetricCard
                    title="UACI"
                    value={comprehensive_analysis.comparison_metrics.encryption_security.uaci_original_vs_encrypted}
                    unit="%"
                    description="Unified Average Changing Intensity"
                  />
                  <div className="p-4 rounded-lg border bg-muted/30">
                    <h3 className="text-sm font-medium text-muted-foreground">Encryption Strength</h3>
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
                    />
                    <MetricCard
                      title="Bits Per Pixel"
                      value={comprehensive_analysis.comparison_metrics.compression_analysis.bits_per_pixel || 0}
                      unit="bpp"
                    />
                    <MetricCard
                      title="Efficiency"
                      value={comprehensive_analysis.comparison_metrics.compression_analysis.compression_efficiency || 0}
                      unit="%"
                    />
                    <div className="p-4 rounded-lg border bg-muted/30">
                      <h3 className="text-sm font-medium text-muted-foreground">Rating</h3>
                      <div className="text-2xl font-bold text-foreground mt-2">
                        {comprehensive_analysis.summary.compression_efficiency}
                      </div>
                    </div>
                  </>
                )}
              </div>
            </TabsContent>
          </Tabs>
        </Card>
      </div>
    </div>
  );
}