import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Shield, Lock, Unlock, Key, AlertTriangle } from 'lucide-react';
import { cn } from '@/lib/utils';

interface EncryptionFormProps {
  onSubmit: (key: string, operation: 'encrypt' | 'decrypt') => void;
  isLoading: boolean;
  hasImage: boolean;
  selectedFile?: File | null;
}

export const EncryptionForm: React.FC<EncryptionFormProps> = ({
  onSubmit,
  isLoading,
  hasImage,
  selectedFile,
}) => {
  const [key, setKey] = useState('');
  const [operation, setOperation] = useState<'encrypt' | 'decrypt'>('encrypt');
  const [keyError, setKeyError] = useState('');

  // Auto-detect operation based on file type
  React.useEffect(() => {
    if (selectedFile) {
      if (selectedFile.name.endsWith('.bin')) {
        setOperation('decrypt');
      } else if (selectedFile.type.startsWith('image/')) {
        setOperation('encrypt');
      }
    }
  }, [selectedFile]);

  const validateKey = (value: string) => {
    if (value.length === 0) {
      setKeyError('Encryption key is required');
      return false;
    }
    if (value.length !== 16) {
      setKeyError('Key must be exactly 16 characters long');
      return false;
    }
    setKeyError('');
    return true;
  };

  const handleKeyChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setKey(value);
    if (value.length > 0) {
      validateKey(value);
    } else {
      setKeyError('');
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!hasImage) {
      return;
    }
    if (validateKey(key)) {
      onSubmit(key, operation);
    }
  };

  return (
    <Card className="p-6 bg-gradient-secondary border-border/50">
      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="flex items-center space-x-3">
          <div className="p-2 rounded-lg bg-primary/10">
            <Shield className="h-5 w-5 text-primary" />
          </div>
          <h3 className="text-lg font-semibold text-foreground">
            Encryption Settings
          </h3>
        </div>

        <div className="space-y-2">
          <Label htmlFor="key" className="text-foreground font-medium">
            Encryption Key
          </Label>
          <div className="relative">
            <div className="absolute left-3 top-1/2 transform -translate-y-1/2">
              <Key className="h-4 w-4 text-muted-foreground" />
            </div>
            <Input
              id="key"
              type="text"
              value={key}
              onChange={handleKeyChange}
              placeholder="Enter 16-character key"
              className={cn(
                "pl-10 transition-all duration-300",
                keyError 
                  ? "border-destructive focus:border-destructive" 
                  : key.length === 16 
                    ? "border-success focus:border-success" 
                    : ""
              )}
              maxLength={16}
            />
          </div>
          <div className="flex items-center justify-between text-xs">
            <span className={cn(
              "transition-colors duration-300",
              keyError 
                ? "text-destructive" 
                : key.length === 16 
                  ? "text-success" 
                  : "text-muted-foreground"
            )}>
              {keyError || `${key.length}/16 characters`}
            </span>
            {key.length === 16 && !keyError && (
              <span className="text-success">✓ Valid key</span>
            )}
          </div>
        </div>

        <div className="space-y-3">
          <Label className="text-foreground font-medium">Operation</Label>
          <RadioGroup
            value={operation}
            onValueChange={(value) => setOperation(value as 'encrypt' | 'decrypt')}
            className="flex space-x-6"
          >
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="encrypt" id="encrypt" />
              <Label
                htmlFor="encrypt"
                className="flex items-center space-x-2 cursor-pointer"
              >
                <Lock className="h-4 w-4 text-primary" />
                <span>Encrypt</span>
              </Label>
            </div>
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="decrypt" id="decrypt" />
              <Label
                htmlFor="decrypt"
                className="flex items-center space-x-2 cursor-pointer"
              >
                <Unlock className="h-4 w-4 text-accent" />
                <span>Decrypt</span>
              </Label>
            </div>
          </RadioGroup>
        </div>

        {!hasImage && (
          <Alert className="border-warning/50 bg-warning/10">
            <AlertTriangle className="h-4 w-4 text-warning" />
            <AlertDescription className="text-warning-foreground">
              Please select a file first. Use images (.jpg, .png, .jpeg) for encryption or .bin files for decryption.
            </AlertDescription>
          </Alert>
        )}

        <Button
          type="submit"
          variant="gradient"
          className="w-full"
          disabled={isLoading || !hasImage || key.length !== 16}
        >
          {isLoading ? (
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" />
              <span>Processing...</span>
            </div>
          ) : (
            <div className="flex items-center space-x-2">
              {operation === 'encrypt' ? (
                <Lock className="h-4 w-4" />
              ) : (
                <Unlock className="h-4 w-4" />
              )}
              <span>
                {operation === 'encrypt' ? 'Encrypt Image' : 'Decrypt File'}
              </span>
            </div>
          )}
        </Button>
      </form>
    </Card>
  );
};