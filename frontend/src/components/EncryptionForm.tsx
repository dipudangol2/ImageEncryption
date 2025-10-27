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
  onSubmit: (password: string, operation: 'encrypt' | 'decrypt') => void;
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
  const [password, setPassword] = useState('');
  const [operation, setOperation] = useState<'encrypt' | 'decrypt'>('encrypt');
  const [passwordError, setPasswordError] = useState('');

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

  const validatePassword = (value: string) => {
    if (value.trim().length === 0) {
      setPasswordError('Password is required');
      return false;
    }
    setPasswordError('');
    return true;
  };

  const handlePasswordChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setPassword(value);
    if (value.trim().length > 0) {
      validatePassword(value);
    } else {
      setPasswordError('');
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!hasImage) {
      return;
    }
    if (validatePassword(password)) {
      onSubmit(password, operation);
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
          <Label htmlFor="password" className="text-foreground font-medium">
            Key
          </Label>
          <div className="relative">
            <div className="absolute left-3 top-1/2 transform -translate-y-1/2">
              <Key className="h-4 w-4 text-muted-foreground" />
            </div>
            <Input
              id="password"
              type="password"
              value={password}
              onChange={handlePasswordChange}
              placeholder="Enter your key"
              className={cn(
                "pl-10 transition-all duration-300",
                passwordError
                  ? "border-destructive focus:border-destructive"
                  : password.trim().length > 0
                    ? "border-success focus:border-success"
                    : ""
              )}
            />
          </div>
          <div className="flex items-center justify-between text-xs">
            <span className={cn(
              "transition-colors duration-300",
              passwordError
                ? "text-destructive"
                : password.trim().length > 0
                  ? "text-success"
                  : "text-muted-foreground"
            )}>
              {passwordError || "Use any length key for encryption"}
            </span>
            {password.trim().length > 0 && !passwordError && (
              <span className="text-success">✓ Valid password</span>
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
            <AlertTriangle className="h-4 w-4 text-warnin" />
            <AlertDescription className="text-warning">
              Please select a file first. Use images (.jpg, .png, .jpeg) for encryption or .bin files for decryption.
            </AlertDescription>
          </Alert>
        )}

        <Button
          type="submit"
          variant="gradient"
          className="w-full"
          disabled={isLoading || !hasImage || password.trim().length === 0}
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