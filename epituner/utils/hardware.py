#!/usr/bin/env python3
"""
Simple hardware detection utilities
"""

import torch
import subprocess
from typing import Dict, List


def get_gpu_info() -> Dict[str, any]:
    """Get simple GPU information"""
    info = {
        'has_cuda': False,
        'gpu_name': 'CPU Only',
        'memory_gb': 0,
        'device_count': 0,
        'recommended_batch_size': 1,
        'can_use_fp16': False
    }
    
    try:
        if torch.cuda.is_available():
            info['has_cuda'] = True
            info['device_count'] = torch.cuda.device_count()
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
            info['can_use_fp16'] = True
            
            # Simple batch size recommendation based on memory
            if info['memory_gb'] >= 8:
                info['recommended_batch_size'] = 2
            elif info['memory_gb'] >= 6:
                info['recommended_batch_size'] = 1
            else:
                info['recommended_batch_size'] = 1
                
    except Exception:
        pass  # Keep defaults
    
    return info


def check_ollama_available() -> bool:
    """Check if Ollama is installed and running"""
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except:
        return False


def get_ollama_models() -> List[str]:
    """Get list of available Ollama models"""
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            models = []
            for line in lines:
                if line.strip():
                    model_name = line.split()[0]
                    models.append(model_name)
            return models
    except:
        pass
    return []


def validate_system() -> Dict[str, bool]:
    """Basic system validation"""
    checks = {}
    
    # Check PyTorch
    try:
        import torch
        checks['pytorch'] = True
    except ImportError:
        checks['pytorch'] = False
    
    # Check transformers
    try:
        import transformers
        checks['transformers'] = True
    except ImportError:
        checks['transformers'] = False
    
    # Check PEFT
    try:
        import peft
        checks['peft'] = True
    except ImportError:
        checks['peft'] = False
    
    # Check Ollama
    checks['ollama'] = check_ollama_available()
    
    return checks

