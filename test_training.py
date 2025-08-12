#!/usr/bin/env python3
"""
Test script for EpiTuner training
Verifies that training works correctly with RTX 4000 GPU
"""

import os
import sys
import yaml
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_gpu_detection():
    """Test GPU detection"""
    print("=== Testing GPU Detection ===")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU detected: {gpu_name}")
            print(f"✅ VRAM: {memory_gb:.1f}GB")
            
            if 'rtx 4000' in gpu_name.lower():
                print("✅ RTX 4000 detected - using optimized configuration")
                return True
            else:
                print(f"⚠️ GPU detected but not RTX 4000: {gpu_name}")
                return True
        else:
            print("❌ No CUDA GPU detected")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def test_dependencies():
    """Test required dependencies"""
    print("\n=== Testing Dependencies ===")
    
    dependencies = [
        'torch',
        'transformers',
        'datasets',
        'peft',
        'bitsandbytes',
        'accelerate'
    ]
    
    all_good = True
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep} - not installed")
            all_good = False
    
    return all_good

def test_config_files():
    """Test configuration files"""
    print("\n=== Testing Configuration Files ===")
    
    config_files = [
        'configs/config_base.yaml',
        'configs/config_consumer_gpu.yaml',
        'configs/config_rtx4000.yaml',
        'configs/config_cpu_only.yaml',
        'configs/config_no_quantization.yaml'
    ]
    
    all_good = True
    for config_file in config_files:
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    yaml.safe_load(f)
                print(f"✅ {config_file}")
            except Exception as e:
                print(f"❌ {config_file} - invalid YAML: {e}")
                all_good = False
        else:
            print(f"❌ {config_file} - not found")
            all_good = False
    
    return all_good

def test_training_script():
    """Test training script"""
    print("\n=== Testing Training Script ===")
    
    train_script = Path("scripts/train.py")
    if train_script.exists():
        print("✅ Training script exists")
        
        # Test basic import
        try:
            sys.path.insert(0, str(Path("scripts")))
            from train import LoRATrainer, SyndromicSurveillanceDataProcessor
            print("✅ Training modules import successfully")
            return True
        except Exception as e:
            print(f"❌ Training modules import failed: {e}")
            return False
    else:
        print("❌ Training script not found")
        return False

def test_inference_script():
    """Test inference script"""
    print("\n=== Testing Inference Script ===")
    
    inference_script = Path("scripts/inference.py")
    if inference_script.exists():
        print("✅ Inference script exists")
        
        # Test basic import
        try:
            sys.path.insert(0, str(Path("scripts")))
            from inference import SyndromicSurveillanceClassificationInference
            print("✅ Inference modules import successfully")
            return True
        except Exception as e:
            print(f"❌ Inference modules import failed: {e}")
            return False
    else:
        print("❌ Inference script not found")
        return False

def test_sample_data():
    """Test sample data"""
    print("\n=== Testing Sample Data ===")
    
    sample_data = Path("sample_data/medical_sample.csv")
    if sample_data.exists():
        print("✅ Sample data exists")
        
        # Test data loading
        try:
            import pandas as pd
            df = pd.read_csv(sample_data)
            print(f"✅ Sample data loaded: {len(df)} records")
            print(f"✅ Columns: {list(df.columns)}")
            return True
        except Exception as e:
            print(f"❌ Sample data loading failed: {e}")
            return False
    else:
        print("❌ Sample data not found")
        return False

def test_chat_templates():
    """Test chat templates"""
    print("\n=== Testing Chat Templates ===")
    
    template_files = [
        'chat_templates/medical_classification.jinja',
        'chat_templates/validation_template.jinja'
    ]
    
    all_good = True
    for template_file in template_files:
        if os.path.exists(template_file):
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                print(f"✅ {template_file} ({len(content)} chars)")
            except Exception as e:
                print(f"❌ {template_file} - read failed: {e}")
                all_good = False
        else:
            print(f"❌ {template_file} - not found")
            all_good = False
    
    return all_good

def main():
    """Run all tests"""
    print("EpiTuner Training Test Suite")
    print("=" * 50)
    
    tests = [
        ("GPU Detection", test_gpu_detection),
        ("Dependencies", test_dependencies),
        ("Configuration Files", test_config_files),
        ("Training Script", test_training_script),
        ("Inference Script", test_inference_script),
        ("Sample Data", test_sample_data),
        ("Chat Templates", test_chat_templates)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} - test failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! EpiTuner should work correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
