#!/usr/bin/env python3
"""
Simple CLI training script for EpiTuner
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from epituner.core.data_processor import MedicalDataProcessor
from epituner.core.simple_trainer import SimpleTrainer


def main():
    parser = argparse.ArgumentParser(description="Train medical classification model")
    parser.add_argument("--data", required=True, help="Path to CSV data file")
    parser.add_argument("--model", required=True, help="Model name (e.g., microsoft/Phi-3-mini-4k-instruct)")
    parser.add_argument("--topic", required=True, help="Classification topic")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    
    args = parser.parse_args()
    
    # Load and validate data
    print("Loading data...")
    try:
        df = MedicalDataProcessor.load_data(args.data)
        training_texts = MedicalDataProcessor.prepare_for_training(df, args.topic)
        print(f"Loaded {len(training_texts)} training examples")
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1
    
    # Train model
    print("Starting training...")
    trainer = SimpleTrainer(args.model)
    result = trainer.train(training_texts, args.output, args.epochs, args.lr)
    
    if result.success:
        print(f"✅ Training completed successfully!")
        print(f"📁 Model saved to: {result.model_path}")
        print(f"📉 Final loss: {result.training_loss:.4f}")
        return 0
    else:
        print(f"❌ Training failed: {result.error_message}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

