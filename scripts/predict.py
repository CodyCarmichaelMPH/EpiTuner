#!/usr/bin/env python3
"""
Simple CLI prediction script for EpiTuner
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from epituner.core.simple_inference import SimpleInference
from epituner.core.data_processor import MedicalDataProcessor


def main():
    parser = argparse.ArgumentParser(description="Make predictions with trained model")
    parser.add_argument("--model", required=True, help="Path to trained model directory")
    parser.add_argument("--data", required=True, help="Path to CSV data file")
    parser.add_argument("--topic", required=True, help="Classification topic")
    parser.add_argument("--output", required=True, help="Output JSON file")
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    inference = SimpleInference(args.model)
    if not inference.load_model():
        print("❌ Failed to load model")
        return 1
    
    # Load data
    print("Loading data...")
    try:
        df = MedicalDataProcessor.load_data(args.data)
        records = df.to_dict('records')
        print(f"Loaded {len(records)} records")
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1
    
    # Make predictions
    print("Making predictions...")
    results = []
    
    for i, record in enumerate(records):
        if i % 10 == 0:
            print(f"Processing {i+1}/{len(records)}...")
        
        pred = inference.predict(record, args.topic)
        results.append({
            'id': record.get('C_Biosense_ID', f'record_{i}'),
            'classification': pred.classification,
            'confidence': pred.confidence,
            'confidence_score': pred.confidence_score,
            'rationale': pred.rationale,
            'expert_rating': record.get('Expert Rating', 'Unknown'),
            'agreement': pred.classification == record.get('Expert Rating', '')
        })
    
    # Save results
    output_data = {
        'model_path': args.model,
        'topic': args.topic,
        'total_records': len(records),
        'predictions': results
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Calculate accuracy
    agreements = [r['agreement'] for r in results]
    accuracy = sum(agreements) / len(agreements) if agreements else 0
    
    print(f"✅ Predictions completed!")
    print(f"📁 Results saved to: {args.output}")
    print(f"🎯 Accuracy: {accuracy:.1%}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

