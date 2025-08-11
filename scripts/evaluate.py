#!/usr/bin/env python3
"""
EpiTuner Model Evaluation Script
Evaluate trained models against expert annotations
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class ModelEvaluator:
    """Evaluate model performance against expert annotations"""
    
    def __init__(self):
        self.label_mapping = {
            'Match': 1,
            'Not a Match': 0,
            'Unknown/Not able to determine': 2
        }
        self.reverse_mapping = {v: k for k, v in self.label_mapping.items()}
    
    def load_predictions(self, predictions_path: str) -> Tuple[List[Dict], str]:
        """Load model predictions from JSON file"""
        with open(predictions_path, 'r') as f:
            data = json.load(f)
        
        predictions = data.get('predictions', [])
        topic = data.get('classification_topic', 'Unknown')
        
        return predictions, topic
    
    def load_ground_truth(self, data_path: str) -> Dict[str, str]:
        """Load expert annotations from CSV"""
        df = pd.read_csv(data_path)
        
        ground_truth = {}
        for _, row in df.iterrows():
            biosense_id = str(row['C_Biosense_ID'])
            expert_rating = row['Expert Rating']
            ground_truth[biosense_id] = expert_rating
        
        return ground_truth
    
    def align_predictions_and_truth(self, predictions: List[Dict], ground_truth: Dict[str, str]) -> Tuple[List[str], List[str], List[Dict]]:
        """Align model predictions with expert annotations"""
        
        model_preds = []
        expert_labels = []
        aligned_records = []
        
        for pred in predictions:
            biosense_id = str(pred.get('biosense_id', ''))
            
            if biosense_id in ground_truth:
                model_classification = pred.get('classification', 'Unknown/Not able to determine')
                expert_classification = ground_truth[biosense_id]
                
                model_preds.append(model_classification)
                expert_labels.append(expert_classification)
                
                aligned_records.append({
                    'biosense_id': biosense_id,
                    'model_prediction': model_classification,
                    'expert_label': expert_classification,
                    'confidence': pred.get('confidence_level', 'Unknown'),
                    'confidence_score': pred.get('confidence_score', 0.0),
                    'rationale': pred.get('rationale', ''),
                    'agreement': model_classification == expert_classification
                })
        
        return model_preds, expert_labels, aligned_records
    
    def calculate_metrics(self, model_preds: List[str], expert_labels: List[str]) -> Dict[str, Any]:
        """Calculate evaluation metrics"""
        
        # Convert to numeric labels for sklearn
        model_numeric = [self.label_mapping[pred] for pred in model_preds]
        expert_numeric = [self.label_mapping[label] for label in expert_labels]
        
        # Overall accuracy
        accuracy = accuracy_score(expert_numeric, model_numeric)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            expert_numeric, model_numeric, average=None, zero_division=0
        )
        
        # Weighted averages
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            expert_numeric, model_numeric, average='weighted', zero_division=0
        )
        
        # Macro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            expert_numeric, model_numeric, average='macro', zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(expert_numeric, model_numeric)
        
        # Per-class results
        class_results = {}
        for i, (label, class_name) in enumerate(self.reverse_mapping.items()):
            class_results[class_name] = {
                'precision': float(precision[i]) if i < len(precision) else 0.0,
                'recall': float(recall[i]) if i < len(recall) else 0.0,
                'f1_score': float(f1[i]) if i < len(f1) else 0.0,
                'support': int(support[i]) if i < len(support) else 0
            }
        
        return {
            'overall': {
                'accuracy': float(accuracy),
                'precision_weighted': float(precision_weighted),
                'recall_weighted': float(recall_weighted),
                'f1_weighted': float(f1_weighted),
                'precision_macro': float(precision_macro),
                'recall_macro': float(recall_macro),
                'f1_macro': float(f1_macro),
                'total_samples': len(model_preds)
            },
            'per_class': class_results,
            'confusion_matrix': cm.tolist(),
            'class_names': list(self.reverse_mapping.values())
        }
    
    def analyze_confidence(self, aligned_records: List[Dict]) -> Dict[str, Any]:
        """Analyze confidence calibration"""
        
        # Group by confidence level
        confidence_analysis = {}
        
        for record in aligned_records:
            confidence_level = record['confidence']
            
            if confidence_level not in confidence_analysis:
                confidence_analysis[confidence_level] = {
                    'total': 0,
                    'correct': 0,
                    'confidence_scores': []
                }
            
            confidence_analysis[confidence_level]['total'] += 1
            confidence_analysis[confidence_level]['confidence_scores'].append(record['confidence_score'])
            
            if record['agreement']:
                confidence_analysis[confidence_level]['correct'] += 1
        
        # Calculate accuracy for each confidence level
        for level in confidence_analysis:
            total = confidence_analysis[level]['total']
            correct = confidence_analysis[level]['correct']
            confidence_analysis[level]['accuracy'] = correct / total if total > 0 else 0.0
            confidence_analysis[level]['avg_confidence_score'] = np.mean(confidence_analysis[level]['confidence_scores'])
        
        return confidence_analysis
    
    def analyze_disagreements(self, aligned_records: List[Dict]) -> Dict[str, Any]:
        """Analyze cases where model and expert disagree"""
        
        disagreements = [r for r in aligned_records if not r['agreement']]
        
        if not disagreements:
            return {'total_disagreements': 0}
        
        # Disagreement patterns
        disagreement_patterns = Counter()
        for record in disagreements:
            pattern = f"{record['expert_label']} → {record['model_prediction']}"
            disagreement_patterns[pattern] += 1
        
        # Confidence distribution of disagreements
        disagreement_confidence = Counter(r['confidence'] for r in disagreements)
        
        # Most confident disagreements (potential model errors)
        confident_disagreements = [
            r for r in disagreements 
            if r['confidence'] in ['Very Confident', 'Confident']
        ]
        
        return {
            'total_disagreements': len(disagreements),
            'disagreement_rate': len(disagreements) / len(aligned_records),
            'disagreement_patterns': dict(disagreement_patterns),
            'confidence_distribution': dict(disagreement_confidence),
            'confident_disagreements': len(confident_disagreements),
            'sample_disagreements': disagreements[:5]  # Sample for review
        }
    
    def generate_report(self, metrics: Dict, confidence_analysis: Dict, 
                       disagreement_analysis: Dict, topic: str) -> str:
        """Generate a comprehensive evaluation report"""
        
        report = f"""
# EpiTuner Model Evaluation Report

## Classification Task
**Topic:** {topic}

## Overall Performance
- **Accuracy:** {metrics['overall']['accuracy']:.3f}
- **Weighted F1-Score:** {metrics['overall']['f1_weighted']:.3f}
- **Macro F1-Score:** {metrics['overall']['f1_macro']:.3f}
- **Total Samples:** {metrics['overall']['total_samples']}

## Per-Class Performance
"""
        
        for class_name, class_metrics in metrics['per_class'].items():
            report += f"""
### {class_name}
- Precision: {class_metrics['precision']:.3f}
- Recall: {class_metrics['recall']:.3f}
- F1-Score: {class_metrics['f1_score']:.3f}
- Support: {class_metrics['support']}
"""
        
        report += f"""
## Confidence Analysis
"""
        
        for level, analysis in confidence_analysis.items():
            report += f"""
### {level}
- Samples: {analysis['total']}
- Accuracy: {analysis['accuracy']:.3f}
- Avg Confidence Score: {analysis['avg_confidence_score']:.3f}
"""
        
        report += f"""
## Disagreement Analysis
- **Total Disagreements:** {disagreement_analysis['total_disagreements']}
- **Disagreement Rate:** {disagreement_analysis['disagreement_rate']:.3f}
- **Confident Disagreements:** {disagreement_analysis.get('confident_disagreements', 0)}

### Common Disagreement Patterns
"""
        
        for pattern, count in disagreement_analysis.get('disagreement_patterns', {}).items():
            report += f"- {pattern}: {count} cases\n"
        
        return report
    
    def save_detailed_results(self, aligned_records: List[Dict], output_path: str):
        """Save detailed results to CSV for further analysis"""
        
        df = pd.DataFrame(aligned_records)
        df.to_csv(output_path, index=False)
        print(f"Detailed results saved to: {output_path}")
    
    def plot_confusion_matrix(self, metrics: Dict, output_path: str):
        """Create and save confusion matrix plot"""
        
        cm = np.array(metrics['confusion_matrix'])
        class_names = metrics['class_names']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix: Model vs Expert Annotations')
        plt.xlabel('Model Predictions')
        plt.ylabel('Expert Labels')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to: {output_path}")
    
    def plot_confidence_calibration(self, confidence_analysis: Dict, output_path: str):
        """Create confidence calibration plot"""
        
        levels = list(confidence_analysis.keys())
        accuracies = [confidence_analysis[level]['accuracy'] for level in levels]
        confidence_scores = [confidence_analysis[level]['avg_confidence_score'] for level in levels]
        
        plt.figure(figsize=(12, 6))
        
        # Plot 1: Accuracy by confidence level
        plt.subplot(1, 2, 1)
        plt.bar(levels, accuracies)
        plt.title('Accuracy by Confidence Level')
        plt.xlabel('Confidence Level')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        
        # Plot 2: Confidence score vs accuracy
        plt.subplot(1, 2, 2)
        plt.scatter(confidence_scores, accuracies, s=100)
        plt.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Perfect Calibration')
        plt.title('Confidence Calibration')
        plt.xlabel('Average Confidence Score')
        plt.ylabel('Accuracy')
        plt.legend()
        
        for i, level in enumerate(levels):
            plt.annotate(level, (confidence_scores[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confidence calibration plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions JSON file")
    parser.add_argument("--ground_truth", type=str, required=True, help="Path to ground truth CSV file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Load data
    print("Loading predictions and ground truth...")
    predictions, topic = evaluator.load_predictions(args.predictions)
    ground_truth = evaluator.load_ground_truth(args.ground_truth)
    
    # Align predictions with ground truth
    model_preds, expert_labels, aligned_records = evaluator.align_predictions_and_truth(
        predictions, ground_truth
    )
    
    print(f"Aligned {len(aligned_records)} records for evaluation")
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = evaluator.calculate_metrics(model_preds, expert_labels)
    
    # Analyze confidence
    print("Analyzing confidence calibration...")
    confidence_analysis = evaluator.analyze_confidence(aligned_records)
    
    # Analyze disagreements
    print("Analyzing disagreements...")
    disagreement_analysis = evaluator.analyze_disagreements(aligned_records)
    
    # Generate report
    report = evaluator.generate_report(metrics, confidence_analysis, disagreement_analysis, topic)
    
    # Save results
    report_path = output_dir / "evaluation_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Evaluation report saved to: {report_path}")
    
    # Save metrics as JSON
    metrics_path = output_dir / "evaluation_metrics.json"
    full_results = {
        'metrics': metrics,
        'confidence_analysis': confidence_analysis,
        'disagreement_analysis': disagreement_analysis,
        'topic': topic
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    print(f"Detailed metrics saved to: {metrics_path}")
    
    # Save detailed results
    detailed_path = output_dir / "detailed_results.csv"
    evaluator.save_detailed_results(aligned_records, detailed_path)
    
    # Create plots
    cm_path = output_dir / "confusion_matrix.png"
    evaluator.plot_confusion_matrix(metrics, cm_path)
    
    calibration_path = output_dir / "confidence_calibration.png"
    evaluator.plot_confidence_calibration(confidence_analysis, calibration_path)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Topic: {topic}")
    print(f"Accuracy: {metrics['overall']['accuracy']:.3f}")
    print(f"Weighted F1: {metrics['overall']['f1_weighted']:.3f}")
    print(f"Disagreement Rate: {disagreement_analysis['disagreement_rate']:.3f}")
    print(f"Total Samples: {metrics['overall']['total_samples']}")
    print("="*50)


if __name__ == "__main__":
    main()
