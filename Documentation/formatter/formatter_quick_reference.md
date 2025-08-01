# Formatter Quick Reference

## Quick Start

```python
from scripts.formatter import Formatter

# Initialize
formatter = Formatter(debug_mode=True)

# Create training prompts
training_prompts = formatter.create_training_prompts(
    df, topic="Respiratory Issues", include_rationale=True
)

# Create inference prompts
inference_prompts = formatter.create_inference_prompts(
    df, topic="Respiratory Issues"
)
```

## Required Data Format

```python
# Training data
df = pd.DataFrame({
    'C_BioSense_ID': ['P001', 'P002'],
    'Context_Block': ['Patient data...', 'Patient data...'],
    'Standardized_Rating': [1, 2],
    'Rationale of Rating': ['Expert reason...', 'Expert reason...']
})

# Inference data
df = pd.DataFrame({
    'C_BioSense_ID': ['P001', 'P002'],
    'Context_Block': ['Patient data...', 'Patient data...']
})
```

## Common Operations

### Mixed Processing
```python
mixed_prompts = formatter.create_mixed_prompts(df, topic="Topic")
training = mixed_prompts['training']
inference = mixed_prompts['inference']
```

### Batch Processing
```python
prompts = formatter.batch_process(df, topic="Topic", batch_size=100)
```

### Custom Templates
```python
custom_template = """CUSTOM TEMPLATE:
Topic: {topic}
Data: {context_block}
Rating: {rating}
Reason: {rationale}"""

prompts = formatter.create_training_prompts(
    df, topic="Topic", custom_template=custom_template
)
```

### File Operations
```python
# Save prompts
formatter.save_prompts_to_jsonl(prompts, "output.jsonl")

# Load prompts
loaded_prompts = formatter.load_prompts_from_jsonl("input.jsonl")
```

## Validation

```python
# Validate input data
is_valid, errors = formatter.validate_input_data(df)

# Validate individual prompts
is_valid, errors = formatter.validate_prompt_format(prompt)
```

## Statistics

```python
stats = formatter.get_prompt_statistics(prompts)
print(f"Total: {stats['total_prompts']}")
print(f"Avg length: {stats['avg_prompt_length']}")
print(f"Rating dist: {stats['rating_distribution']}")
```

## Template Placeholders

- `{topic}`: Analysis topic
- `{context_block}`: Patient data
- `{rating}`: Expert rating (training only)
- `{rationale}`: Expert rationale (training only)

## Output Format

### Training Prompt
```python
{
    'id': 'P001',
    'prompt': 'INPUT:\nTopic: {topic}\nData: {context_block}\nRating: {rating}\nRationale: {rationale}',
    'rating': 1,
    'rationale': 'Expert reason',
    'topic': 'Respiratory Issues',
    'row_index': 0
}
```

### Inference Prompt
```python
{
    'id': 'P001',
    'prompt': 'INPUT:\nTopic: {topic}\nData: {context_block}\nOUTPUT:\nRating: <predicted>\nRationale: <explanation>',
    'topic': 'Respiratory Issues',
    'row_index': 0
}
```

## Error Handling

```python
try:
    prompts = formatter.create_training_prompts(df, topic)
except ValidationError as e:
    print(f"Validation error: {e}")
except FormatterError as e:
    print(f"Formatter error: {e}")
``` 