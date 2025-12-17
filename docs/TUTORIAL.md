# Hamerspace Tutorial

This tutorial walks through using Hamerspace for model optimization, from basic usage to advanced scenarios.

## Installation

```bash
# Basic installation
pip install hamerspace

# Full installation with all backends
pip install hamerspace[full]

# Development installation
git clone https://github.com/yourusername/hamerspace.git
cd hamerspace
pip install -e ".[dev,full]"
```

## Tutorial 1: Basic Quantization

Let's start by quantizing a simple PyTorch model.

```python
import torch
import torch.nn as nn
from hamerspace import Optimizer, OptimizationGoal, Constraints

# Define a simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 8 * 8, 10)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Create and save model
model = SimpleCNN()
torch.save(model, "simple_cnn.pt")

# Load with Hamerspace
optimizer = Optimizer.from_pytorch(
    "simple_cnn.pt",
    input_shape=[1, 3, 32, 32]
)

# Define constraints
constraints = Constraints(
    target_size_mb=2,
    target_hardware="cpu"
)

# Optimize
result = optimizer.optimize(
    goal=OptimizationGoal.QUANTIZE,
    constraints=constraints
)

# View results
print(result.report)

# Save optimized model
result.save_model("simple_cnn_optimized.pt")
```

**Output:**
```
============================================================
Hamerspace Optimization Report
============================================================

Original Model:
  Size: 8.32 MB, Latency: 15.43 ms

Optimized Model:
  Size: 2.08 MB, Latency: 9.21 ms

Improvements:
  Size Reduction: 75.00%
  Latency Improvement: 40.31%

Backend: pytorch
Technique: int8_quantization
Optimization Time: 3.45s

Constraints Satisfied: ✓ Yes
============================================================
```

## Tutorial 2: AUTO Mode with Multiple Constraints

Let AUTO mode figure out the best optimization strategy.

```python
from hamerspace import Optimizer, OptimizationGoal, Constraints

# Load a pretrained model
optimizer = Optimizer.from_pytorch(
    "resnet18.pt",
    input_shape=[1, 3, 224, 224]
)

# Define strict constraints
constraints = Constraints(
    target_size_mb=15,      # Must be under 15MB
    max_latency_ms=30,      # Must run in <30ms
    max_accuracy_drop=0.03, # Max 3% accuracy drop
    target_hardware="cpu"
)

# Let Hamerspace choose the best approach
result = optimizer.optimize(
    goal=OptimizationGoal.AUTO,
    constraints=constraints
)

print(result.report)
print(f"\nSelected technique: {result.config.technique}")
print(f"Backend used: {result.config.backend.value}")
```

**Key Points:**
- AUTO mode analyzes all available strategies
- Selects the one most likely to satisfy constraints
- May use composite strategies (quantization + graph optimization)
- Falls back gracefully if constraints can't be met

## Tutorial 3: Accuracy-Aware Optimization

Include accuracy validation to ensure model quality.

```python
import torch
from torch.utils.data import DataLoader
from hamerspace import Optimizer, OptimizationGoal, Constraints

# Load model
optimizer = Optimizer.from_pytorch(
    "classifier.pt",
    input_shape=[1, 3, 224, 224]
)

# Define validation function
def validate_accuracy(model):
    """Evaluate model accuracy on validation set."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return correct / total

# Define constraints with accuracy requirement
constraints = Constraints(
    target_size_mb=10,
    max_accuracy_drop=0.02,  # Max 2% drop
    target_hardware="cpu"
)

# Optimize with validation
result = optimizer.optimize(
    goal=OptimizationGoal.QUANTIZE,
    constraints=constraints,
    validation_fn=validate_accuracy
)

print(f"Original accuracy: {result.report.original_metrics.accuracy:.4f}")
print(f"Optimized accuracy: {result.report.optimized_metrics.accuracy:.4f}")
print(f"Accuracy drop: {result.report.accuracy_drop:.4f}")
```

## Tutorial 4: Hardware-Specific Optimization

Optimize for specific target hardware.

```python
from hamerspace import Optimizer, OptimizationGoal, Constraints

# Load model
optimizer = Optimizer.from_pytorch(
    "model.pt",
    input_shape=[1, 3, 224, 224]
)

# Optimize for edge device (prefers OpenVINO)
edge_constraints = Constraints(
    target_size_mb=5,
    max_latency_ms=50,
    target_hardware="edge"
)

edge_result = optimizer.optimize(
    goal=OptimizationGoal.AUTO,
    constraints=edge_constraints
)

print(f"Edge optimization: {edge_result.config.backend.value}")
edge_result.save_model("model_edge.onnx")

# Optimize for ARM CPU
arm_constraints = Constraints(
    target_size_mb=8,
    max_latency_ms=40,
    target_hardware="arm"
)

arm_result = optimizer.optimize(
    goal=OptimizationGoal.AUTO,
    constraints=arm_constraints
)

print(f"ARM optimization: {arm_result.config.backend.value}")
arm_result.save_model("model_arm.pt")
```

## Tutorial 5: Benchmarking Only

Sometimes you just want to measure performance.

```python
from hamerspace import Optimizer

# Load model
optimizer = Optimizer.from_pytorch(
    "model.pt",
    input_shape=[1, 3, 224, 224]
)

# Benchmark on different hardware
cpu_metrics = optimizer.benchmark(
    hardware="cpu",
    num_runs=100,
    batch_size=1
)

print("CPU Performance:")
print(f"  Size: {cpu_metrics.size_mb:.2f} MB")
print(f"  Latency: {cpu_metrics.latency_ms:.2f} ms")
print(f"  Throughput: {cpu_metrics.throughput:.2f} inf/s")

# If GPU available
if torch.cuda.is_available():
    gpu_metrics = optimizer.benchmark(
        hardware="cuda",
        num_runs=100,
        batch_size=1
    )
    
    print("\nGPU Performance:")
    print(f"  Latency: {gpu_metrics.latency_ms:.2f} ms")
    print(f"  Throughput: {gpu_metrics.throughput:.2f} inf/s")
```

## Tutorial 6: Using the CLI

Hamerspace includes a command-line interface.

```bash
# Basic quantization
hamerspace optimize model.pt \
    --goal quantize \
    --size 10 \
    --output optimized.pt

# Auto optimization with all constraints
hamerspace optimize model.onnx \
    --goal auto \
    --size 5 \
    --latency 50 \
    --accuracy-drop 0.02 \
    --hardware cpu \
    --output optimized.onnx \
    --report report.txt

# Benchmark
hamerspace benchmark model.pt \
    --hardware cpu \
    --runs 100 \
    --input-shape "1,3,224,224"

# Verbose output
hamerspace optimize model.pt \
    --goal quantize \
    --size 10 \
    --output optimized.pt \
    --verbose
```

## Tutorial 7: Reproducible Optimization

Save and reproduce optimization configurations.

```python
from hamerspace import Optimizer, OptimizationGoal, Constraints

# Initial optimization
optimizer = Optimizer.from_pytorch("model.pt", input_shape=[1, 3, 224, 224])

constraints = Constraints(
    target_size_mb=10,
    max_latency_ms=50,
    target_hardware="cpu"
)

result = optimizer.optimize(
    goal=OptimizationGoal.QUANTIZE,
    constraints=constraints
)

# Save everything
result.save_model("optimized.pt")
result.save_config("config.json")
result.save_report("report.txt")

# Later: Reproduce the same optimization
# (This requires storing more info in the config)
import json

with open("config.json") as f:
    config = json.load(f)
    print(f"Used technique: {config['technique']}")
    print(f"Backend: {config['backend']}")
```

## Tutorial 8: Comparing Multiple Approaches

Try different optimization strategies and compare.

```python
from hamerspace import Optimizer, OptimizationGoal, Constraints

optimizer = Optimizer.from_pytorch("model.pt", input_shape=[1, 3, 224, 224])
constraints = Constraints(target_size_mb=10, target_hardware="cpu")

# Try quantization
quant_result = optimizer.optimize(
    goal=OptimizationGoal.QUANTIZE,
    constraints=constraints
)

# Try pruning
prune_result = optimizer.optimize(
    goal=OptimizationGoal.PRUNE,
    constraints=constraints
)

# Compare results
print("Quantization:")
print(f"  Size: {quant_result.report.optimized_metrics.size_mb:.2f} MB")
print(f"  Latency: {quant_result.report.optimized_metrics.latency_ms:.2f} ms")

print("\nPruning:")
print(f"  Size: {prune_result.report.optimized_metrics.size_mb:.2f} MB")
print(f"  Latency: {prune_result.report.optimized_metrics.latency_ms:.2f} ms")

# Save the better one
if quant_result.report.size_reduction_ratio > prune_result.report.size_reduction_ratio:
    quant_result.save_model("best_optimized.pt")
    print("\nQuantization performed better!")
else:
    prune_result.save_model("best_optimized.pt")
    print("\nPruning performed better!")
```

## Common Patterns

### Pattern 1: Production Deployment

```python
# 1. Optimize with production constraints
result = optimizer.optimize(
    goal=OptimizationGoal.AUTO,
    constraints=Constraints(
        target_size_mb=20,
        max_latency_ms=100,
        max_accuracy_drop=0.01,
        target_hardware="cpu"
    ),
    validation_fn=validate_on_full_test_set
)

# 2. Verify constraints
assert result.report.constraints_satisfied, "Optimization failed!"

# 3. Save for deployment
result.save_model("production_model.onnx")
result.save_config("production_config.json")
result.save_report("production_report.txt")
```

### Pattern 2: Research/Experimentation

```python
# Try multiple configurations
configs = [
    {"target_size_mb": 10, "max_latency_ms": 50},
    {"target_size_mb": 5, "max_latency_ms": 30},
    {"target_size_mb": 3, "max_latency_ms": 20},
]

results = []
for config in configs:
    constraints = Constraints(**config, target_hardware="cpu")
    result = optimizer.optimize(goal=OptimizationGoal.AUTO, constraints=constraints)
    results.append(result)

# Analyze trade-offs
for i, result in enumerate(results):
    print(f"\nConfig {i+1}:")
    print(f"  Size: {result.report.optimized_metrics.size_mb:.2f} MB")
    print(f"  Latency: {result.report.optimized_metrics.latency_ms:.2f} ms")
    print(f"  Satisfied: {result.report.constraints_satisfied}")
```

## Best Practices

1. **Always specify input_shape**: Required for accurate benchmarking
2. **Use validation functions**: Ensure accuracy is maintained
3. **Start with AUTO mode**: Let Hamerspace choose the best approach
4. **Benchmark before optimizing**: Understand baseline performance
5. **Save configurations**: Enable reproducibility
6. **Test on target hardware**: Benchmark on deployment environment
7. **Iterate constraints**: Adjust based on results

## Troubleshooting

### Backend Not Available

```python
from hamerspace.backends.openvino_backend import OpenVINOBackend

backend = OpenVINOBackend()
if not backend.is_available():
    print("OpenVINO not installed. Install with:")
    print("pip install hamerspace[full]")
```

### Constraints Not Satisfied

```python
result = optimizer.optimize(goal=OptimizationGoal.AUTO, constraints=constraints)

if not result.report.constraints_satisfied:
    print("Constraints not met:")
    for violation in result.report.constraint_violations:
        print(f"  - {violation}")
    
    # Try relaxing constraints or different goal
```

### Memory Issues

```python
# Use smaller batch size for benchmarking
constraints = Constraints(
    target_size_mb=10,
    target_hardware="cpu",
    batch_size=1  # Smaller batch
)
```

## Next Steps

- Explore the [API Documentation](API.md)
- Read the [Architecture Guide](ARCHITECTURE.md)
- Check out more [Examples](../examples/)
- Contribute to the project (see [CONTRIBUTING.md](../CONTRIBUTING.md))
