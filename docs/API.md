# Hamerspace API Documentation

## Table of Contents

1. [Core API](#core-api)
2. [Models](#models)
3. [Backends](#backends)
4. [Strategies](#strategies)
5. [Benchmarking](#benchmarking)
6. [Utilities](#utilities)

---

## Core API

### Optimizer

The main entry point for model optimization.

#### `Optimizer(model, model_info, input_shape=None)`

Initialize an optimizer with a loaded model.

**Parameters:**
- `model`: The model object
- `model_info`: ModelInfo object with model metadata
- `input_shape`: Optional input tensor shape

#### Class Methods

##### `Optimizer.from_pytorch(model_path, input_shape=None)`

Load a PyTorch model.

**Parameters:**
- `model_path`: Path to .pt/.pth file or model object
- `input_shape`: Input tensor shape [batch, channels, height, width]

**Returns:** Optimizer instance

**Example:**
```python
optimizer = Optimizer.from_pytorch("resnet18.pt", input_shape=[1, 3, 224, 224])
```

##### `Optimizer.from_tensorflow(model_path, input_shape=None)`

Load a TensorFlow model.

**Parameters:**
- `model_path`: Path to .h5 file or SavedModel directory
- `input_shape`: Input tensor shape

**Returns:** Optimizer instance

##### `Optimizer.from_onnx(model_path)`

Load an ONNX model.

**Parameters:**
- `model_path`: Path to .onnx file

**Returns:** Optimizer instance

#### Instance Methods

##### `optimize(goal, constraints=None, validation_data=None, validation_fn=None, preferred_backends=None)`

Optimize the model.

**Parameters:**
- `goal`: OptimizationGoal enum (QUANTIZE, PRUNE, DISTILL, AUTO)
- `constraints`: Constraints object with optimization constraints
- `validation_data`: Optional data for accuracy validation
- `validation_fn`: Optional function(model) -> accuracy
- `preferred_backends`: Optional list of preferred backends

**Returns:** OptimizationResult

**Example:**
```python
result = optimizer.optimize(
    goal=OptimizationGoal.AUTO,
    constraints=Constraints(
        target_size_mb=10,
        max_latency_ms=50,
        target_hardware="cpu"
    )
)
```

##### `benchmark(hardware="cpu", num_runs=100, batch_size=1)`

Benchmark the model without optimization.

**Parameters:**
- `hardware`: Target hardware (cpu, gpu, etc.)
- `num_runs`: Number of inference runs for averaging
- `batch_size`: Batch size for inference

**Returns:** ModelMetrics

---

## Models

### Constraints

Defines optimization constraints.

```python
Constraints(
    target_size_mb=10.0,         # Target model size in MB
    max_latency_ms=50.0,         # Max inference latency in ms
    max_accuracy_drop=0.02,      # Max accuracy drop (0.0-1.0)
    target_hardware="cpu",       # Target hardware
    batch_size=1,                # Inference batch size
    calibration_samples=100      # Samples for calibration
)
```

**Methods:**
- `has_size_constraint()`: Check if size constraint specified
- `has_latency_constraint()`: Check if latency constraint specified
- `has_accuracy_constraint()`: Check if accuracy constraint specified

### OptimizationGoal

Enum of available optimization goals.

```python
class OptimizationGoal(str, Enum):
    QUANTIZE = "quantize"  # Apply quantization
    PRUNE = "prune"        # Apply pruning
    DISTILL = "distill"    # Knowledge distillation
    AUTO = "auto"          # Automatically select best
```

### Backend

Enum of available backends.

```python
class Backend(str, Enum):
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    ONNX = "onnx"
    ONNX_RUNTIME = "onnx_runtime"
    OPENVINO = "openvino"
    TVM = "tvm"
    OPTIMUM = "optimum"
    BITSANDBYTES = "bitsandbytes"
```

### ModelMetrics

Performance metrics for a model.

```python
class ModelMetrics:
    size_mb: float              # Model size in MB
    latency_ms: float           # Average inference latency in ms
    accuracy: Optional[float]   # Model accuracy if available
    memory_mb: Optional[float]  # Peak memory usage in MB
    throughput: Optional[float] # Inferences per second
```

### OptimizationResult

Result of an optimization operation.

**Attributes:**
- `optimized_model`: The optimized model object
- `report`: CompressionReport with detailed metrics
- `config`: OptimizationConfig used

**Methods:**
- `save_model(path)`: Save the optimized model
- `save_config(path)`: Save configuration for reproducibility
- `save_report(path)`: Save the optimization report

**Example:**
```python
result.save_model("optimized.pt")
result.save_config("config.json")
result.save_report("report.txt")
```

### CompressionReport

Detailed report comparing original and optimized models.

**Attributes:**
- `original_metrics`: ModelMetrics for original model
- `optimized_metrics`: ModelMetrics for optimized model
- `optimization_config`: Configuration used
- `size_reduction_ratio`: Size reduction ratio (0.0-1.0)
- `latency_improvement_ratio`: Latency improvement ratio
- `accuracy_drop`: Accuracy drop if measurable
- `constraints_satisfied`: Whether constraints are satisfied
- `constraint_violations`: List of constraint violations
- `execution_time_seconds`: Optimization time
- `backend_used`: Backend that was used

---

## Backends

### BaseBackend

Abstract base class for optimization backends.

**Key Methods:**
- `is_available()`: Check if backend dependencies are installed
- `supports_model(model_info)`: Check if backend can handle the model
- `quantize(model, ...)`: Apply quantization
- `prune(model, ...)`: Apply pruning
- `optimize_graph(model, ...)`: Apply graph optimizations

### PyTorchBackend

Backend using PyTorch's native optimization tools.

**Capabilities:**
- Dynamic quantization
- Static quantization (with calibration)
- L1 unstructured pruning
- JIT optimization

**Best For:** PyTorch models, CPU inference

### ONNXBackend

Backend using ONNX and ONNX Runtime.

**Capabilities:**
- Dynamic quantization
- Static quantization (with calibration)
- Graph optimization

**Best For:** Framework-agnostic optimization, deployment

### OpenVINOBackend

Backend using Intel OpenVINO.

**Capabilities:**
- NNCF-based quantization
- Graph optimization
- Device-specific optimization

**Best For:** Intel CPUs, edge devices

---

## Strategies

### OptimizationStrategy

Base class for optimization strategies.

**Key Methods:**
- `apply(model, ...)`: Apply the optimization
- `estimate_impact(size, latency)`: Estimate optimization impact

### QuantizationStrategy

Strategy for quantization-based optimization.

Supports INT8 and dynamic quantization.

### PruningStrategy

Strategy for pruning-based optimization.

Removes unnecessary weights to reduce model size.

### GraphOptimizationStrategy

Strategy for graph-level optimizations.

Applies operator fusion and other graph transformations.

### CompositeStrategy

Composes multiple optimization techniques.

Example: Quantization + Graph Optimization

---

## Benchmarking

### Benchmarker

Benchmarks models to measure performance.

```python
benchmarker = Benchmarker()
metrics = benchmarker.benchmark_model(
    model=model,
    model_info=model_info,
    target_hardware="cpu",
    input_shape=[1, 3, 224, 224],
    batch_size=1,
    num_runs=100
)
```

**Returns:** ModelMetrics with size, latency, throughput

---

## Utilities

### Logger

```python
from hamerspace.utils import get_logger, set_log_level
import logging

logger = get_logger(__name__)
set_log_level(logging.DEBUG)  # Enable debug logging
```

---

## Complete Example

```python
from hamerspace import Optimizer, OptimizationGoal, Constraints

# Load model
optimizer = Optimizer.from_pytorch(
    "model.pt",
    input_shape=[1, 3, 224, 224]
)

# Define constraints
constraints = Constraints(
    target_size_mb=10,
    max_latency_ms=50,
    max_accuracy_drop=0.02,
    target_hardware="cpu"
)

# Optimize
result = optimizer.optimize(
    goal=OptimizationGoal.AUTO,
    constraints=constraints
)

# View results
print(result.report)

# Save optimized model
result.save_model("optimized.pt")
```

---

## Error Handling

Hamerspace uses standard Python exceptions:

- `ValueError`: Invalid parameters or constraints
- `FileNotFoundError`: Model file not found
- `RuntimeError`: Optimization or backend errors
- `NotImplementedError`: Unsupported features

**Example:**
```python
try:
    result = optimizer.optimize(goal=OptimizationGoal.AUTO)
except ValueError as e:
    print(f"Invalid parameters: {e}")
except RuntimeError as e:
    print(f"Optimization failed: {e}")
```
