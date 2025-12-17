# Hamerspace Architecture

## Overview

Hamerspace is a model compression orchestration engine that acts as a control layer over existing open-source optimization toolkits. This document describes the architecture, design decisions, and implementation details.

## Design Principles

1. **Orchestration, not re-implementation**: Leverage existing tools rather than reimplementing algorithms
2. **Modular backend system**: Easy to add new optimization toolkits
3. **Constraint-driven optimization**: Users specify what they want, system figures out how
4. **Hardware-aware**: Optimize for specific target hardware
5. **Reproducible**: Deterministic results with saved configurations
6. **Framework-agnostic**: Support PyTorch, TensorFlow, ONNX, and more

## Architecture Layers

```
┌─────────────────────────────────────────────────────┐
│                   Public API Layer                   │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │  Optimizer   │  │ Constraints  │  │  Results  │ │
│  └──────────────┘  └──────────────┘  └───────────┘ │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│             Orchestration Layer                      │
│  ┌──────────────────┐  ┌─────────────────────────┐ │
│  │ StrategySelector │  │  Strategy Composition   │ │
│  └──────────────────┘  └─────────────────────────┘ │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│                Backend Layer                         │
│  ┌──────────┬──────────┬──────────┬──────────┐     │
│  │ PyTorch  │   ONNX   │ OpenVINO │   TVM    │     │
│  │ Backend  │ Backend  │ Backend  │ Backend  │     │
│  └──────────┴──────────┴──────────┴──────────┘     │
└─────────────────────────────────────────────────────┘
```

## Core Components

### 1. Public API Layer

**Optimizer Class**
- Main entry point for users
- Factory methods for loading models from different frameworks
- Orchestrates the optimization pipeline
- Handles benchmarking and result generation

**Constraints Model**
- Pydantic model for type-safe constraints
- Validates user inputs
- Provides helper methods for constraint checking

**OptimizationResult**
- Encapsulates optimization results
- Provides methods for saving models and configurations
- Contains detailed compression report

### 2. Orchestration Layer

**StrategySelector**
- Analyzes model and constraints
- Ranks available backends
- Selects optimal optimization strategy
- Handles AUTO mode logic

**Optimization Strategies**
- `QuantizationStrategy`: INT8/INT4 quantization
- `PruningStrategy`: Weight pruning
- `GraphOptimizationStrategy`: Graph-level transformations
- `CompositeStrategy`: Combines multiple techniques

Strategy selection algorithm:
```python
1. Filter backends by model compatibility
2. Filter backends by constraint satisfaction capability
3. Generate candidate strategies
4. Estimate impact of each strategy
5. Rank by:
   - Constraint satisfaction (highest priority)
   - Expected compression ratio
   - Latency improvement
   - Backend reliability
6. Select top-ranked strategy
```

### 3. Backend Layer

**BaseBackend Interface**
All backends implement:
- `is_available()`: Check dependencies
- `supports_model()`: Model compatibility
- `can_satisfy_constraints()`: Constraint feasibility
- `quantize()`: Quantization implementation
- `prune()`: Pruning implementation
- `optimize_graph()`: Graph optimization
- `save_model()`: Model serialization
- `load_model()`: Model deserialization

**PyTorchBackend**
- Uses `torch.quantization` for quantization
- Uses `torch.nn.utils.prune` for pruning
- Uses `torch.jit` for graph optimization
- Best for: PyTorch models, CPU inference

**ONNXBackend**
- Uses ONNX Runtime quantization APIs
- Framework-agnostic through ONNX conversion
- Graph optimization via ONNX optimizer
- Best for: Cross-framework deployment

**OpenVINOBackend**
- Uses NNCF for quantization
- Intel-optimized graph transformations
- Device-specific compilation
- Best for: Intel CPUs, edge devices

### 4. Supporting Components

**Benchmarker**
- Measures model size, latency, memory
- Framework-specific benchmarking
- Hardware-aware measurements
- Statistical averaging over multiple runs

**ModelLoader**
- Loads models from different frameworks
- Extracts model metadata
- Normalizes model representation

**Logger**
- Structured logging throughout pipeline
- Configurable log levels
- Helps with debugging and monitoring

## Data Flow

### Optimization Pipeline

```
1. User creates Optimizer with model
   ↓
2. User calls optimize() with goal and constraints
   ↓
3. Optimizer benchmarks original model
   ↓
4. StrategySelector analyzes constraints and model
   ↓
5. StrategySelector generates candidate strategies
   ↓
6. StrategySelector ranks and selects best strategy
   ↓
7. Selected strategy applies optimization
   ↓
8. Optimizer benchmarks optimized model
   ↓
9. Optimizer checks constraint satisfaction
   ↓
10. Optimizer generates compression report
    ↓
11. OptimizationResult returned to user
```

### Backend Selection Algorithm

```python
def select_backend(model_info, constraints):
    candidates = []
    
    for backend in available_backends:
        # Filter by compatibility
        if not backend.supports_model(model_info):
            continue
        
        # Filter by constraint feasibility
        if not backend.can_satisfy_constraints(constraints, model_info):
            continue
        
        candidates.append(backend)
    
    # Prefer backends by hardware target
    if constraints.target_hardware == "cpu":
        priority = [PyTorch, ONNX, OpenVINO]
    elif constraints.target_hardware == "edge":
        priority = [OpenVINO, ONNX, PyTorch]
    
    return select_by_priority(candidates, priority)
```

## Key Design Decisions

### 1. Why Orchestration vs Implementation?

**Decision**: Use existing tools rather than implement algorithms

**Rationale**:
- Proven implementations from experts
- Maintained by specialized teams
- Hardware optimizations already done
- Faster development and testing
- Focus on integration and usability

### 2. Why Constraint-Driven?

**Decision**: Users specify constraints, system figures out how

**Rationale**:
- More intuitive than requiring optimization expertise
- Allows automatic technique selection
- Enables composite strategies
- Better matches real-world deployment requirements

### 3. Why Separate Strategy Layer?

**Decision**: Strategy layer between API and backends

**Rationale**:
- Enables composing multiple techniques
- Separates "what to do" from "how to do it"
- Makes backend swapping transparent
- Allows for sophisticated selection logic

### 4. Why Pydantic for Models?

**Decision**: Use Pydantic for data models

**Rationale**:
- Type safety and validation
- Automatic documentation
- Serialization support
- IDE autocomplete

## Extension Points

### Adding a New Backend

1. Create new file in `src/hamerspace/backends/`
2. Inherit from `BaseBackend`
3. Implement required methods
4. Add to `StrategySelector` backend list
5. Write tests

Example:
```python
class TensorRTBackend(BaseBackend):
    def __init__(self):
        super().__init__("TensorRT")
    
    def is_available(self) -> bool:
        try:
            import tensorrt
            return True
        except ImportError:
            return False
    
    # Implement other methods...
```

### Adding a New Optimization Technique

1. Create new strategy class in `src/hamerspace/strategies/`
2. Inherit from `OptimizationStrategy`
3. Implement `apply()` and `estimate_impact()`
4. Add to `StrategySelector` strategy generation
5. Add to `OptimizationGoal` enum if needed

### Adding a New Metric

1. Add field to `ModelMetrics` in `core/models.py`
2. Update `Benchmarker` to measure it
3. Update `CompressionReport` if needed

## Performance Considerations

### Benchmarking Overhead

- Warmup runs before measurement
- Statistical averaging over multiple runs
- GPU synchronization when applicable
- Memory measurement with psutil

### Optimization Time

- Strategy selection is fast (< 1 second)
- Optimization time depends on backend and model size
- Benchmarking can be slow for accurate measurements
- Consider caching for repeated optimizations

### Memory Usage

- Original model kept in memory during optimization
- Optimized model created separately
- Temporary files used for format conversions
- Cleaned up after optimization

## Testing Strategy

### Unit Tests
- Individual component testing
- Mock backends for isolation
- Constraint validation
- Model loading

### Integration Tests
- Full optimization pipeline
- Multiple backends
- Real models (small sizes)
- Constraint satisfaction

### Backend Tests
- Backend availability
- Optimization correctness
- Model compatibility

## Future Enhancements

### Planned Features
1. Knowledge distillation support (requires training loop)
2. TVM backend integration
3. More quantization schemes (INT4, mixed precision)
4. Structural pruning
5. Neural architecture search
6. Multi-objective optimization
7. Distributed optimization

### Potential Improvements
1. Caching of optimization results
2. Progressive optimization (iterative refinement)
3. Visualization of compression trade-offs
4. Model zoo with pre-optimized models
5. Cloud-based optimization service
6. AutoML for hyperparameter tuning

## References

- PyTorch Quantization: https://pytorch.org/docs/stable/quantization.html
- ONNX Runtime Quantization: https://onnxruntime.ai/docs/performance/quantization.html
- OpenVINO NNCF: https://github.com/openvinotoolkit/nncf
- Apache TVM: https://tvm.apache.org/
