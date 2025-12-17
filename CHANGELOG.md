# Changelog

All notable changes to Hamerspace will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-01-XX

### Added
- Initial release of Hamerspace
- Core optimizer API with `Optimizer` class
- Support for PyTorch, TensorFlow, and ONNX models
- Multiple optimization backends:
  - PyTorch (quantization, pruning, JIT optimization)
  - ONNX Runtime (quantization, graph optimization)
  - OpenVINO (Intel hardware optimization)
- Optimization strategies:
  - Quantization (INT8, dynamic)
  - Pruning (L1 unstructured)
  - Graph optimization
  - Composite strategies
- Automatic strategy selection (AUTO mode)
- Comprehensive benchmarking system
- Constraint-based optimization:
  - Target model size
  - Max latency
  - Max accuracy drop
  - Target hardware
- Detailed optimization reports
- Model size, latency, and memory measurements
- Reproducible optimization configs
- Examples and documentation

### Features
- Modular backend architecture
- Hardware-aware optimization
- Deterministic results
- Framework-agnostic design
- Comprehensive logging

### Documentation
- README with quickstart guide
- API documentation
- Example scripts
- Contributing guidelines

## [0.0.1] - 2025-01-XX (Development)
- Project initialization
- Basic structure setup
