"""
Training Pipeline
=================

Distributed training modules for Qwen2.5-VL fine-tuning.

Modules
-------
train
    Main entry point - orchestrates the complete training pipeline
config
    Configuration system - YAML for training, env vars for infrastructure
trainer
    Custom HuggingFace Trainer with multi-component learning rates
data
    PyTorch Dataset and DataCollator for multimodal training
callbacks
    Ray Train callback for checkpoint and metrics reporting
mlflow_wrapper
    PyFunc wrapper for MLflow model serving
dvc_loader
    Data versioning integration with DVC
log
    Structured JSON logging with Rich console output
"""
