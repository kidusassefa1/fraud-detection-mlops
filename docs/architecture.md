# System Architecture

This system simulates a production ML pipeline.

Components:

Training VM
- data processing
- model training
- MLflow experiment tracking

Inference LXC
- FastAPI fraud detection API
- model loading
- prediction endpoints
- Prometheus metrics

Monitoring LXC
- Prometheus metrics scraping
- Grafana dashboards

The system allows simulated transaction traffic to be processed by the API and monitored through Grafana dashboards.