# Fraud Detection ML System

This project implements a production-style machine learning pipeline for detecting fraudulent financial transactions.

The goal of the project is to simulate a real-world ML engineering workflow, including model training, experiment tracking, model serving, and system monitoring.

## Project Objectives

- Train a machine learning model to detect fraudulent transactions
- Track experiments using MLflow
- Serve predictions through a FastAPI inference service
- Monitor system performance using Prometheus and Grafana
- Simulate production traffic for testing

## System Architecture

Training Environment
→ Model Training
→ MLflow Experiment Tracking

Inference Service
→ FastAPI API
→ Fraud Prediction Endpoint

Monitoring Stack
→ Prometheus metrics collection
→ Grafana dashboards

## Project Structure

data/  
Raw and processed datasets

notebooks/  
Exploratory analysis and experimentation

src/  
Core project code for data processing, modeling, and monitoring

scripts/  
Command line scripts for training and testing workflows

api/  
Inference API service

monitoring/  
Prometheus and Grafana configuration

infra/  
Infrastructure and deployment configuration

tests/  
Automated tests

docs/  
Architecture documentation and runbooks

## Technologies Used

- Python  
- Scikit-Learn  
- FastAPI  
- MLflow  
- Prometheus  
- Grafana  
- Docker  
- Proxmox Homelab Infrastructure
