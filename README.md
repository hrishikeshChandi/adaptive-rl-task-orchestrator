# 🚀 Adaptive RL Task Orchestrator

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)
[![MongoDB](https://img.shields.io/badge/MongoDB-7.0-green.svg)](https://www.mongodb.com/)
[![Prometheus](https://img.shields.io/badge/Prometheus-monitoring-orange)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/Grafana-dashboards-blue)](https://grafana.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **distributed task orchestrator** with **Q‑learning based adaptive scheduling**, full observability (Prometheus/Grafana), and a real‑time web dashboard. Built with FastAPI, MongoDB, and Docker.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [RL Scheduling in Depth](#rl-scheduling-in-depth)
- [System Architecture](#system-architecture)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation & Running](#installation--running)
- [API Endpoints](#api-endpoints)
- [Monitoring & Dashboard](#monitoring--dashboard)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Overview

The **Adaptive RL Task Orchestrator** intelligently distributes computational tasks across a fleet of worker nodes. Unlike static schedulers, it uses **Reinforcement Learning (Q‑learning)** to learn which tasks to favour – those needing higher priority or those with shorter duration – based on real‑time worker and task characteristics. Workers self‑register with their CPU/RAM capacity, tasks specify resource requirements, and the RL agent continuously improves scheduling decisions.

**Real‑world analogy:** A smarter Kubernetes scheduler that learns from past assignments to optimise for throughput or latency.

---

## Key Features

### Core Scheduling

| Feature                       | Description                                                         |
| ----------------------------- | ------------------------------------------------------------------- |
| **Worker Registration**       | Workers register with random (or fixed) CPU / RAM capacities.       |
| **Resource‑Aware Assignment** | Tasks only assigned to workers with sufficient free resources.      |
| **Priority Queue**            | Tasks have an integer priority (higher = more urgent).              |
| **Multi‑Task Pull**           | Workers can pull up to 4 tasks in one request, respecting capacity. |

### Reinforcement Learning (Q‑Learning)

| Feature                       | Description                                                                 |
| ----------------------------- | --------------------------------------------------------------------------- |
| **State Discretisation**      | Worker CPU/RAM and task CPU/RAM are binned into 5×5×5×5 = 625 states.       |
| **Two Actions**               | `0` = favour priority, `1` = favour estimated duration.                     |
| **Reward Function**           | `+1` if the assignment respected worker capacity, `-1` otherwise.           |
| **Background Learner**        | Processes completed tasks every 30s, updates Q‑table in MongoDB.            |
| **ε‑Greedy Exploration**      | Starts with ε=0.2, decays over time to exploit learned weights.             |
| **Softmax Weight Extraction** | Q‑values converted to weights (priority weight, duration weight) per state. |

### Fault Tolerance & Resilience

| Feature                   | Description                                                                     |
| ------------------------- | ------------------------------------------------------------------------------- |
| **Heartbeat Monitoring**  | Workers send heartbeats every 5 seconds.                                        |
| **Dead Worker Detection** | Missing 3 heartbeats → worker marked `dead`, tasks are retried.                 |
| **Task Timeout**          | Tasks running longer than 30 seconds are reset and retried (max 3 times).       |
| **Resource Clean‑up**     | Resources automatically freed on completion, failure, timeout, or worker death. |

### Observability & Dashboard

| Component              | Description                                                                               |
| ---------------------- | ----------------------------------------------------------------------------------------- |
| **Prometheus Metrics** | Exposed at `/metrics`: task counters, pending tasks, active workers, RL success rate.     |
| **Grafana Dashboards** | Pre‑configured dashboards for real‑time monitoring.                                       |
| **Web Frontend**       | HTML/JS dashboard to submit tasks, register workers, view RL metrics and Q‑table samples. |

### DevOps Ready

| Feature                  | Description                                                                                              |
| ------------------------ | -------------------------------------------------------------------------------------------------------- |
| **Docker Compose**       | One‑command startup of API, workers, recovery, heartbeat, MongoDB, Prometheus, Grafana, frontend.        |
| **Kubernetes Manifests** | Deployments, Services, ConfigMaps for the whole stack.                                                   |
| **CI/CD Pipeline**       | GitHub Actions: lint, test (pytest + Postman), security scan (Trivy), build & push to GHCR / Docker Hub. |

---

## RL Scheduling in Depth

The RL agent influences task _selection_ at pull time, not resource allocation.

### 1. State Representation

Each (worker, task) pair is discretised into a **4‑tuple**:

- `worker_cpu_bin`: 0 (≤2) … 4 (16+)
- `worker_ram_bin`: 0 (≤4GB) … 4 (32+)
- `task_cpu_bin`: 0…4
- `task_ram_bin`: 0…4

Example: `"3-2-1-1"` means worker has 8‑16 CPU cores & 8‑16GB RAM, task needs 2‑4 CPU cores & 4‑8GB RAM.

### 2. Action & Policy

- Action 0: assign higher weight to the task’s **priority**.
- Action 1: assign higher weight to the **estimated duration** (shorter tasks get higher score).
- The agent uses **ε‑greedy** (explore 20% initially) to balance exploration vs exploitation.

### 3. Reward

After a task finishes, the learner checks if the worker had enough resources for it.

- **Scheduling valid** → reward +1 (the action was appropriate).
- **Invalid** (worker overloaded) → reward -1.

### 4. Q‑Learning Update

Each task outcome triggers an off‑policy update:

`Q(s,a) ← Q(s,a) + α [ r + γ · maxₐ' Q(s',a') - Q(s,a) ]`

- α = 0.1 (learning rate), γ = 0.9 (discount factor).
- Q‑table is stored in MongoDB collection `q_table`.

### 5. Weight Extraction for Scoring

When a worker asks for a task, the agent computes:

`w_priority = e^Q(s,0) / (e^Q(s,0) + e^Q(s,1))`

`w_duration = e^Q(s,1) / (e^Q(s,0) + e^Q(s,1))`

The final score for a candidate task is:

`score = w_priority × (priority / 5) + w_duration × (1 / max(0.1, est_duration))`

The task with the highest score is atomically assigned.

---

## System Architecture

The system consists of:

- FastAPI Control Plane — Central orchestrator handling task submission, worker registration, heartbeat processing, and RL‑aware task assignment. Exposes REST endpoints with rate limiting and Prometheus metrics.

- MongoDB (Standalone) — Stores tasks, worker state, task history for RL training, and the Q‑table for adaptive scheduling. No replica set required, simplifying deployment.

- Worker Nodes — Self‑register with CPU/RAM capacity, run independent heartbeat threads, and pull tasks from the API using an RL‑scored priority queue. Each worker can execute multiple tasks concurrently respecting local capacity limits.

- Background Services Trio — Recovery daemon (resets timed‑out tasks), heartbeat monitor (detects dead workers and frees resources), and RL background learner (updates Q‑table every 30 seconds from completed task history).

<!-- ![System Architecture](diagrams/system-architecture.png) -->

---

## How It Works

### 1. Worker Lifecycle

- Worker calls `POST /workers/add_worker` with its CPU/RAM → gets a unique `worker_id`.
- It starts a background heartbeat thread (every 5s) using `PUT /workers/heartbeat/{worker_id}`.
- Heartbeat monitor marks workers `dead` after 30s of silence and frees their resources.

### 2. Task Submission

- Client sends `POST /tasks/submit` with `required_cpu`, `required_ram`, `priority`, and custom `data`.
- System checks if _any_ worker exists with enough total capacity; if not, rejects the task.
- Task is stored with state `pending`.

### 3. Task Assignment (Worker Pull)

- Worker calls `GET /tasks/get_task?worker_id=...`.
- The API:
  1. Loads worker’s current available resources.
  2. Queries pending tasks that fit available resources.
  3. Scores each candidate using the **RL weights** (or falls back to priority only if RL is disabled).
  4. Atomically updates the chosen task to `running`, deducts resources from the worker.
  5. Returns the task(s) to the worker (up to 4 tasks per pull).

### 4. Execution & Reporting

- Worker executes the task (simulated with `sleep` + random failure).
- Calls `PUT /tasks/update_status/{task_id}` with `"completed"` or `"failed"`.
- On finalisation, resources are added back to the worker and the task’s outcome is stored in `task_history`.

### 5. Background Learning

- The `RLBackgroundLearner` runs every 30 seconds, fetches unprocessed history entries.
- For each entry: computes reward based on `scheduling_valid`, updates Q‑table, and decays ε.
- Updated Q‑table is persisted to MongoDB.

---

## Tech Stack

| Component                | Technology                                      |
| ------------------------ | ----------------------------------------------- |
| **Backend Framework**    | FastAPI                                         |
| **Database**             | MongoDB 7.0 (standalone, no replica set needed) |
| **RL & Numerics**        | Python + NumPy                                  |
| **Worker Communication** | HTTP REST + `requests`                          |
| **Monitoring**           | Prometheus + Grafana + `prometheus_client`      |
| **Frontend**             | HTML5 / CSS / JavaScript (static)               |
| **Containerisation**     | Docker + Docker Compose                         |
| **Orchestration**        | Kubernetes (manifests provided)                 |
| **CI/CD**                | GitHub Actions (lint, test, security, publish)  |

---

## Prerequisites

- **Python 3.12+** (only if running manually)
- **Docker** and **Docker Compose** (recommended)
- **Git**
- (Optional) **kubectl** + **minikube** / K8s cluster

---

## Installation & Running

### 1. Clone the Repository

```bash
git clone https://github.com/hrishikeshChandi/adaptive-rl-task-orchestrator
cd adaptive-rl-task-orchestrator
```

### 2. Run with Docker Compose (Easiest)

```bash
docker compose up -d
```

This starts:

- MongoDB on `:27017`
- FastAPI server on `:8000`
- 1 worker (auto‑registered, pulls tasks)
- Recovery daemon
- Heartbeat monitor
- Prometheus on `:9090`
- Grafana on `:3000` (admin/admin)
- Frontend dashboard on `:80`

> 💡 To run more workers, scale the `worker` service:  
> `docker compose up -d --scale worker=3`

### 3. Manual (without Docker)

#### MongoDB (standalone, no replica set needed)

```bash
mongod --dbpath /data/db
```

#### Start the API Server

```bash
uv run main.py
```

#### Start Background Services

```bash
uv run -m recovery.recovery
uv run -m workers.heartbeat
```

#### Start Workers (run multiple)

```bash
uv run -m workers.worker      # registers new worker
uv run -m workers.worker      # second worker, etc.
```

> Set `DISABLE_WORKER_SPAWN=true` to prevent the API from auto‑spawning worker sub‑processes when adding via frontend.

---

## API Endpoints

### Tasks

| Method | Endpoint                         | Description                      | Rate Limit |
| ------ | -------------------------------- | -------------------------------- | ---------- |
| `POST` | `/tasks/submit`                  | Submit a new task                | 20/min     |
| `GET`  | `/tasks/get_task`                | Worker pulls task(s) (RL‑aware)  | 20/sec     |
| `GET`  | `/tasks/tasks`                   | List only pending tasks          | unlimited  |
| `GET`  | `/tasks/all_tasks`               | List last 100 tasks (any status) | unlimited  |
| `PUT`  | `/tasks/update_status/{task_id}` | Mark as completed / failed       | 20/min     |

### Workers

| Method | Endpoint                         | Description             | Rate Limit |
| ------ | -------------------------------- | ----------------------- | ---------- |
| `POST` | `/workers/add_worker`            | Register a new worker   | 20/min     |
| `PUT`  | `/workers/heartbeat/{worker_id}` | Update worker heartbeat | 20/min     |
| `GET`  | `/workers/get_workers`           | List all workers        | 10/min     |
| `GET`  | `/workers/worker/{worker_id}`    | Get a specific worker   | 10/min     |

### Learning & RL

| Method | Endpoint               | Description                                |
| ------ | ---------------------- | ------------------------------------------ |
| `POST` | `/learning/update`     | (Internal) report task outcome to RL       |
| `GET`  | `/learning/metrics/rl` | Get Q‑table samples, success rate, updates |

### Metrics (Prometheus)

| Method | Endpoint   | Description                          |
| ------ | ---------- | ------------------------------------ |
| `GET`  | `/metrics` | Prometheus metrics (counter, gauges) |

### Health

| Method | Endpoint  | Description   |
| ------ | --------- | ------------- |
| `GET`  | `/health` | Simple health |

---

## Monitoring & Dashboard

### Web Dashboard

Open `http://localhost:80` (or the NodePort if on Kubernetes).  
From there you can:

- Submit new tasks (set CPU/RAM, priority, JSON payload)
- Register workers manually
- View active workers and their resource usage
- See tasks filtered by status / worker
- Inspect RL metrics (top Q‑table entries, global success rate)

### Prometheus

- Targets: `http://localhost:9090`
- Automatically scrapes `/metrics` from the API every 15s.
- Metrics include:
  - `tasks_submitted_total`, `tasks_completed_total`, `tasks_failed_total`
  - `pending_tasks`, `active_workers`
  - `rl_success_rate`

### Grafana

- Login at `http://localhost:3000` (admin/admin)
- Pre‑provisioned dashboard shows task throughput, worker health, RL success rate.

---

## Testing

### Run Pytest (unit + integration)

```bash
pytest tests/ -v --cov=. --cov-report=term
```

### Run Postman Collection (Newman)

```bash
docker compose up -d
npm install -g newman
newman run tests/postman-collection.json
```

### Manual Test with cURL

**Submit a task:**

```bash
curl -X POST http://localhost:8000/tasks/submit \
  -H "Content-Type: application/json" \
  -d '{"data": {"cmd": "sleep"}, "required_cpu": 2, "required_ram": 4, "priority": 3}'
```

**Register a worker:**

```bash
curl -X POST http://localhost:8000/workers/add_worker \
  -H "Content-Type: application/json" \
  -d '{"cpu_cores": 8, "ram": 16}'
```

**List workers:**

```bash
curl http://localhost:8000/workers/get_workers
```

**Get RL metrics:**

```bash
curl http://localhost:8000/learning/metrics/rl | jq
```

---

## Project Structure

```
orchestrator/
├── config/
│   ├── __init__.py
│   └── constants.py           # Heartbeat, RL, MongoDB parameters
├── core/
│   ├── __init__.py
│   ├── limiter.py             # SlowAPI rate limiting
│   └── metrics.py             # Prometheus metrics definitions
├── db/
│   ├── __init__.py
│   └── connection.py          # MongoDB client, indexes, retry logic
├── models/
│   ├── __init__.py
│   └── model.py               # Pydantic schemas (Task, Worker, etc.)
├── routers/
│   ├── __init__.py
│   ├── task.py                # Task endpoints (RL‑aware assignment)
│   ├── workers.py             # Worker registration & heartbeat
│   └── learning.py            # RL metrics & update endpoint
├── scheduling/
│   ├── __init__.py
│   ├── rl_agent.py            # QLearningAgent, discretisation, Q‑update
│   ├── learner.py             # Background learner thread
│   └── estimator.py           # Duration estimator (bin averaging)
├── workers/
│   ├── __init__.py
│   ├── worker.py              # Worker runtime (pull, execute, heartbeat)
│   └── heartbeat.py           # Dead‑worker monitor & resource freeing
├── recovery/
│   ├── __init__.py
│   └── recovery.py            # Timeout task recovery daemon
├── frontend/
│   └── index.html             # Interactive dashboard
├── k8s/                       # Kubernetes manifests
│   ├── configmap.yaml
│   ├── deployment.yaml
│   └── service.yaml
├── grafana/                   # Pre‑configured dashboards & datasources
├── prometheus.yml
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.worker
├── Dockerfile.recovery
├── Dockerfile.heartbeat
├── Dockerfile.frontend
├── main.py
└── README.md                  # This file
```

---

## Configuration

Environment variables (can be set in `.env` or passed to containers):

| Variable               | Default                    | Description                                      |
| ---------------------- | -------------------------- | ------------------------------------------------ |
| `MONGO_URI`            | `mongodb://mongodb:27017/` | MongoDB connection string (standalone)           |
| `HOST`                 | `0.0.0.0` (or `localhost`) | API host                                         |
| `PORT`                 | `8000`                     | API port                                         |
| `TIMEOUT`              | `30`                       | Task timeout (seconds)                           |
| `MAX_RETRIES`          | `3`                        | Max retries per task                             |
| `HEARTBEAT_INTERVAL`   | `5`                        | Heartbeat interval (seconds)                     |
| `HEARTBEAT_TIMEOUT`    | `30`                       | Seconds without heartbeat → worker dead          |
| `RL_ENABLED`           | `true`                     | Enable RL scoring (otherwise uses priority only) |
| `RL_EPSILON`           | `0.2`                      | Initial exploration rate                         |
| `RL_EPSILON_DECAY`     | `0.995`                    | Multiplicative decay per learning step           |
| `RL_ALPHA`             | `0.1`                      | Q‑learning learning rate                         |
| `RL_GAMMA`             | `0.9`                      | Discount factor                                  |
| `DISABLE_WORKER_SPAWN` | `false`                    | If `true`, API won’t spawn worker subprocesses   |

All RL constants are defined in `config/constants.py`.

---

## Troubleshooting

| Issue                                  | Solution                                                                                               |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **MongoDB transaction errors**         | Not applicable – this orchestrator uses standalone MongoDB (no replica set needed).                    |
| **Workers never get tasks**            | Check if there are pending tasks that fit a worker’s _available_ resources.                            |
| **RL not improving**                   | Ensure `RL_ENABLED=true` and task history is being written (check `task_history` collection).          |
| **Rate limit (429 Too Many Requests)** | Wait a few seconds or adjust limits in `routers/task.py` / `workers.py`.                               |
| **Frontend dashboard shows no data**   | Verify API is reachable at `http://localhost:8000`. Use browser dev tools to see CORS/fetch errors.    |
| **Prometheus not scraping**            | Ensure `api` container is running and `/metrics` returns data. Check `prometheus.yml` target.          |
| **Worker fails to register**           | If using `DISABLE_WORKER_SPAWN=true`, you must manually start workers with `python -m workers.worker`. |

---

## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.
