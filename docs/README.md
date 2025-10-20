# LapXcel: Advanced Sim Racing Telemetry Analytics Platform

[![CI/CD Pipeline](https://github.com/LapXcel/LapXcel/actions/workflows/ci.yml/badge.svg)](https://github.com/LapXcel/LapXcel/actions/workflows/ci.yml)
[![Backend Coverage](https://codecov.io/gh/LapXcel/LapXcel/branch/main/graph/badge.svg?flag=backend)](https://codecov.io/gh/LapXcel/LapXcel)
[![Frontend Coverage](https://codecov.io/gh/LapXcel/LapXcel/branch/main/graph/badge.svg?flag=frontend)](https://codecov.io/gh/LapXcel/LapXcel)
[![Code Quality](https://sonarcloud.io/api/project_badges/measure?project=LapXcel_LapXcel&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=LapXcel_LapXcel)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

LapXcel is a cutting-edge telemetry analytics platform designed to optimize sim racing performance through advanced machine learning and real-time data analysis. Built for professional e-sports teams and racing enthusiasts, LapXcel provides comprehensive insights, performance optimization, and AI-driven coaching to help drivers achieve their theoretical best lap times.

## 🏆 Key Features

### 🚗 Real-Time Telemetry Analysis
- **Live Data Streaming**: Real-time telemetry collection from Assetto Corsa
- **High-Frequency Data Processing**: Handle 100+ Hz data streams with minimal latency
- **Multi-Track Support**: Comprehensive analysis across various racing circuits
- **Vehicle Dynamics Monitoring**: Track speed, acceleration, braking, steering, and more

### 🤖 AI-Powered Performance Optimization
- **Reinforcement Learning Models**: SAC, TQC, and PPO algorithms for optimal lap time prediction
- **Hyperparameter Optimization**: Automated model tuning with Optuna
- **Theoretical Best Calculation**: AI-driven computation of optimal lap times
- **Adaptive Learning**: Continuous improvement based on driving patterns

### 📊 Professional Analytics Dashboard
- **Interactive Visualizations**: Real-time charts and performance metrics
- **Lap Comparison Tools**: Detailed sector-by-sector analysis
- **Performance Insights**: Actionable recommendations for improvement
- **Export Capabilities**: Data export in CSV, JSON, and compressed formats

### 🔬 Advanced Performance Metrics
- **Consistency Scoring**: Measure driving consistency across sessions
- **Efficiency Analysis**: Fuel consumption and tire wear optimization
- **Smoothness Evaluation**: Steering, throttle, and brake input analysis
- **Competitive Benchmarking**: Compare performance against global leaderboards

## 🏗️ System Architecture

```mermaid
graph TB
    AC[Assetto Corsa] --> ACRL[ACRL Plugin]
    ACRL --> API[FastAPI Backend]
    API --> DB[(PostgreSQL)]
    API --> Redis[(Redis Cache)]
    API --> ML[ML Training Service]
    ML --> Models[Model Storage]
    API --> WS[WebSocket Manager]
    WS --> Frontend[React Dashboard]
    Frontend --> User[Racing Driver]
```

### Backend Services
- **FastAPI**: High-performance async API server
- **PostgreSQL**: Primary database for telemetry and user data
- **Redis**: Caching and real-time data streaming
- **MLflow**: ML experiment tracking and model management
- **Celery**: Distributed task processing for ML training

### Frontend Application
- **React 18**: Modern UI with TypeScript
- **Material-UI**: Professional component library
- **Chart.js & D3.js**: Advanced data visualization
- **WebSocket**: Real-time data streaming
- **Redux Toolkit**: State management

### ML Pipeline
- **Stable Baselines3**: Reinforcement learning algorithms
- **Optuna**: Hyperparameter optimization
- **JAX/Flax**: High-performance neural networks
- **TensorBoard**: Training visualization and monitoring

## 🚀 Quick Start

### Prerequisites
- **Python 3.11+**
- **Node.js 18+**
- **PostgreSQL 15+**
- **Redis 7+**
- **Docker & Docker Compose** (optional)

### Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/LapXcel/LapXcel.git
   cd LapXcel
   ```

2. **Backend Setup**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   
   # Set up environment variables
   cp .env.example .env
   # Edit .env with your database and Redis configurations
   
   # Run database migrations
   alembic upgrade head
   
   # Start the backend server
   uvicorn app.main:app --reload
   ```

3. **Frontend Setup**
   ```bash
   cd frontend/my-dashboard
   npm install
   npm run dev
   ```

4. **Assetto Corsa Integration**
   ```bash
   # Copy ACRL plugin to Assetto Corsa
   cp -r ACRL/ "C:/Program Files (x86)/Steam/steamapps/common/assettocorsa/apps/python/"
   
   # Enable the ACRL app in Assetto Corsa settings
   ```

### Docker Deployment

```bash
# Start all services with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 📖 Documentation

### API Documentation
- **Interactive API Docs**: http://localhost:8000/api/docs
- **OpenAPI Spec**: http://localhost:8000/openapi.json
- **Redoc**: http://localhost:8000/api/redoc

### User Guides
- [Getting Started Guide](docs/user-guide/getting-started.md)
- [Dashboard Tutorial](docs/user-guide/dashboard-tutorial.md)
- [Telemetry Analysis](docs/user-guide/telemetry-analysis.md)
- [AI Training Guide](docs/user-guide/ai-training.md)

### Developer Documentation
- [API Reference](docs/api/README.md)
- [Architecture Overview](docs/architecture/README.md)
- [Database Schema](docs/database/schema.md)
- [ML Pipeline](docs/ml/pipeline.md)
- [Deployment Guide](docs/deployment/README.md)

## 🧪 Testing

### Backend Tests
```bash
cd backend
pytest tests/ -v --cov=app --cov-report=html
```

### Frontend Tests
```bash
cd frontend/my-dashboard
npm test
npm run test:coverage
```

### Integration Tests
```bash
# Run full test suite
npm run test:e2e
```

### Performance Tests
```bash
cd backend
locust -f tests/performance/locustfile.py --host=http://localhost:8000
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://user:pass@localhost/lapxcel` |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379/0` |
| `SECRET_KEY` | JWT secret key | `your-secret-key` |
| `ENVIRONMENT` | Deployment environment | `development` |
| `MODEL_STORAGE_PATH` | ML model storage directory | `./models` |
| `MAX_TRAINING_JOBS` | Concurrent training jobs | `3` |

### Database Configuration
```python
# Database settings
DATABASE_POOL_SIZE = 10
DATABASE_MAX_OVERFLOW = 20
TELEMETRY_RETENTION_DAYS = 90
```

### ML Configuration
```python
# Training settings
DEFAULT_ALGORITHM = "SAC"
DEFAULT_TOTAL_STEPS = 1000000
HYPEROPT_TRIALS = 100
```

## 📊 Performance Metrics

### System Performance
- **API Response Time**: < 100ms (95th percentile)
- **Telemetry Ingestion**: 10,000+ data points/second
- **WebSocket Latency**: < 50ms
- **Database Query Time**: < 10ms average

### ML Performance
- **Training Convergence**: 500K-1M steps typical
- **Model Inference**: < 5ms per prediction
- **Hyperparameter Optimization**: 24-48 hours for full sweep
- **Lap Time Improvement**: 2-5 seconds average improvement

### Test Coverage
- **Backend Coverage**: 95%+ statement coverage
- **Frontend Coverage**: 85%+ line coverage
- **Integration Tests**: 150+ test scenarios
- **E2E Tests**: 50+ user journeys

## 🛡️ Security

### Authentication & Authorization
- **JWT Tokens**: Secure API authentication
- **Role-Based Access**: User/Admin/Premium tiers
- **API Rate Limiting**: Prevent abuse and DoS attacks
- **Input Validation**: Comprehensive request validation

### Data Protection
- **Encryption**: TLS 1.3 for data in transit
- **Database Security**: Encrypted connections and backups
- **Secrets Management**: Environment-based configuration
- **Privacy Compliance**: GDPR-ready data handling

### Infrastructure Security
- **Container Security**: Non-root users, minimal images
- **Network Security**: Firewall rules and VPC isolation
- **Vulnerability Scanning**: Automated security checks
- **Audit Logging**: Comprehensive access logging

## 🚀 Deployment

### Production Deployment
```bash
# Build and push Docker images
docker build -t lapxcel/backend:latest backend/
docker build -t lapxcel/frontend:latest frontend/my-dashboard/

# Deploy with Kubernetes
kubectl apply -f k8s/

# Or deploy with Docker Swarm
docker stack deploy -c docker-compose.prod.yml lapxcel
```

### Cloud Deployment
- **AWS**: ECS/EKS with RDS and ElastiCache
- **GCP**: GKE with Cloud SQL and Memorystore
- **Azure**: AKS with Azure Database and Redis Cache

### Monitoring & Observability
- **Metrics**: Prometheus + Grafana
- **Logging**: ELK Stack (Elasticsearch, Logstash, Kibana)
- **Tracing**: Jaeger distributed tracing
- **Alerting**: PagerDuty integration

## 🤝 Contributing

We welcome contributions from the racing and developer communities!

### Development Process
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`npm test` and `pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Code Standards
- **Python**: Black formatting, flake8 linting, mypy type checking
- **TypeScript**: ESLint + Prettier, strict type checking
- **Commit Messages**: Conventional Commits format
- **Documentation**: Update docs for all new features

### Issue Reporting
- Use GitHub Issues for bug reports and feature requests
- Include detailed reproduction steps for bugs
- Provide system information and logs when relevant

## 📋 Roadmap

### Q1 2024
- [ ] **Multi-Sim Support**: iRacing and rFactor 2 integration
- [ ] **Advanced Analytics**: Machine learning-based driving style analysis
- [ ] **Mobile App**: iOS/Android companion app
- [ ] **Cloud Deployment**: Fully managed SaaS offering

### Q2 2024
- [ ] **Team Features**: Multi-user team management and collaboration
- [ ] **Race Strategy**: AI-powered race strategy optimization
- [ ] **VR Integration**: Virtual reality telemetry visualization
- [ ] **API Marketplace**: Third-party plugin ecosystem

### Q3 2024
- [ ] **Professional Tools**: Advanced setup optimization and car development
- [ ] **Esports Integration**: Tournament and championship management
- [ ] **AI Coaching**: Personalized AI driving coach
- [ ] **Performance Prediction**: Race outcome prediction models

## 👥 Team

### Core Development Team
- **Samuel Braun** ([@samuelbraun](https://github.com/samuelbraun)) - Lead Developer & ML Engineer
- **Colby Todd** ([@colbytodd](https://github.com/colbytodd)) - Backend Developer & DevOps
- **Engy Elsayed** ([@engyelsayed](https://github.com/engyelsayed)) - Frontend Developer & UX
- **Sarah Siage** ([@sarahsiage](https://github.com/sarahsiage)) - Full-Stack Developer & QA

### Special Thanks
- **Alfred Genadri** - Product Owner & Racing Expert
- **University of Ottawa** - Academic Support and Resources
- **Assetto Corsa Community** - Testing and Feedback

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎯 Acknowledgments

- **Stable Baselines3** team for excellent RL implementations
- **FastAPI** community for the amazing web framework
- **React** team for the powerful frontend library
- **Assetto Corsa** for providing the simulation platform
- **Open Source Community** for countless libraries and tools

## 📞 Support

### Getting Help
- **Documentation**: Check our comprehensive docs first
- **GitHub Issues**: For bug reports and feature requests
- **Discord Community**: Join our racing and development discussions
- **Email Support**: support@lapxcel.com

### Commercial Support
Professional support and custom development services are available for racing teams and organizations. Contact us at enterprise@lapxcel.com for more information.

---

**Built with ❤️ for the sim racing community**

*LapXcel - Where Data Meets Speed*