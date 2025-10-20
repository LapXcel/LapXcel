# LapXcel API Documentation

**Version:** 1.0.0  
**Author:** Sarah Siage  
**Last Updated:** October 2024

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Telemetry API](#telemetry-api)
4. [Analytics API](#analytics-api)
5. [Training API](#training-api)
6. [WebSocket API](#websocket-api)
7. [Error Handling](#error-handling)
8. [Rate Limiting](#rate-limiting)

---

## Overview

The LapXcel API provides programmatic access to racing telemetry data, performance analytics, and machine learning training capabilities. All API endpoints are REST-based and return JSON responses.

**Base URL:** `https://api.lapxcel.app/api/v1`  
**Development URL:** `http://localhost:8000/api/v1`

### API Features

- JWT-based authentication
- Real-time telemetry streaming via WebSockets
- Comprehensive performance analytics
- ML model training and deployment
- Global leaderboards
- High-performance data processing

---

## Authentication

### Register User

Create a new user account.

**Endpoint:** `POST /auth/register`

**Request Body:**
```json
{
  "email": "user@example.com",
  "username": "racing_pro",
  "password": "SecurePass123",
  "full_name": "John Doe"
}
```

**Response:** `201 Created`
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

### Login

Authenticate and receive access token.

**Endpoint:** `POST /auth/login`

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "SecurePass123"
}
```

**Response:** `200 OK`
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

### Get Current User

Retrieve authenticated user information.

**Endpoint:** `GET /auth/me`

**Headers:**
```
Authorization: Bearer {access_token}
```

**Response:** `200 OK`
```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "email": "user@example.com",
  "username": "racing_pro",
  "full_name": "John Doe",
  "is_active": true,
  "is_admin": false,
  "created_at": "2024-01-01T00:00:00Z",
  "last_login": "2024-01-15T10:30:00Z"
}
```

---

## Telemetry API

### List Sessions

Retrieve telemetry sessions for the authenticated user.

**Endpoint:** `GET /telemetry/sessions`

**Query Parameters:**
- `limit` (optional): Number of results (default: 50, max: 100)
- `offset` (optional): Pagination offset (default: 0)
- `track_name` (optional): Filter by track name
- `car_model` (optional): Filter by car model

**Response:** `200 OK`
```json
{
  "sessions": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "session_name": "Monza Practice",
      "track_name": "Monza",
      "car_model": "Ferrari 488 GT3",
      "total_laps": 15,
      "best_lap_time": 85.234,
      "session_start": "2024-01-15T14:30:00Z",
      "is_complete": true
    }
  ],
  "total": 42,
  "limit": 50,
  "offset": 0
}
```

### Create Session

Create a new telemetry session.

**Endpoint:** `POST /telemetry/sessions`

**Request Body:**
```json
{
  "session_name": "Monza Practice",
  "track_name": "Monza",
  "car_model": "Ferrari 488 GT3",
  "weather_conditions": "Clear",
  "track_temperature": 25.5,
  "air_temperature": 20.0
}
```

**Response:** `201 Created`
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "session_name": "Monza Practice",
  "track_name": "Monza",
  "car_model": "Ferrari 488 GT3",
  "created_at": "2024-01-15T14:30:00Z"
}
```

### Upload Telemetry Data

Upload telemetry data points for a session.

**Endpoint:** `POST /telemetry/sessions/{session_id}/data`

**Request Body:**
```json
{
  "data_points": [
    {
      "timestamp": "2024-01-15T14:30:05Z",
      "speed_kmh": 245.5,
      "throttle_input": 1.0,
      "brake_input": 0.0,
      "steering_input": 0.15,
      "gear": 6,
      "rpm": 7500
    }
  ]
}
```

**Response:** `201 Created`
```json
{
  "message": "Telemetry data uploaded successfully",
  "points_created": 1234
}
```

---

## Analytics API

### Performance Overview

Get comprehensive performance overview.

**Endpoint:** `GET /analytics/performance/overview`

**Query Parameters:**
- `period`: Time period (daily, weekly, monthly, all_time)
- `track_name` (optional): Filter by track
- `car_model` (optional): Filter by car

**Response:** `200 OK`
```json
{
  "status": "success",
  "data": {
    "total_sessions": 42,
    "total_laps": 630,
    "best_lap_time": 85.234,
    "average_lap_time": 87.543,
    "consistency_score": 87.5,
    "improvement_trend": "improving",
    "tracks_driven": 5,
    "cars_used": 3
  }
}
```

### Consistency Metrics

Calculate lap time consistency.

**Endpoint:** `GET /analytics/consistency`

**Query Parameters:**
- `session_id` (optional): Specific session
- `track_name` (optional): Filter by track
- `car_model` (optional): Filter by car

**Response:** `200 OK`
```json
{
  "status": "success",
  "data": {
    "consistency_score": 87.5,
    "total_laps_analyzed": 150,
    "lap_time_statistics": {
      "mean": 87.543,
      "std_deviation": 0.345,
      "best": 85.234,
      "worst": 89.123
    },
    "sector_consistency": {
      "sector_1": {"score": 88.2, "mean": 28.5},
      "sector_2": {"score": 86.8, "mean": 35.2},
      "sector_3": {"score": 87.5, "mean": 23.8}
    }
  }
}
```

### Compare Sessions

Compare two telemetry sessions.

**Endpoint:** `POST /analytics/compare/sessions`

**Request Body:**
```json
{
  "primary_session_id": "550e8400-e29b-41d4-a716-446655440000",
  "secondary_session_id": "6fa459ea-ee8a-3ca4-894e-db77e160355e",
  "comparison_type": "detailed"
}
```

**Response:** `200 OK`
```json
{
  "status": "success",
  "data": {
    "overall_score": 15.5,
    "lap_time_delta": -0.523,
    "consistency_delta": 2.3,
    "efficiency_delta": 1.8,
    "significance": "high",
    "recommendations": [
      "Improve braking consistency in sector 2",
      "Optimize racing line in turn 3"
    ]
  }
}
```

### Leaderboard

Get global leaderboard for track/car combination.

**Endpoint:** `GET /analytics/leaderboard`

**Query Parameters:**
- `track_name`: Track name (required)
- `car_model`: Car model (required)
- `category`: Leaderboard category (overall, sector_1, sector_2, sector_3)
- `limit`: Number of results (default: 100)

**Response:** `200 OK`
```json
{
  "status": "success",
  "data": {
    "track": "Monza",
    "car": "Ferrari 488 GT3",
    "category": "overall",
    "entries": [
      {
        "rank": 1,
        "user_id": "123e4567-e89b-12d3-a456-426614174000",
        "time": 85.234,
        "consistency_score": 92.5,
        "record_date": "2024-01-15T14:30:00Z",
        "is_verified": true
      }
    ]
  }
}
```

---

## Training API

### List Training Jobs

Get all training jobs for authenticated user.

**Endpoint:** `GET /training/jobs`

**Query Parameters:**
- `status` (optional): Filter by status (pending, running, completed, failed)
- `algorithm` (optional): Filter by algorithm
- `limit`: Number of results
- `offset`: Pagination offset

**Response:** `200 OK`
```json
{
  "jobs": [
    {
      "id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
      "algorithm": "SAC",
      "environment": "AssettoCorsaEnv-v1",
      "status": "running",
      "progress_percentage": 45.2,
      "current_timestep": 452000,
      "total_timesteps": 1000000,
      "created_at": "2024-01-15T10:00:00Z"
    }
  ]
}
```

### Create Training Job

Start a new ML training job.

**Endpoint:** `POST /training/jobs`

**Request Body:**
```json
{
  "algorithm": "SAC",
  "environment": "AssettoCorsaEnv-v1",
  "total_timesteps": 1000000,
  "hyperparameters": {
    "learning_rate": 0.0003,
    "buffer_size": 1000000,
    "batch_size": 256
  },
  "description": "SAC training on Monza"
}
```

**Response:** `201 Created`
```json
{
  "id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
  "algorithm": "SAC",
  "status": "pending",
  "created_at": "2024-01-15T10:00:00Z"
}
```

### Get Training Metrics

Retrieve training metrics for a job.

**Endpoint:** `GET /training/jobs/{job_id}/metrics`

**Response:** `200 OK`
```json
{
  "job_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
  "metrics": [
    {
      "timestep": 100000,
      "episode_reward": 245.5,
      "episode_length": 500,
      "fps": 120,
      "loss": 0.05
    }
  ]
}
```

---

## WebSocket API

### Telemetry WebSocket

Connect to real-time telemetry stream.

**Endpoint:** `ws://api.lapxcel.app/ws/telemetry?token={access_token}`

**Subscribe to Session:**
```json
{
  "type": "subscribe",
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Receive Telemetry Data:**
```json
{
  "type": "telemetry_data",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "telemetry": {
    "timestamp": "2024-01-15T14:30:05Z",
    "speed": 245.5,
    "throttle": 1.0,
    "brake": 0.0,
    "steering": 0.15,
    "gear": 6,
    "rpm": 7500
  }
}
```

### Training WebSocket

Connect to training progress stream.

**Endpoint:** `ws://api.lapxcel.app/ws/training?token={access_token}`

**Subscribe to Job:**
```json
{
  "type": "subscribe",
  "job_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7"
}
```

**Receive Progress Updates:**
```json
{
  "type": "training_progress",
  "job_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
  "progress_percentage": 45.2,
  "current_timestep": 452000,
  "estimated_time_remaining": 3600
}
```

---

## Error Handling

All API errors follow a consistent format:

**Error Response:**
```json
{
  "detail": "Error message description",
  "status_code": 400,
  "error_type": "ValidationError"
}
```

### Common HTTP Status Codes

- `200 OK`: Request successful
- `201 Created`: Resource created successfully
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `422 Unprocessable Entity`: Validation error
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

---

## Rate Limiting

API requests are rate-limited to ensure fair usage:

- **Authenticated requests**: 1000 requests/hour
- **Unauthenticated requests**: 100 requests/hour
- **WebSocket connections**: 10 concurrent connections per user

Rate limit headers are included in all responses:

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 995
X-RateLimit-Reset: 1642329600
```

---

## SDK Examples

### Python

```python
import requests

# Authentication
response = requests.post(
    "https://api.lapxcel.app/api/v1/auth/login",
    json={"email": "user@example.com", "password": "password"}
)
token = response.json()["access_token"]

# Get sessions
headers = {"Authorization": f"Bearer {token}"}
sessions = requests.get(
    "https://api.lapxcel.app/api/v1/telemetry/sessions",
    headers=headers
)
```

### JavaScript

```javascript
// Authentication
const response = await fetch('https://api.lapxcel.app/api/v1/auth/login', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ email: 'user@example.com', password: 'password' })
});
const { access_token } = await response.json();

// Get sessions
const sessions = await fetch('https://api.lapxcel.app/api/v1/telemetry/sessions', {
  headers: { 'Authorization': `Bearer ${access_token}` }
});
```

---

## Support

For API support and questions:
- **Email:** support@lapxcel.app
- **Documentation:** https://docs.lapxcel.app
- **GitHub Issues:** https://github.com/lapxcel/lapxcel/issues

---

**© 2024 LapXcel. All rights reserved.**

