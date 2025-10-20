"""
Load Testing with Locust
Performance and load testing for LapXcel Backend API.
Author: Sarah Siage
"""

from locust import HttpUser, task, between, events
import json
import random
from datetime import datetime


class LapXcelUser(HttpUser):
    """Simulated user for load testing LapXcel API."""
    
    wait_time = between(1, 5)  # Wait 1-5 seconds between tasks
    host = "http://localhost:8000"
    
    def on_start(self):
        """Initialize user session with authentication."""
        # Register or login
        self.email = f"loadtest_{random.randint(1000, 9999)}@example.com"
        self.password = "LoadTest123"
        
        # Try to register
        response = self.client.post("/api/v1/auth/register", json={
            "email": self.email,
            "username": f"loadtest_{random.randint(1000, 9999)}",
            "password": self.password
        }, catch_response=True)
        
        if response.status_code not in [200, 201]:
            # Already exists, just login
            response = self.client.post("/api/v1/auth/login", json={
                "email": self.email,
                "password": self.password
            })
        
        if response.status_code in [200, 201]:
            data = response.json()
            self.access_token = data.get("access_token")
            self.headers = {"Authorization": f"Bearer {self.access_token}"}
        else:
            self.access_token = None
            self.headers = {}
        
        self.session_ids = []
    
    @task(3)
    def view_dashboard(self):
        """Simulate viewing the dashboard."""
        if not self.access_token:
            return
        
        self.client.get(
            "/api/v1/analytics/performance/overview",
            params={"period": random.choice(["daily", "weekly", "monthly"])},
            headers=self.headers,
            name="/api/v1/analytics/performance/overview"
        )
    
    @task(2)
    def list_sessions(self):
        """Simulate listing telemetry sessions."""
        if not self.access_token:
            return
        
        self.client.get(
            "/api/v1/telemetry/sessions",
            headers=self.headers,
            name="/api/v1/telemetry/sessions"
        )
    
    @task(1)
    def create_session(self):
        """Simulate creating a new telemetry session."""
        if not self.access_token:
            return
        
        tracks = ["Monza", "Spa", "Silverstone", "Nürburgring", "Suzuka"]
        cars = ["Ferrari 488 GT3", "McLaren 720S GT3", "Mercedes AMG GT3", "Porsche 911 GT3"]
        
        session_data = {
            "session_name": f"Load Test Session {random.randint(1, 1000)}",
            "track_name": random.choice(tracks),
            "car_model": random.choice(cars),
            "weather_conditions": random.choice(["Clear", "Cloudy", "Rainy"]),
            "track_temperature": random.uniform(20, 35),
            "air_temperature": random.uniform(15, 30)
        }
        
        response = self.client.post(
            "/api/v1/telemetry/sessions",
            json=session_data,
            headers=self.headers,
            name="/api/v1/telemetry/sessions [POST]"
        )
        
        if response.status_code in [200, 201]:
            session_id = response.json().get("id")
            if session_id:
                self.session_ids.append(session_id)
    
    @task(2)
    def view_session_detail(self):
        """Simulate viewing session details."""
        if not self.access_token or not self.session_ids:
            return
        
        session_id = random.choice(self.session_ids)
        self.client.get(
            f"/api/v1/telemetry/sessions/{session_id}",
            headers=self.headers,
            name="/api/v1/telemetry/sessions/[id]"
        )
    
    @task(1)
    def get_analytics(self):
        """Simulate getting analytics."""
        if not self.access_token:
            return
        
        self.client.get(
            "/api/v1/analytics/consistency",
            headers=self.headers,
            name="/api/v1/analytics/consistency"
        )
    
    @task(1)
    def list_training_jobs(self):
        """Simulate listing training jobs."""
        if not self.access_token:
            return
        
        self.client.get(
            "/api/v1/training/jobs",
            headers=self.headers,
            name="/api/v1/training/jobs"
        )
    
    @task(1)
    def create_training_job(self):
        """Simulate creating a training job."""
        if not self.access_token:
            return
        
        job_data = {
            "algorithm": random.choice(["SAC", "TQC", "PPO", "TD3"]),
            "environment": "AssettoCorsaEnv-v1",
            "total_timesteps": random.choice([100000, 500000, 1000000]),
            "description": f"Load test job {random.randint(1, 1000)}"
        }
        
        self.client.post(
            "/api/v1/training/jobs",
            json=job_data,
            headers=self.headers,
            name="/api/v1/training/jobs [POST]"
        )
    
    @task(1)
    def get_leaderboard(self):
        """Simulate getting leaderboard."""
        tracks = ["Monza", "Spa", "Silverstone"]
        cars = ["Ferrari 488 GT3", "McLaren 720S GT3"]
        
        self.client.get(
            "/api/v1/analytics/leaderboard",
            params={
                "track_name": random.choice(tracks),
                "car_model": random.choice(cars),
                "category": "overall"
            },
            name="/api/v1/analytics/leaderboard"
        )


class AdminUser(HttpUser):
    """Simulated admin user for testing admin endpoints."""
    
    wait_time = between(2, 5)
    host = "http://localhost:8000"
    weight = 1  # Lower weight than regular users
    
    def on_start(self):
        """Login as admin."""
        response = self.client.post("/api/v1/auth/login", json={
            "email": "admin@lapxcel.com",
            "password": "AdminPassword123"
        })
        
        if response.status_code == 200:
            data = response.json()
            self.access_token = data.get("access_token")
            self.headers = {"Authorization": f"Bearer {self.access_token}"}
        else:
            self.access_token = None
            self.headers = {}
    
    @task
    def list_all_users(self):
        """Simulate listing all users."""
        if not self.access_token:
            return
        
        self.client.get(
            "/api/v1/auth/users",
            headers=self.headers,
            name="/api/v1/auth/users [Admin]"
        )
    
    @task
    def view_system_metrics(self):
        """Simulate viewing system metrics."""
        if not self.access_token:
            return
        
        self.client.get(
            "/api/v1/admin/metrics",
            headers=self.headers,
            name="/api/v1/admin/metrics [Admin]"
        )


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when load test starts."""
    print("Starting load test...")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when load test stops."""
    print("Load test completed!")
    print(f"Total requests: {environment.stats.total.num_requests}")
    print(f"Failed requests: {environment.stats.total.num_failures}")
    print(f"Average response time: {environment.stats.total.avg_response_time:.2f}ms")

