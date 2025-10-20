"""
Load Test Configuration
Configuration settings for performance testing.
Author: Sarah Siage
"""

from typing import Dict, Any


# Load test scenarios
LOAD_TEST_SCENARIOS = {
    "smoke": {
        "users": 10,
        "spawn_rate": 2,
        "run_time": "1m",
        "description": "Smoke test with minimal load"
    },
    "load": {
        "users": 100,
        "spawn_rate": 10,
        "run_time": "10m",
        "description": "Standard load test"
    },
    "stress": {
        "users": 500,
        "spawn_rate": 50,
        "run_time": "30m",
        "description": "Stress test to find breaking point"
    },
    "spike": {
        "users": 1000,
        "spawn_rate": 100,
        "run_time": "5m",
        "description": "Spike test with sudden load increase"
    },
    "endurance": {
        "users": 200,
        "spawn_rate": 20,
        "run_time": "2h",
        "description": "Endurance test for stability over time"
    }
}


# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    "max_response_time_ms": 2000,  # Maximum acceptable response time
    "avg_response_time_ms": 500,   # Target average response time
    "p95_response_time_ms": 1000,  # 95th percentile response time
    "p99_response_time_ms": 1500,  # 99th percentile response time
    "max_error_rate": 0.01,        # Maximum 1% error rate
    "min_throughput_rps": 50       # Minimum requests per second
}


# Endpoint-specific thresholds
ENDPOINT_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "/api/v1/auth/login": {
        "max_response_time_ms": 1000,
        "avg_response_time_ms": 300,
    },
    "/api/v1/telemetry/sessions": {
        "max_response_time_ms": 1500,
        "avg_response_time_ms": 400,
    },
    "/api/v1/analytics/performance/overview": {
        "max_response_time_ms": 2000,
        "avg_response_time_ms": 600,
    },
    "/api/v1/training/jobs": {
        "max_response_time_ms": 2500,
        "avg_response_time_ms": 800,
    }
}


# Test environment configuration
TEST_ENVIRONMENTS = {
    "local": {
        "host": "http://localhost:8000",
        "description": "Local development environment"
    },
    "staging": {
        "host": "https://staging.lapxcel.app",
        "description": "Staging environment"
    },
    "production": {
        "host": "https://api.lapxcel.app",
        "description": "Production environment (use with caution!)"
    }
}


# Test data generation settings
TEST_DATA_CONFIG = {
    "num_test_users": 100,
    "num_test_sessions": 500,
    "num_test_laps": 5000,
    "tracks": [
        "Monza",
        "Spa-Francorchamps",
        "Silverstone",
        "Nürburgring",
        "Suzuka",
        "Circuit de Barcelona-Catalunya",
        "Brands Hatch",
        "Imola"
    ],
    "cars": [
        "Ferrari 488 GT3",
        "McLaren 720S GT3",
        "Mercedes-AMG GT3",
        "Porsche 911 GT3 R",
        "Audi R8 LMS",
        "BMW M6 GT3",
        "Lamborghini Huracan GT3",
        "Nissan GT-R Nismo GT3"
    ]
}


# Monitoring and reporting
MONITORING_CONFIG = {
    "enable_real_time_charts": True,
    "save_html_report": True,
    "save_csv_stats": True,
    "save_json_stats": True,
    "report_directory": "./load_test_reports",
    "screenshot_on_failure": True
}


# Custom headers for load testing
LOAD_TEST_HEADERS = {
    "User-Agent": "LapXcel-LoadTest/1.0",
    "X-Load-Test": "true"
}


def get_scenario_config(scenario_name: str) -> Dict[str, Any]:
    """Get configuration for a specific load test scenario."""
    if scenario_name not in LOAD_TEST_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_name}")
    return LOAD_TEST_SCENARIOS[scenario_name]


def get_locust_command(scenario: str, environment: str = "local") -> str:
    """Generate Locust command for a specific scenario."""
    config = get_scenario_config(scenario)
    env_config = TEST_ENVIRONMENTS.get(environment, TEST_ENVIRONMENTS["local"])
    
    return (
        f"locust -f locustfile.py "
        f"--users {config['users']} "
        f"--spawn-rate {config['spawn_rate']} "
        f"--run-time {config['run_time']} "
        f"--host {env_config['host']} "
        f"--headless "
        f"--html ./load_test_reports/{scenario}_{environment}_report.html"
    )


def validate_test_results(stats: Dict[str, Any]) -> Dict[str, bool]:
    """Validate test results against performance thresholds."""
    results = {
        "response_time_ok": stats.get("avg_response_time", 0) <= PERFORMANCE_THRESHOLDS["avg_response_time_ms"],
        "error_rate_ok": stats.get("error_rate", 0) <= PERFORMANCE_THRESHOLDS["max_error_rate"],
        "throughput_ok": stats.get("requests_per_second", 0) >= PERFORMANCE_THRESHOLDS["min_throughput_rps"]
    }
    
    results["all_passed"] = all(results.values())
    return results


# Example usage
if __name__ == "__main__":
    # Print all available scenarios
    print("Available load test scenarios:")
    for name, config in LOAD_TEST_SCENARIOS.items():
        print(f"\n{name}: {config['description']}")
        print(f"  Command: {get_locust_command(name, 'local')}")

