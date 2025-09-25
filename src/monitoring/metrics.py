from prometheus_client import Counter, Histogram

REQUESTS = Counter("dm_api_requests_total", "Total API requests", ["endpoint", "method", "status"])
LATENCY = Histogram("dm_api_latency_seconds", "API latency seconds", ["endpoint"])
