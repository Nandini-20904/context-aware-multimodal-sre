import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta

fake = Faker()
np.random.seed(42)
random.seed(42)

# ------ PARAMETERS -------
TOTAL_METRIC_RECORDS = 25000         # number of metric points
LOGS_PER_METRIC = (1, 4)             # min/max logs per metric timestamp
CHATS_PER_METRIC = (0, 2)            # min/max chats per metric timestamp
# -------------------------

# 1. Generate timestamps evenly spaced
start_time = datetime.now() - timedelta(days=5)
time_stamps = [start_time + timedelta(minutes=i) for i in range(TOTAL_METRIC_RECORDS)]

# 🎯 Metrics dataset
metrics_data = {
    "timestamp": time_stamps,
    "cpu_util": np.clip(np.random.normal(55, 15, TOTAL_METRIC_RECORDS), 0, 100),
    "memory_util": np.clip(np.random.normal(60, 10, TOTAL_METRIC_RECORDS), 0, 100),
    "error_rate": np.where(
        np.random.rand(TOTAL_METRIC_RECORDS) > 0.995,
        np.random.uniform(0.05, 0.2, TOTAL_METRIC_RECORDS),
        np.random.uniform(0, 0.01, TOTAL_METRIC_RECORDS)
    )
}
df_metrics = pd.DataFrame(metrics_data)

# 🎯 Logs dataset
log_levels = ["INFO", "DEBUG", "WARN", "ERROR"]
log_messages = [
    "Service started",
    "User login succeeded",
    "Cache refreshed",
    "Connection lost to database",
    "API request timeout",
    "Disk I/O failure detected",
    "High memory usage detected",
    "Slow response time",
    "Heartbeat OK",
    "Configuration loaded successfully"
]
logs_list = []

for ts in time_stamps:
    for _ in range(random.randint(*LOGS_PER_METRIC)):
        logs_list.append({
            "timestamp": ts,
            "level": random.choices(log_levels, weights=[70, 10, 10, 10])[0],
            "message": random.choice(log_messages)
        })
df_logs = pd.DataFrame(logs_list)

# 🎯 Chat dataset
chat_users = [fake.user_name() for _ in range(15)]
chat_templates = [
    "Looking into the CPU spike on {host} now.",
    "Any errors in the {service} logs?",
    "Restarted the {service} service at {time}.",
    "Seeing 500 errors from {service}, anyone else?",
    "Resolved after redeploying {service}.",
    "Noticed memory leak before the failure.",
    "Ticket #{ticket_id} updated."
]

services = ["auth", "payment", "search", "inventory", "analytics"]
hosts = [f"node-{i}" for i in range(1, 8)]

chat_list = []
for ts in time_stamps:
    for _ in range(random.randint(*CHATS_PER_METRIC)):
        tmpl = random.choice(chat_templates)
        msg = tmpl.format(
            host=random.choice(hosts),
            service=random.choice(services),
            time=ts.strftime("%H:%M"),
            ticket_id=random.randint(100, 999)
        )
        chat_list.append({
            "timestamp": ts,
            "user": random.choice(chat_users),
            "message": msg
        })
df_chat = pd.DataFrame(chat_list)

# 🎯 Ticket dataset (optional)
ticket_list = []
for _ in range(200):
    created_at = random.choice(time_stamps)
    ticket_list.append({
        "ticket_id": fake.random_int(min=900, max=2000),
        "created_at": created_at,
        "status": random.choice(["open", "investigating", "resolved"]),
        "summary": random.choice([
            "CPU spike on {}".format(random.choice(hosts)),
            "High error rate in {}".format(random.choice(services)),
            "Database connection loss",
            "Memory leak in {}".format(random.choice(services))
        ])
    })
df_tickets = pd.DataFrame(ticket_list)

# Save all datasets
df_metrics.to_csv("metrics.csv", index=False)
df_logs.to_csv("logs.csv", index=False)
df_chat.to_csv("chat.csv", index=False)
df_tickets.to_csv("tickets.csv", index=False)

print(f"Generated datasets:\n- metrics.csv ({len(df_metrics)})\n- logs.csv ({len(df_logs)})\n- chat.csv ({len(df_chat)})\n- tickets.csv ({len(df_tickets)})")
