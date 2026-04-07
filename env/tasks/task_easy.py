def get_task():
    import random

    # Strong signals
    strong_spam = ["suspicious_domain", "spoofed_sender"]

    # Decide label
    if random.random() < 0.6:
        correct_label = "spam"
    else:
        correct_label = "not_spam"

    return {
        "task_id": "task_easy",
        "description": "Email received",
        "correct_label": correct_label
    }