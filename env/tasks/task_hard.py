def get_task(signals):
    # truth depends on signals (hard!)
    if "trusted_sender" in signals:
        correct = "important"
    else:
        correct = "spam"

    return {
        "task_id": "hard",
        "description": "Truth depends on context",
        "correct_label": correct
    }