def inject_noise():
    import random

    signals = ["urgent_tone"]

    if random.random() < 0.5:
        signals.append("suspicious_domain")

    if random.random() < 0.3:
        signals.append("benign_context")

    return signals