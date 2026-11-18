import re
from collections import defaultdict


def parse_logs_to_matlab(log_text):

    scores_by_epoch = defaultdict(list)

    pattern = re.compile(r'Epoch\s+(\d+)/\d+:.*?Validation Dice score:\s*([0-9.]+)')

    for line in log_text.strip().split('\n'):
        match = pattern.search(line)
        if match:
            epoch = int(match.group(1))
            score = float(match.group(2))
            scores_by_epoch[epoch].append(score)

    sorted_epochs = sorted(scores_by_epoch.keys())
    means = []

    for epoch in sorted_epochs:
        scores = scores_by_epoch[epoch]
        avg_score = sum(scores) / len(scores)
        means.append(avg_score)
        print(f"Epoch {epoch:2d}: {len(scores)} evaluations averaged -> {avg_score:.6f}")


    matlab_array = "dice_scores = [" + ", ".join(f"{x:.6f}" for x in means) + "];"

    print("\n--- MATLAB Output ---")
    print(matlab_array)
    return means



if __name__ == "__main__":

    with open('training.log', 'r') as f:
        sample_logs = f.read()

    parse_logs_to_matlab(sample_logs)