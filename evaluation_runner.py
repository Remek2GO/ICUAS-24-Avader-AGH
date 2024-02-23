#!env/bin/python3
"""Script to run all test cases in the random order."""

from copy import deepcopy
import os
import random
import subprocess
import sys
import time
import uuid

BED_IDS = list(range(1, 28))
EVALUATOR_TEMPLATE = "rosrun icuas24_competition evaluator.py"
INPUT_BEDS_CSV = "/root/sim_ws/src/icuas24_competition/beds.csv"
OUTPUT_DIR = "/root/sim_ws/src/icuas24_competition/eval_results"
PLANTS_BEDS_TEMPLATE = (
    "rostopic pub --latch /$UAV_NAMESPACE/plants_beds std_msgs/String"
)
SESSION_FILE_PATH = "/root/sim_ws/src/icuas24_competition/startup/challenge/session.yml"
TIMEOUT_SEC = 10 * 60.0  # 10 minutes


def main(n_cases: int):
    """Run all test cases in the random order."""
    for case_no in range(n_cases):
        print(f"Running test case {case_no + 1} / {n_cases}")

        # Generate random plants_beds message
        plant_type = random.choice(["Tomato", "Eggplant", "Pepper"])
        n_samples = random.randint(1, len(BED_IDS))
        samples = random.sample(BED_IDS, n_samples)
        samples.sort()
        plants_beds_str = f"{plant_type} " + " ".join(map(str, samples))
        print(plants_beds_str)

        # Generate unique output files id
        output_id = str(uuid.uuid4()).replace("-", "")
        output_file = f"{output_id}.json"

        # Create session file
        output_lines = []
        with open(SESSION_FILE_PATH, "r") as file:
            lines = file.readlines()
            output_lines = deepcopy(lines)
            for i, line in enumerate(lines):
                if line.strip().startswith(PLANTS_BEDS_TEMPLATE):
                    beginning_space = line.split(PLANTS_BEDS_TEMPLATE)[0]
                    output_lines[i] = (
                        f'{beginning_space}{PLANTS_BEDS_TEMPLATE} "{plants_beds_str}"\n'
                    )
                elif line.strip().startswith(EVALUATOR_TEMPLATE):
                    beginning_space = line.split(EVALUATOR_TEMPLATE)[0]
                    output_lines[i] = (
                        f"{beginning_space}{EVALUATOR_TEMPLATE} {INPUT_BEDS_CSV} {OUTPUT_DIR}/{output_file}\n"
                    )

        # Write session file
        with open(SESSION_FILE_PATH, "w") as file:
            file.writelines(output_lines)

        # Run the test case
        test_proc = subprocess.Popen(
            "./startup/challenge/start.sh",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            preexec_fn=os.setsid,
        )

        # Wait for the test case to finish
        start_time = time.time()
        while output_file not in os.listdir(OUTPUT_DIR):
            time.sleep(1)
            time_diff = time.time() - start_time
            if time_diff > TIMEOUT_SEC:
                print(f"Timeout! [{time_diff:.2f} sec]")
                break
        subprocess.run(["pkill", "-f", "tmux"])
        time.sleep(5)
        print("Terminate the test case...", end=" ")
        test_proc.terminate()
        print("DONE!")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 evaluation_runner.py <n_cases>")
        sys.exit(1)
    n_cases = int(sys.argv[1])
    main(n_cases)
