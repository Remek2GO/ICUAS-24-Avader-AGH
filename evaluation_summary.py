#!env/bin/python3
"""Create a summary of the evaluation results."""

import json
import numpy as np
import os
from tqdm import tqdm


def main(results_dir: str):
    """Create a summary of the evaluation results."""
    # Read all results from the directory
    input_json_files = [f for f in os.listdir(results_dir) if f.endswith(".json")]

    # Gather data from all jsons to lists
    fruit_count_gt = []
    fruit_count_result = []
    collision_cnt = []
    beds_not_counted = []
    final_points = []
    tasks = []
    for input_json_file in tqdm(input_json_files):
        with open(os.path.join(results_dir, input_json_file), "r") as f:
            result = json.load(f)
            fruit_count_gt.append(result["fruit_count_gt"])
            fruit_count_result.append(result["fruit_count_result"])
            collision_cnt.append(result["collision_cnt"])
            beds_not_counted.append(len(result["beds_not_counted"]))
            final_points.append(result["final_points"])
            tasks.append(
                result["fruit_type"][0]
                + result["fruit_type"][1:].lower()
                + " "
                + str(result["beds_to_visit"])
                .replace("[", "")
                .replace("]", "")
                .replace(",", "")
            )

    # Gather some statistics
    percentage_incorrect_fruit_count = sum(
        [1 for i, j in zip(fruit_count_gt, fruit_count_result) if i != j]
    ) / len(fruit_count_gt)
    percentage_with_collision = sum([1 for i in collision_cnt if i > 0]) / len(
        collision_cnt
    )
    percentage_with_missing_beds = sum([1 for i in beds_not_counted if i > 0]) / len(
        beds_not_counted
    )
    avg_final_points = sum(final_points) / len(final_points)
    min_final_points = min(final_points)
    max_final_points = max(final_points)
    std_final_points = np.std(final_points)

    # Print the summary
    print(f"Results for {len(input_json_files)} test cases:")
    print(
        f"\t{percentage_incorrect_fruit_count:.2%} with incorrect fruit count "
        f"({sum([1 for i, j in zip(fruit_count_gt, fruit_count_result) if i != j])} of "
        f"{len(fruit_count_gt)})"
    )
    print(
        f"\t{percentage_with_collision:.2%} with collision "
        f"({sum([1 for i in collision_cnt if i > 0])} of {len(collision_cnt)})"
    )
    print(
        f"\t{percentage_with_missing_beds:.2%} with missing beds "
        f"({sum([1 for i in beds_not_counted if i > 0])} of {len(beds_not_counted)})"
    )
    print(
        f"\tAverage final points: {avg_final_points:.2f} "
        f"(min: {min_final_points:.2f}, max: {max_final_points:.2f}, "
        f"std: {std_final_points:.2f})"
    )

    # Get errorenous results
    errorenous_results = [
        (input_json_files[i], tasks[i])
        for i, result in enumerate(
            zip(fruit_count_gt, fruit_count_result, collision_cnt, beds_not_counted)
        )
        if result[0] != result[1] or result[2] > 0 or result[3] > 0
    ]
    if len(errorenous_results) > 0:
        print("Errorenous result(s):")
        for file, result in errorenous_results:
            print(f"{file}: {result}")

    # # Create a summary
    # summary = {
    #     "n_test_cases": 0,
    #     "n_test_cases_with_incorrect_fruit_count": 0,
    #     "n_test_cases_with_collision": 0,
    #     "n_test_cases with_missing_beds": 0,
    #     "n_total_points_sum": 0,
    # }

    # # Gather data
    # errorenous_results = {}
    # for input_json_file in tqdm(input_json_files):
    #     is_errorenous = False
    #     with open(os.path.join(results_dir, input_json_file), "r") as f:
    #         result = json.load(f)
    #         summary["n_test_cases"] += 1
    #         if result["fruit_count_gt"] != result["fruit_count_result"]:
    #             summary["n_test_cases_with_incorrect_fruit_count"] += 1
    #             is_errorenous = True
    #         if result["collision_cnt"] > 0:
    #             summary["n_test_cases_with_collision"] += 1
    #             is_errorenous = True
    #         if len(result["beds_not_counted"]) > 0:
    #             summary["n_test_cases with_missing_beds"] += 1
    #             is_errorenous = True
    #         summary["n_total_points_sum"] += result["final_points"]
    #     if is_errorenous:
    #         errorenous_results[input_json_file] = result

    # # Gather some statistics
    # percentage_incorrect_fruit_count = (
    #     summary["n_test_cases_with_incorrect_fruit_count"] / summary["n_test_cases"]
    # )
    # percentage_with_collision = (
    #     summary["n_test_cases_with_collision"] / summary["n_test_cases"]
    # )
    # percentage_with_missing_beds = (
    #     summary["n_test_cases with_missing_beds"] / summary["n_test_cases"]
    # )
    # avg_final_points = summary["n_total_points_sum"] / summary["n_test_cases"]

    # # Print the summary
    # print(f"{summary['n_test_cases']} test cases:")
    # print(
    #     f"\t{summary['n_test_cases_with_incorrect_fruit_count']} with incorrect fruit "
    #     f"count [{percentage_incorrect_fruit_count:.2%}]"
    # )
    # print(
    #     f"\t{summary['n_test_cases_with_collision']} with collision "
    #     f"[{percentage_with_collision:.2%}]"
    # )
    # print(
    #     f"\t{summary['n_test_cases with_missing_beds']} with missing beds "
    #     f"[{percentage_with_missing_beds:.2%}]"
    # )
    # print(f"\tAverage final points: {avg_final_points:.2f}")
    # if len(errorenous_results) > 0:
    #     print("Errorenous result(s):")
    #     for file, result in errorenous_results.items():
    #         print(f"{file}:\n {result}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results_dir",
        help="The directory containing the evaluation results.",
    )
    args = parser.parse_args()
    main(args.results_dir)
