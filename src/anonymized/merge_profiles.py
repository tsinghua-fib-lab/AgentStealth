import argparse
import json
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.reddit.reddit_utils import load_data
from src.reddit.reddit_types import Profile
from typing import List


if __name__ == "__main__":

    # PArse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_paths",
        type=str,
        nargs="+",
        help="Paths to the merge_files",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        help="Path to the output file",
    )

    args = parser.parse_args()

    # Load data
    profiles: List[List[Profile]] = []
    for path in args.in_paths:
        profiles.append(load_data(path))

    lengths = [len(x) for x in profiles]
    # assert that all length are equal
    assert len(set(lengths)) == 1

    resulting_profiles: List[Profile] = []
    for i in range(lengths[0]):
        resulting_profiles.append(profiles[0][i])

        for j in range(1, len(profiles)):

            if resulting_profiles[i].username != profiles[j][i].username:
                print("Usernames do not match!")
                print(resulting_profiles[i].username)
                print(profiles[j][i].username)
                exit(1)

            # If one has more comments than the other, add the comments
            len1 = len(resulting_profiles[i].comments)
            len2 = len(profiles[j][i].comments)
            if len1 < len2:
                resulting_profiles[i].comments += profiles[j][i].comments[len1:]

            # Iterate over all joint comments and check if they have predictions and utility
            shared_comments = min(len1, len2)
            for k in range(shared_comments):
                for key in profiles[j][i].comments[k].predictions:
                    if key not in resulting_profiles[i].comments[k].predictions:
                        resulting_profiles[i].comments[k].predictions[key] = (
                            profiles[j][i].comments[k].predictions[key]
                        )

                for key in profiles[j][i].comments[k].utility:
                    if key not in resulting_profiles[i].comments[k].utility:
                        resulting_profiles[i].comments[k].utility[key] = (
                            profiles[j][i].comments[k].utility[key]
                        )

    # Write data
    if os.path.exists(args.out_path):
        assert False, "File already exists!"

    with open(args.out_path, "w") as f:
        for profile in resulting_profiles:
            f.write(json.dumps(profile.to_json()) + "\n")
