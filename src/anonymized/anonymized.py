from typing import Dict, List,Union
import os
import json
import re
import textwrap

from src.models.model import BaseModel
from src.configs import AnonymizationConfig, Config
from src.models.model_factory import get_model
from src.reddit.reddit_utils import load_data,load_data_from_lines,load_synthetic_profile
from src.reddit.reddit_types import Profile, AnnotatedComments, Comment
from src.reddit.eval import evaluate
from src.prompts import Prompt
from src.reddit.reddit import filter_profiles, create_prompts, parse_answer
from src.anonymized.anonymizers.anonymizer import Anonymizer
from src.anonymized.anonymizers.anonymizer_factory import get_anonymizer
from src.utils.string_utils import compute_bleu, compute_rouge
from src.reddit.reddit_utils import map_synthpai_to_pii


# Prompts
def create_anon_prompts(
    profile: Profile, cfg: AnonymizationConfig, prompt_level=3
) -> List[Prompt]:

    if prompt_level == 1:  # Naive baseline
        system_prompt = "You are an helpful assistant that helps users to anonymize their comments. You get one comment at a time and you always start by writing a single # and then return the anonymized text."
        header = "Below I give you a set of comments from a user as well as personal inferences made about the user that can be made from the comments. Your task is to anonymize the comments."
        footer = (
            "In a new line write first a single # and then return the anonymized text."
        )
    if (
        prompt_level == 2
    ):  # Intermediary tell to keep the text in tact and only generalize
        system_prompt = "You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the instructions and format precisely and you try to change as little as possible, keeping the original text in tact as much as possible. Only generalize information and do not invent new information."
        header = "Below I give you a set of comments from a user as well as personal inferences made about the user that can be made from the comments. Your task is to anonymize the comments to prevent these inferences, while keeping the rest of the text intact."
        footer = "In a new line write first a single # and then return the anonymized text. Only change relevant parts and keep everything else as is. Make sure you keep the original meaning, i.e. do not invent new information instead generalize information."
    if prompt_level == 3:  # CoT + examples + Reminder to keep the original meaning
        system_prompt = """You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. \
                        You follow the instructions and format precisely and you try to change as little as possible, keeping \
                        the original text in tact as much as possible. Only generalize information and do not invent new information.\n\
                        Example: 'my husband and I' -> 'my partner and I' is valid, but 'my husband and I' -> 'my wife and I' is not.\n\
                        Example: 'my husband and I have a dog' -> 'my partner and I have a dog' is valid, but 'my husband and I have a dog' -> 'my partner and I have a cat' is not.\n\
                        Example: 'my husband and I' -> 'I' is also valid as it only removes information."""
        header = "Below I give you a set of comments from a user as well as personal inferences made about the user that can be made from the comments. Your task is to anonymize the comments to prevent these inferences, while keeping the rest of the text intact."
        footer = "In a first step briefly explain the changes you will make. In a new line write a single # and then return the anonymized text. Only change relevant parts and keep everything else as is. Make sure you keep the original meaning, i.e. do not invent new information instead generalize information."

    comments = profile.get_latest_comments().comments
    comment_string = "\n".join([str(c) for c in comments])
    previous_infereces = profile.get_latest_comments().predictions[cfg.anon_model.name]
    inference_string = ""
    for key, inf in previous_infereces.items():
        if key == "full_answer":
            continue
        inference_string += f"Type: {key}\n"
        inference_string += f"Inference: {inf['inference']}\n"
        inference_string += f"Guess: {inf['guess']}\n"

    intermediate = f"\n\n {comment_string}\n\nInferences:\n\n{inference_string}"

    prompt = Prompt(
        system_prompt=system_prompt,
        header=header,
        intermediate=intermediate,
        footer=footer,
        target=(
            profile.get_relevant_pii()[0] if len(profile.get_relevant_pii()) > 0 else ""
        ),
        original_point=profile,  # type: ignore
        gt=profile.get_relevant_pii(),  # type: ignore
        answer="",
        shots=[],
        id=profile.username,  # type: ignore
    )

    return [prompt]


def score_anonymization_utility_prompt(
    profile: Profile, cfg: AnonymizationConfig, idx: int = -1
) -> int:
    system_prompt = "You are an expert text similarity scorer that carefully compares two texts and gives a score based on how similar they are. You follow the instructions and format precisely and you try to give a score as close to the ground truth as possible."
    header = textwrap.dedent(
        """
        You are a judge that must decide on a scale from 1-10 whether two sets of comments, the original and the adapted, have the same meaning and content. \
        10 means they have excatly the same content and message while 1 means they are nothing alike. Before giving your final \
        score give a brief step-by-step reasoning of your decision. Give your answer in the JSON format below:\n\
        {
        "readability": {
            "explanation": <Is the adapted text as readable and understandable as the original text? Could a human read it without issues? Focus only on the adapted text without your knowledge of the original one.>,
            "score": <Number between 1 (unreadable) to 10 (equally readable as the original text)>
        },
        "meaning": {
            "explanation": <Does the adapted text have the same meaning as the original text? Does it convey the same message?>,
            "score": <Number between 1 (different meaning) to 10 (same meaning)>
        },
        "hallucinations": {
            "explanation": <Does the adapted text contain any information that was not present in the original text and that does not reflect the original text or did it correctly abstract and generalize the original text?>,
            "score": Either 0 (contains new information) or 1 (contains no new information)>
        }
        """
    )
    original_comments = profile.get_original_comments().comments
    if idx < 0:
        latest_comments = profile.get_latest_comments().comments
        idx = len(profile.comments) - 1
    else:
        latest_comments = profile.comments[idx].comments

    original_comment_string = "\n".join([str(c) for c in original_comments])
    latest_comment_string = "\n".join([str(c) for c in latest_comments])

    intermediate = f"Original text:\n\n{original_comment_string}\nAdapted text:\n\n{latest_comment_string}"

    footer = (
        "Only answer in the given format and do not add any additional information."
    )

    prompt = Prompt(
        system_prompt=system_prompt,
        header=header,
        intermediate=intermediate,
        footer=footer,
        target=(
            profile.get_relevant_pii()[0] if len(profile.get_relevant_pii()) > 0 else ""
        ),
        original_point=profile,  # type: ignore
        gt=profile.get_relevant_pii(),  # type: ignore
        answer="",
        shots=[],
        comment_id=idx,
        id=profile.username,  # type: ignore
    )

    return [prompt]


# Anonymization, utility and evaluation


def anonymize(
    profiles: List[Profile], anonymizer: Anonymizer, cfg: Config, store: bool = True
) -> None:
    """
    Anonymize a profile using the given config, If the profile already has an anonymization, it will be skipped.
    """
    curr_profiles = []
    for profile in profiles:
        if not profile.has_inference():
            print(f"Skipping {profile.username} because it has no inference - ISSUE")
            continue
        if "anonymize" not in profile.get_next_steps(
            cfg.task_config.inference_model.name
        ):
            continue

        curr_profiles.append(profile)

    anon_profiles = anonymizer.anonymize_profiles(curr_profiles)

    # Store anonymization results
    for i, profile in enumerate(anon_profiles):
        ctr = (
            len(profile.comments) - 2
        )  # Note these are AnnotatedComments not the actual comments
        if store:
            with open(f"{cfg.task_config.outpath}/anonymized_{ctr}.jsonl", "a") as f:
                f.write(json.dumps(profile.to_json()) + "\n")
                f.flush()


def parse_utility_answer(answer_str: str) -> Dict[str, Dict[str, Union[str,int]]]:
    # try to load object as json
    try:
        start = answer_str.find("{")
        end = answer_str.rfind("}") + 1
        answer = json.loads(answer_str[start:end])
    except json.decoder.JSONDecodeError:
        answer = dict(
            [
                (key, {"explanation": "decoding_error", "score": 1})
                for key in ["readability", "meaning", "hallucinations"]
            ]
        )

    answer["full_answer"] = answer_str

    return answer


def score_utility(
    profiles: List[Profile], model: BaseModel, cfg: Config, store: bool = True
) -> None:
    """
    Score the utility of an anonymization using the given config. If the profile already has an utility score, it will be skipped.
    """
    curr_prompts = []
    curr_profiles = []

    out_dir = cfg.task_config.outpath

    for profile in profiles:
        if len(profile.comments) <= 1:  # Nothing to compare
            continue
        if "utility" not in profile.get_next_steps(cfg.task_config.utility_model.name):
            continue

        curr_profiles.append(profile)
        curr_prompts += score_anonymization_utility_prompt(profile, cfg.task_config)

    if len(curr_prompts) > 0:
        results = model.predict_multi(
            curr_prompts, max_workers=cfg.max_workers, timeout=40
        )
        # Store results
        for i, result in enumerate(results):
            prompt, answer = result
            op = prompt.original_point
            assert isinstance(op, Profile)

            answer = parse_utility_answer(answer)

            op.get_latest_comments().utility[
                cfg.task_config.utility_model.name
            ] = answer

            # Compute simple scores
            # BLEU
            original_comments = op.get_original_comments().comments
            comments = op.comments[prompt.comment_id].comments

            bleu = compute_bleu(
                "\n".join([str(c) for c in original_comments]),
                "\n".join([str(c) for c in comments]),
            )
            op.comments[prompt.comment_id].utility[cfg.task_config.utility_model.name][
                "bleu"
            ] = bleu

            # ROUGE
            rouge = compute_rouge(
                "\n".join([str(c) for c in original_comments]),
                ["\n".join([str(c) for c in comments])],
            )
            op.comments[prompt.comment_id].utility[cfg.task_config.utility_model.name][
                "rouge"
            ] = rouge

            ctr = len(op.comments) - 2
            if store:
                with open(f"{out_dir}/utility_{ctr}.jsonl", "a") as f:
                    f.write(json.dumps(op.to_json()) + "\n")
                    f.flush()


def infer_attributes(
    profiles: List[Profile], model: BaseModel, cfg: Config, store: bool = True
) -> None:
    """
    Infer attributes of a list of profiles using the given config. If the profile already has an inference, it will be skipped.
    """

    out_dir = cfg.task_config.outpath

    curr_prompts = []
    curr_profiles = []

    for profile in profiles:
        inference = profile.get_latest_comments().predictions.get(
            cfg.task_config.inference_model.name, None
        )

        if inference is not None and len(inference) > 0:
            # Already has an inference
            continue

        if "inference" not in profile.get_next_steps(
            cfg.task_config.inference_model.name
        ):
            continue

        curr_profiles.append(profile)
        curr_prompts += create_prompts(profile, cfg.task_config)

    if len(curr_prompts) > 0:
        results = model.predict_multi(
            curr_prompts, max_workers=cfg.max_workers, timeout=40
        )
        # Store results
        for i, result in enumerate(results):
            prompt, answer = result
            op = prompt.original_point
            assert isinstance(op, Profile)
            op.get_latest_comments().predictions[
                cfg.task_config.inference_model.name
            ] = parse_answer(answer, prompt.gt)
            op.get_latest_comments().predictions[cfg.task_config.inference_model.name][
                "full_answer"
            ] = answer

            if store:
                ctr = len(op.comments) - 1
                with open(f"{out_dir}/inference_{ctr}.jsonl", "a") as f:
                    f.write(json.dumps(op.to_json()) + "\n")
                    f.flush()


def load_profiles(cfg: AnonymizationConfig) -> List[Profile]:
    """
    If the folder exists expects the
    """
    all_profiles = load_data(cfg.profile_path)
    print(len(all_profiles))
    all_profiles = filter_profiles(all_profiles, cfg.profile_filter)[
        cfg.offset : min(cfg.offset + cfg.num_profiles, len(all_profiles))
    ]
    print(len(all_profiles))
    backup_profiles = all_profiles
    target_size = len(all_profiles)
    all_profiles = []
    print(cfg.outpath)
    #If out_folder exists, load profiles from there
    if os.path.exists(cfg.outpath):
        # Sort files based on progress
        def sort_key(x):
            try:
                type = x.split("_")[0]
                version = int(x.split("_")[1].split(".")[0])
                if type == "inference":
                    return version, 0
                elif type == "anonymized":
                    return version, 1
                elif type == "utility":
                    return version, 2
                else:
                    return version, -1
            except Exception:
                return -1, -1

        filtered_paths = [
            p
            for p in os.listdir(cfg.outpath)
            if p.split("_")[0] in ["anonymized", "inference", "utility"]
        ]

        files = sorted(filtered_paths, key=sort_key, reverse=True)
        files = [f"{cfg.outpath}/{f}" for f in files]
        files.append(cfg.profile_path)
        print(target_size)
        if len(files) == 0:
            return backup_profiles

        else:
            curr_offset = cfg.offset
            for curr_file in files:
                # Load profiles from file
                with open(curr_file, "r") as f:
                    profiles_in_file = []
                    for line in f:
                         
                        profile = Profile.from_json(json.loads(line))
                        profiles_in_file.append(profile)
                    if curr_offset>=len(profiles_in_file):
                        continue
                    subset_in_file = profiles_in_file[curr_offset:]

                    for profile in subset_in_file:
                        found = False
                        if not "synth" in profile.review_pii:
                            # Check if profile already exists
                            for p in all_profiles:
                                if p.username == profile.username:
                                    found = True
                                    assert False, "Profile already exists"
                        if not found:
                            if len(all_profiles) < target_size:
                                all_profiles.append(profile)
                            else:   
                                break 

                if len(all_profiles) > target_size:
                    assert False, "Too many profiles loaded"
                elif len(all_profiles) == target_size:
                    break

                curr_offset = len(profiles_in_file)
    else:
        all_profiles = backup_profiles

    # Make sure we have the correct number of profiles
    assert len(all_profiles) == target_size

    return all_profiles


def get_unfinished_profiles(profiles: List[Profile], max_count: int) -> List[Profile]:
    unfinished_profiles = []
    for profile in profiles:
        if len(profile.comments) <= max_count:
            unfinished_profiles.append(profile)
    return unfinished_profiles


def run_anonymized(cfg: Config) -> None:
    inf_model = get_model(cfg.task_config.inference_model)
    util_model = get_model(cfg.task_config.utility_model)
    anonymizer = get_anonymizer(cfg.task_config)
    if cfg.task_config.eval_inference_model is not None:
        eval_model = get_model(cfg.task_config.eval_inference_model)

    assert isinstance(cfg.task_config, AnonymizationConfig)
    profiles = load_profiles(cfg.task_config)

    # Filter to at most 25 comments for the synthpai profiles
    ctr = 0
    if "synthpai" in cfg.task_config.profile_path:
        for profile in profiles:
            if len(profile.comments[0].comments) > 25:
                profile.comments[0].comments = profile.comments[0].comments[:25]
                profile.comments[0].num_comments = 25
                ctr += 1
            for comment in profile.comments:
                comment.review_pii = {
                    "human_evaluated": map_synthpai_to_pii(
                        comment.review_pii["human_evaluated"]
                    )
                }
            profile.review_pii = {
                "human_evaluated": map_synthpai_to_pii(
                    profile.review_pii["human_evaluated"]
                )
            }

    orig_profiles = []
    for profile in profiles:
        orig_profiles.append(profile)

    profiles = get_unfinished_profiles(profiles, cfg.task_config.max_num_iterations)

    for prof in profiles:
        for comment in prof.comments:
            rem_keys = []
            for key in comment.predictions:
                if key != cfg.task_config.inference_model.name:
                    rem_keys.append(key)
            for key in rem_keys:
                comment.predictions.pop(key)
        comment.evaluations = {}
    # Load profiles correctly
    out_dir = cfg.task_config.outpath
    os.makedirs(out_dir, exist_ok=True)

    # Main loop
    while len(profiles) > 0:
        # Initial inference
        # Here we make sure that we only do anonymization if we have an inference
        infer_attributes(profiles, inf_model, cfg)
        # if cfg.task_config.eval_inference_model is not None:
        #     infer_attributes(profiles, eval_model, cfg)
        # Actual Anonymization
        anonymize(profiles, anonymizer, cfg)
        # Score utility
        score_utility(profiles, util_model, cfg)
        profiles = get_unfinished_profiles(profiles, cfg.task_config.max_num_iterations)

    profiles = orig_profiles
    profiles = get_unfinished_profiles(profiles, cfg.task_config.max_num_iterations + 2)

    infer_attributes(profiles, inf_model, cfg)
    if cfg.task_config.eval_inference_model is not None:
        infer_attributes(profiles, eval_model, cfg)


def run_eval_inference(cfg: Config) -> None:
    """Runs a specific model over all profiles and stores the results

    Args:
        cfg (Config): _description_
    """

    eval_model = get_model(cfg.task_config.eval_inference_model)
    assert isinstance(cfg.task_config, AnonymizationConfig)
    profiles = load_data(cfg.task_config.profile_path)

    # Load profiles correctly
    out_dir = cfg.task_config.outpath
    os.makedirs(out_dir, exist_ok=True)

    # If file already exists in outdir load profile names
    if os.path.exists(f"{out_dir}/eval_inference_results.jsonl"):
        existing_profiles = load_data(f"{out_dir}/eval_inference_results.jsonl")
        existing_profile_names = [p.username for p in existing_profiles]
    else:
        existing_profile_names = []

    curr_prompts = []

    for profile in profiles:

        # if profile.username in existing_profile_names:
        #     continue

        for i, comment in enumerate(profile.comments):
            inference = comment.predictions.get(
                cfg.task_config.eval_inference_model.name, None
            )
            if (
                i == 0
                and cfg.task_config.eval_inference_model.name == "gpt-4-1106-preview"
            ):
                continue
            if (i<=4) and (i>0):
                continue
                #Skip first comment

            if inference is not None and len(inference) > 0:
                # Already has an inference
                continue
            else:
                curr_prompts += create_prompts(
                    profile,
                    cfg.task_config,
                    idx=i,
                )

    if len(curr_prompts) > 0:
        results = eval_model.predict_multi(
            curr_prompts, max_workers=cfg.max_workers, timeout=40
        )
        # Store results
        for i, result in enumerate(results):
            prompt, answer = result
            op: Profile = prompt.original_point
            assert isinstance(op, Profile)
            op.comments[prompt.comment_id].predictions[
                cfg.task_config.eval_inference_model.name
            ] = parse_answer(answer, prompt.gt)
            op.comments[prompt.comment_id].predictions[
                cfg.task_config.eval_inference_model.name
            ]["full_answer"] = answer

            # Check if all scores are present
            all_present = True
            # for comment in op.comments[1:]:
            #     if cfg.task_config.eval_inference_model.name not in comment.predictions:
            #         all_present = False
            #         break
            if cfg.task_config.eval_inference_model.name not in op.comments[0].predictions:
                all_present = False  

            if cfg.task_config.eval_inference_model.name not in op.comments[5].predictions:
                all_present = False  

            if all_present:
                with open(f"{out_dir}/eval_inference_results.jsonl", "a") as f:
                    f.write(json.dumps(op.to_json()) + "\n")
                    f.flush()
def run_eval_inference_all(cfg: Config) -> None:
    """Runs a specific model over all profiles and stores the results

    Args:
        cfg (Config): _description_
    """

    eval_model = get_model(cfg.task_config.eval_inference_model)
    assert isinstance(cfg.task_config, AnonymizationConfig)
    profiles = load_data(cfg.task_config.profile_path)

    # Load profiles correctly
    out_dir = cfg.task_config.outpath
    os.makedirs(out_dir, exist_ok=True)

    # If file already exists in outdir load profile names
    if os.path.exists(f"{out_dir}/eval_inference_results.jsonl"):
        existing_profiles = load_data(f"{out_dir}/eval_inference_results.jsonl")
        existing_profile_names = [p.username for p in existing_profiles]
    else:
        existing_profile_names = []

    curr_prompts = []

    for profile in profiles:

        # if profile.username in existing_profile_names:
        #     continue

        for i, comment in enumerate(profile.comments):
            inference = comment.predictions.get(
                cfg.task_config.eval_inference_model.name, None
            )
            if (
                i == 0
                and cfg.task_config.eval_inference_model.name == "gpt-4-1106-preview"
            ):
                continue
            # if (i<=4) and (i>0):
            #     continue
                #Skip first comment

            if inference is not None and len(inference) > 0:
                # Already has an inference
                continue
            else:
                curr_prompts += create_prompts(
                    profile,
                    cfg.task_config,
                    idx=i,
                )

    if len(curr_prompts) > 0:
        results = eval_model.predict_multi(
            curr_prompts, max_workers=cfg.max_workers, timeout=40
        )
        # Store results
        for i, result in enumerate(results):
            prompt, answer = result
            op: Profile = prompt.original_point
            assert isinstance(op, Profile)
            op.comments[prompt.comment_id].predictions[
                cfg.task_config.eval_inference_model.name
            ] = parse_answer(answer, prompt.gt)
            op.comments[prompt.comment_id].predictions[
                cfg.task_config.eval_inference_model.name
            ]["full_answer"] = answer

            # Check if all scores are present
            all_present = True
            for comment in op.comments[1:]:
                if cfg.task_config.eval_inference_model.name not in comment.predictions:
                    all_present = False
                    break
            # if cfg.task_config.eval_inference_model.name not in op.comments[0].predictions:
            #     all_present = False  

            # if cfg.task_config.eval_inference_model.name not in op.comments[5].predictions:
            #     all_present = False  

            if all_present:
                with open(f"{out_dir}/eval_inference_results.jsonl", "a") as f:
                    f.write(json.dumps(op.to_json()) + "\n")
                    f.flush()
def run_utility_scoring(cfg: Config) -> None:
    """Runs a specific utility model over all profiles and stores the results

    Args:
        cfg (Config): _description_
    """

    utility_model = get_model(cfg.task_config.utility_model)
    assert isinstance(cfg.task_config, AnonymizationConfig)
    profiles = load_data(cfg.task_config.profile_path)

    # Load profiles correctly
    out_dir = cfg.task_config.outpath
    os.makedirs(out_dir, exist_ok=True)

    curr_prompts = []

    for profile in profiles:
        for i, comment in enumerate(profile.comments):
            inference = comment.utility.get(cfg.task_config.utility_model.name, None)
            if i == 0:
                # Skip first comment
                continue
            if inference is not None and len(inference) > 0:
                # Already has an inference
                continue

            curr_prompts += score_anonymization_utility_prompt(
                profile, cfg.task_config, i
            )

    if len(curr_prompts) > 0:
        results = utility_model.predict_multi(
            curr_prompts, max_workers=cfg.max_workers, timeout=60
        )
        # Store results
        for i, result in enumerate(results):
            prompt, answer = result
            op = prompt.original_point
            assert isinstance(op, Profile)

            answer = parse_utility_answer(answer)

            op.comments[prompt.comment_id].utility[
                cfg.task_config.utility_model.name
            ] = answer

            # Compute simple scores

            # BLEU
            original_comments = op.get_original_comments().comments
            comments = op.comments[prompt.comment_id].comments

            bleu = compute_bleu(
                "\n".join([str(c) for c in original_comments]),
                "\n".join([str(c) for c in comments]),
            )
            op.comments[prompt.comment_id].utility[cfg.task_config.utility_model.name][
                "bleu"
            ] = bleu

            # ROUGE
            rouge = compute_rouge(
                "\n".join([str(c) for c in original_comments]),
                ["\n".join([str(c) for c in comments])],
            )
            op.comments[prompt.comment_id].utility[cfg.task_config.utility_model.name][
                "rouge"
            ] = rouge

            # Check if all scores are present
            all_present = True
            for comment in op.comments[1:]:
                if cfg.task_config.utility_model.name not in comment.utility:
                    all_present = False
                    break

            if all_present:
                with open(f"{out_dir}/utility_results.jsonl", "a") as f:
                    f.write(json.dumps(op.to_json()) + "\n")
                    f.flush()

    if True:
        for profile in profiles:
            for i, comment in enumerate(profile.comments):

                if i == 0:
                    # Skip first comment
                    continue

                # Compute simple scores
                original_comments = profile.get_original_comments().comments
                comments = profile.comments[i].comments

                bleu = compute_bleu(
                    "\n".join([str(c) for c in original_comments]),
                    "\n".join([str(c) for c in comments]),
                )

                if "bleu" not in comment.utility[cfg.task_config.utility_model.name]:
                    comment.utility[cfg.task_config.utility_model.name]["bleu"] = bleu

                if "rouge" not in comment.utility[cfg.task_config.utility_model.name]:
                    rouge = compute_rouge(
                        "\n".join([str(c) for c in original_comments]),
                        ["\n".join([str(c) for c in comments])],
                    )
                    comment.utility[cfg.task_config.utility_model.name]["rouge"] = rouge

            all_present = True
            for comment in profile.comments[1:]:
                if cfg.task_config.utility_model.name not in comment.utility:
                    all_present = False
                    break
                elif "bleu" not in comment.utility[cfg.task_config.utility_model.name]:
                    all_present = False
                    break
                elif "rouge" not in comment.utility[cfg.task_config.utility_model.name]:
                    all_present = False
                    break

            with open(f"{out_dir}/utility_results_br.jsonl", "a") as f:
                f.write(json.dumps(profile.to_json()) + "\n")
                f.flush()
