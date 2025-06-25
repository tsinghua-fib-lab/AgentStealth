from abc import ABC, abstractmethod
from typing import List, Iterator, Dict, Any
from copy import deepcopy
from .anonymizer import Anonymizer
import json
from src.utils.string_utils import select_closest
from src.configs import Config, AnonymizerConfig
from src.prompts import Prompt
from src.reddit.reddit_utils import type_to_str
from src.reddit.reddit_types import Profile
from src.models.model import BaseModel
from src.reddit.reddit_types import Comment, AnnotatedComments
import re
import os
from datetime import datetime
import random
class LLMFullAnonymizer(Anonymizer):
    def __init__(
        self,
        cfg: AnonymizerConfig,
        model: BaseModel,
    ):
        self.model = model
        self.cfg = cfg

    def _create_anon_prompt(self, profile: Profile) -> List[Prompt]:

        if self.cfg.prompt_level == 1:  # Naive
            system_prompt = "You are an helpful assistant that helps users to anonymize their comments."
            header = "Below I give you a set of comments from a user as well as personal inferences made about the user that can be made from the comments. Your task is to anonymize the comments."
            footer = "In a new line write first a single # and then return the anonymized text."
        elif self.cfg.prompt_level == 2:  # Better prompt and explanation
            system_prompt = "You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the instructions and format precisely and you try to change as little as possible, keeping the original text in tact as much as possible."
            header = "Below I give you a set of comments from a user as well as personal inferences made about the user that can be made from the comments. Your task is to anonymize the comments to prevent these inferences, while keeping the rest of the text intact."
            footer = "In a new line write first a single # and then return the anonymized text. Only change relevant parts and keep everything else as is. Make sure you keep the original meaning, i.e. do not invent new information instead generalize information."
        elif self.cfg.prompt_level == 3:  # CoT
            system_prompt = "You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the instructions and format precisely and you try to change as little as possible, keeping the original text in tact as much as possible."
            header = "Below I give you a set of comments from a user as well as personal inferences made about the user that can be made from the comments. Your task is to anonymize the comments to prevent these inferences, while keeping the rest of the text intact."
            footer = "In a first step briefly explain the changes you will make. In a new line write a single # and then return the anonymized text. Only change relevant parts and keep everything else as is. Make sure you keep the original meaning, i.e. do not invent new information."

        comments = profile.get_latest_comments().comments
        comment_string = "\n".join([str(c) for c in comments])
        previous_inferences = profile.get_latest_comments().predictions[
            self.model.config.name
        ]
        inference_string = ""
        try:
            for key, inf in previous_inferences.items():
                if key == "full_answer":
                    continue
                if "guess" not in inf:
                    continue
                inference_string += f"Type: {key}\n"
                inference_string += f"Inference: {inf['inference']}\n"
                inference_string += f"Guess: {inf['guess']}\n"
        except Exception as e:
            # Fall back to full answer
            inference_string = previous_inferences["full_answer"]

        intermediate = f"\n\n {comment_string}\n\nInferences:\n\n{inference_string}"

        prompt = Prompt(
            system_prompt=system_prompt,
            header=header,
            intermediate=intermediate,
            footer=footer,
            target=(
                profile.get_relevant_pii()[0]
                if len(profile.get_relevant_pii()) > 0
                else ""
            ),
            original_point=profile,  # type: ignore
            gt=profile.get_relevant_pii(),  # type: ignore
            answer="",
            shots=[],
            id=profile.username,  # type: ignore
        )

        return [prompt]

    def filter_and_align_comments(self, answer: str, op: Profile) -> List[str]:

        try:
            split_answer = answer.split("\n#")

            if len(split_answer) == 1:
                new_comments = answer.strip()
            elif len(split_answer) == 2:
                new_comments = split_answer[1].strip()
            else:
                new_comments = ("\n").join(split_answer)

        except Exception:
            print("Could not split answer", answer)
            new_comments = deepcopy([c.text for c in op.get_latest_comments().comments])
            return new_comments

        new_comments = new_comments.split("\n")

        # Remove all lines that are empty
        new_comments = [c for c in new_comments if len(c) > 0]

        if len(new_comments) != len(op.get_latest_comments().comments):
            print(
                f"Number of comments does not match: {len(new_comments)} vs {len(op.get_latest_comments().comments)}"
            )

            old_comment_ids = [
                -1 for _ in range(len(op.get_latest_comments().comments))
            ]

            used_idx = set({})

            for i, comment in enumerate(op.get_latest_comments().comments):
                closest_match, sim, idx = select_closest(
                    comment.text,
                    new_comments,
                    dist="jaro_winkler",
                    return_idx=True,
                    return_sim=True,
                )

                if idx not in used_idx and sim > 0.5:
                    old_comment_ids[i] = idx
                    used_idx.add(idx)

            selected_comments = []
            for i, idx in enumerate(old_comment_ids):
                if idx == -1:
                    selected_comments.append(op.get_latest_comments().comments[i].text)
                else:
                    selected_comments.append(new_comments[idx])
        else:
            selected_comments = new_comments

        typed_comments = []
        i = 0

        for comment in selected_comments:
            if re.search(r"\d{4}-\d{2}-\d{2}:", comment[:11]) is not None:
                comment = comment[11:].strip()

            old_com = op.get_latest_comments().comments[i]
            new_com = Comment(
                comment, old_com.subreddit, old_com.user, old_com.timestamp
            )
            typed_comments.append(new_com)
            i += 1

        return typed_comments

    def anonymize(self, text: str) -> str:
        pass

    def anonymize_profiles(self, profiles: List[Profile]) -> Iterator[Profile]:

        prompts = []
        for profile in profiles:
            prompts.extend(self._create_anon_prompt(profile))

        for i, res in enumerate(
            self.model.predict_multi(
                prompts, max_workers=self.cfg.max_workers, timeout=120
            )
        ):
            prompt, answer = res
            # for i, prompt in enumerate(prompts):
            #     answer = self.model.predict(prompt)

            op = prompt.original_point
            assert isinstance(op, Profile)
            print(f"{i}".center(50, "="))
            print(prompt.get_prompt())
            op.print_review_pii()
            print(f"{self.model.config.name}\n" + answer)

            typed_comments = self.filter_and_align_comments(answer, op)
            
            print(typed_comments)

            op.comments.append(AnnotatedComments(typed_comments, op.review_pii, {}, {}))

            yield op


class LLMBaselineAnonymizer(LLMFullAnonymizer):

    def _create_anon_prompt(self, profile: Profile) -> List[Prompt]:

        type_str = ""
        for pii in profile.get_relevant_pii():
            type_str += f"{type_to_str(pii)} "

        system_prompt = "You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the instructions and format precisely and you try to change as little as possible, keeping the original text in tact as much as possible."
        header = f"Below I give you a set of comments from a user. Your task is to anonymize the comments to prevent the inferences of the users {type_str} , while keeping the rest of the text intact."
        footer = "In a first step briefly explain the changes you will make. In a new line write a single # and then return the anonymized text. Only change relevant parts and keep everything else as is. Make sure you keep the original meaning, i.e. do not invent new information."

        comments = profile.get_latest_comments().comments
        comment_string = "\n".join([str(c) for c in comments])
        # previous_infereces = profile.get_latest_comments().predictions[
        #     self.model.config.name
        # ]

        intermediate = f"\n\n {comment_string}\n"

        prompt = Prompt(
            system_prompt=system_prompt,
            header=header,
            intermediate=intermediate,
            footer=footer,
            target= (profile.get_relevant_pii()[0]
                if len(profile.get_relevant_pii()) > 0
                else ""
            ),
            original_point=profile,  # type: ignore
            gt=profile.get_relevant_pii(),  # type: ignore
            answer="",
            shots=[],
            id=profile.username,  # type: ignore
        )

        return [prompt]
class LLMBaselineAnonymizer2(LLMFullAnonymizer):

    def _create_anon_prompt(self, profile: Profile) -> List[Prompt]:

        type_str = ""
        with open(self.cfg.summary_path, "r",encoding="utf-8") as f:
            feature_summaries = json.load(f)
        pii_summary=""    
        if "PAI" in self.cfg.summary_path:
            for pii in profile.get_relevant_pii():
                type_str += f"{type_to_str(pii)} "
                pii_summary+=f"{type_to_str(pii)}: \n{feature_summaries[pii]['summary']}\n"
        else:
            for pii in profile.get_relevant_pii():
                type_str += f"{type_to_str(pii)} "
                if "age" in pii:
                    pii_summary+=f"{type_to_str(pii)}: \n{feature_summaries['age']['summary']}\n"
                elif "gender" in pii:
                    pii_summary+=f"{type_to_str(pii)}: \n{feature_summaries['sex']['summary']}\n"
                elif "location" in pii:
                    pii_summary+=f"{type_to_str(pii)}: \n{feature_summaries['city_country']['summary']}\n"    
                elif "pobp" in pii:
                    pii_summary+=f"{type_to_str(pii)}: \n{feature_summaries['birth_city_country']['summary']}\n"  
                elif "income" in pii:
                    pii_summary+=f"{type_to_str(pii)}: \n{feature_summaries['income_level']['summary']}\n"
                elif "education" in pii:
                    pii_summary+=f"{type_to_str(pii)}: \n{feature_summaries['education']['summary']}\n"
                elif "occupation" in pii:
                    pii_summary+=f"{type_to_str(pii)}: \n{feature_summaries['occupation']['summary']}\n"
                elif "married" in pii:   
                    pii_summary+=f"{type_to_str(pii)}: \n{feature_summaries['relationship_status']['summary']}\n"           


        system_prompt = "You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the instructions and format precisely and you try to change as little as possible, keeping the original text in tact as much as possible."
        header = f"Below I give you a set of comments from a user. Your task is to anonymize the comments to prevent the inferences of the users {type_str} , while keeping the rest of the text intact. For inference, I'll give you some summary of the inference. You can use this summary to help you anonymize the comments. The summary is: {pii_summary}."
        footer = "In a first step briefly explain the changes you will make. In a new line write a single # and then return the anonymized text. Only change relevant parts and keep everything else as is. Make sure you keep the original meaning, i.e. do not invent new information."

        comments = profile.get_latest_comments().comments
        comment_string = "\n".join([str(c) for c in comments])
        # previous_infereces = profile.get_latest_comments().predictions[
        #     self.model.config.name
        # ]

        intermediate = f"\n\n {comment_string}\n"

        prompt = Prompt(
            system_prompt=system_prompt,
            header=header,
            intermediate=intermediate,
            footer=footer,
            target= (profile.get_relevant_pii()[0]
                if len(profile.get_relevant_pii()) > 0
                else ""
            ),
            original_point=profile,  # type: ignore
            gt=profile.get_relevant_pii(),  # type: ignore
            answer="",
            shots=[],
            id=profile.username,  # type: ignore
        )

        return [prompt]
class LLMBaselineAnonymizer3(LLMFullAnonymizer):

    def _create_anon_prompt(self, profile: Profile) -> List[Prompt]:
        type_str = ""
        with open(self.cfg.summary_path, "r",encoding="utf-8") as f:
            feature_summaries = json.load(f)
        pii_summary=""    
        if "PAI" in self.cfg.summary_path:
            for pii in profile.get_relevant_pii():
                type_str += f"{type_to_str(pii)} "
                pii_summary+=f"{type_to_str(pii)}: \n{feature_summaries[pii]['summary']}\n"
        else:
            for pii in profile.get_relevant_pii():
                type_str += f"{type_to_str(pii)} "
                if "age" in pii:
                    pii_summary+=f"{type_to_str(pii)}: \n{feature_summaries['age']['summary']}\n"
                elif "gender" in pii:
                    pii_summary+=f"{type_to_str(pii)}: \n{feature_summaries['sex']['summary']}\n"
                elif "location" in pii:
                    pii_summary+=f"{type_to_str(pii)}: \n{feature_summaries['city_country']['summary']}\n"    
                elif "pobp" in pii:
                    pii_summary+=f"{type_to_str(pii)}: \n{feature_summaries['birth_city_country']['summary']}\n"  
                elif "income" in pii:
                    pii_summary+=f"{type_to_str(pii)}: \n{feature_summaries['income_level']['summary']}\n"
                elif "education" in pii:
                    pii_summary+=f"{type_to_str(pii)}: \n{feature_summaries['education']['summary']}\n"
                elif "occupation" in pii:
                    pii_summary+=f"{type_to_str(pii)}: \n{feature_summaries['occupation']['summary']}\n"
                elif "married" in pii:   
                    pii_summary+=f"{type_to_str(pii)}: \n{feature_summaries['relationship_status']['summary']}\n"           



        if self.cfg.prompt_level == 1:  # Naive
            system_prompt = "You are an helpful assistant that helps users to anonymize their comments."
            header = "Below I give you a set of comments from a user as well as personal inferences made about the user that can be made from the comments. Your task is to anonymize the comments."
            footer = "In a new line write first a single # and then return the anonymized text."
        elif self.cfg.prompt_level == 2:  # Better prompt and explanation
            system_prompt = "You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the instructions and format precisely and you try to change as little as possible, keeping the original text in tact as much as possible."
            header = "Below I give you a set of comments from a user as well as personal inferences made about the user that can be made from the comments. Your task is to anonymize the comments to prevent these inferences, while keeping the rest of the text intact."
            footer = "In a new line write first a single # and then return the anonymized text. Only change relevant parts and keep everything else as is. Make sure you keep the original meaning, i.e. do not invent new information instead generalize information."
        elif self.cfg.prompt_level == 3:  # CoT
            system_prompt = "You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the instructions and format precisely and you try to change as little as possible, keeping the original text in tact as much as possible."
            header = f"Below I give you a set of comments from a user as well as personal inferences made about the user that can be made from the comments. Your task is to anonymize the comments to prevent these inferences, while keeping the rest of the text intact.For inference, I'll give you some summary of the inference. You can use this summary to help you anonymize the comments. The summary is: {pii_summary}."
            footer = "In a first step briefly explain the changes you will make. In a new line write a single # and then return the anonymized text. Only change relevant parts and keep everything else as is. Make sure you keep the original meaning, i.e. do not invent new information."

           
        comments = profile.get_latest_comments().comments
        comment_string = "\n".join([str(c) for c in comments])
        previous_inferences = profile.get_latest_comments().predictions[
            self.model.config.name
        ]
        inference_string = ""
        try:
            for key, inf in previous_inferences.items():
                if key == "full_answer":
                    continue
                if "guess" not in inf:
                    continue
                inference_string += f"Type: {key}\n"
                inference_string += f"Inference: {inf['inference']}\n"
                inference_string += f"Guess: {inf['guess']}\n"
        except Exception as e:
            # Fall back to full answer
            inference_string = previous_inferences["full_answer"]

        intermediate = f"\n\n {comment_string}\n\nInferences:\n\n{inference_string}"

        prompt = Prompt(
            system_prompt=system_prompt,
            header=header,
            intermediate=intermediate,
            footer=footer,
            target=(
                profile.get_relevant_pii()[0]
                if len(profile.get_relevant_pii()) > 0
                else ""
            ),
            original_point=profile,  # type: ignore
            gt=profile.get_relevant_pii(),  # type: ignore
            answer="",
            shots=[],
            id=profile.username,  # type: ignore
        )

        return [prompt]
class LLMSummaryAnonymizer(LLMFullAnonymizer):

    def _create_anon_prompt(self, profile: Profile) -> List[Prompt]:
        type_str = ""
        with open(self.cfg.summary_path, "r",encoding="utf-8") as f:
            feature_summaries = json.load(f)
        pii_summary=""   
        for pii in profile.get_relevant_pii():
            type_str += f"{type_to_str(pii)} "
            summaries=feature_summaries.get(pii, {})
            if "success_summary" in summaries :
                pii_summary += f"\nFor {type_str}:\nSuccess anonymization summaries:\n{summaries['success_summary']}\n"
            if "failure_summary" in summaries:
                pii_summary += f"\nFor {type_str}:\nFailure anonymization summaries:\n{summaries['failure_summary']}\n"
            if "success_summary" not in summaries:
                pii_summary += f"\nFor {type_str}:No success summaries available.\n"
            if "failure_summary" not in summaries:
                pii_summary += f"\nFor {type_str}:No failure summaries available.\n"
        if self.cfg.prompt_level == 1:  # Naive
            system_prompt = "You are an helpful assistant that helps users to anonymize their comments."
            header = "Below I give you a set of comments from a user as well as personal inferences made about the user that can be made from the comments. Your task is to anonymize the comments."
            footer = "In a new line write first a single # and then return the anonymized text."
        elif self.cfg.prompt_level == 2:  # Better prompt and explanation
            system_prompt = "You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the instructions and format precisely and you try to change as little as possible, keeping the original text in tact as much as possible."
            header = "Below I give you a set of comments from a user as well as personal inferences made about the user that can be made from the comments. Your task is to anonymize the comments to prevent these inferences, while keeping the rest of the text intact."
            footer = "In a new line write first a single # and then return the anonymized text. Only change relevant parts and keep everything else as is. Make sure you keep the original meaning, i.e. do not invent new information instead generalize information."
        elif self.cfg.prompt_level == 3:  # CoT
            system_prompt = "You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the instructions and format precisely and you try to change as little as possible, keeping the original text in tact as much as possible."
            header = f"Below I give you a set of comments from a user as well as personal inferences made about the user that can be made from the comments. Your task is to anonymize the comments to prevent these inferences, while keeping the rest of the text intact.For inference, I'll give you some summary of successful and unsuccessful inference. You can use this summary to help you anonymize the comments. The summary is: {pii_summary}."
            footer = "In a first step briefly explain the changes you will make. In a new line write a single # and then return the anonymized text. Only change relevant parts and keep everything else as is. Make sure you keep the original meaning, i.e. do not invent new information."

           
        comments = profile.get_latest_comments().comments
        comment_string = "\n".join([str(c) for c in comments])
        previous_inferences = profile.get_latest_comments().predictions[
            self.model.config.name
        ]
        inference_string = ""
        try:
            for key, inf in previous_inferences.items():
                if key == "full_answer":
                    continue
                if "guess" not in inf:
                    continue
                inference_string += f"Type: {key}\n"
                inference_string += f"Inference: {inf['inference']}\n"
                inference_string += f"Guess: {inf['guess']}\n"
        except Exception as e:
            # Fall back to full answer
            inference_string = previous_inferences["full_answer"]

        intermediate = f"\n\n {comment_string}\n\nInferences:\n\n{inference_string}"

        prompt = Prompt(
            system_prompt=system_prompt,
            header=header,
            intermediate=intermediate,
            footer=footer,
            target=(
                profile.get_relevant_pii()[0]
                if len(profile.get_relevant_pii()) > 0
                else ""
            ),
            original_point=profile,  # type: ignore
            gt=profile.get_relevant_pii(),  # type: ignore
            answer="",
            shots=[],
            id=profile.username,  # type: ignore
        )

        return [prompt]
class LLMStyleAnonymizer(LLMFullAnonymizer):

    def _create_anon_prompt(self, profile: Profile) -> List[Prompt]:

        with open(self.cfg.style_path, "r",encoding="utf-8") as f:
            data= [json.loads(line) for line in f]
        piis=profile.get_pii_gt()
        style_examples="---------Examples Start---------\n"
        num=0
        indices = list(range(len(data)))
        random.shuffle(indices)
        for i in indices:
            per=data[i]
            per_pii={}
            reviews=per.get("reviews", {})
            for key, value in reviews.items():
                for pii_type,pii_value in value.items():                    
                    if pii_type in piis:
                        per_pii[pii_type] = pii_value["estimate"]
            if len(per_pii) > len(piis)*2/3:
                com=per["comments"][0]["comments"]
                w=[]
                for item in com:
                    time=item["timestamp"] if isinstance(item["timestamp"], datetime) else datetime.fromtimestamp(int(float(item["timestamp"])))
                    w.append(f"{time.strftime('%Y-%m-%d')}: {item['text']}")
                comments="\n".join(w) 
                pii_str="" 
                for key, value in per_pii.items():
                    pii_str+=f"{type_to_str(key)}: {value}\n"
                style_examples+=f"Piis:\n{pii_str}Comments:\n{comments}\n"        
                num+=1
            if num>=5:
                break
        if num==0:
            style_examples="Cannot find any examples."   
        else:
            style_examples+="---------Examples End---------\n"               
        if self.cfg.prompt_level == 1:  # Naive
            system_prompt = "You are an helpful assistant that helps users to anonymize their comments."
            header = "Below I give you a set of comments from a user as well as personal inferences made about the user that can be made from the comments. Your task is to anonymize the comments."
            footer = "In a new line write first a single # and then return the anonymized text."
        elif self.cfg.prompt_level == 2:  # Better prompt and explanation
            system_prompt = "You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the instructions and format precisely and you try to change as little as possible, keeping the original text in tact as much as possible."
            header = "Below I give you a set of comments from a user as well as personal inferences made about the user that can be made from the comments. Your task is to anonymize the comments to prevent these inferences, while keeping the rest of the text intact."
            footer = "In a new line write first a single # and then return the anonymized text. Only change relevant parts and keep everything else as is. Make sure you keep the original meaning, i.e. do not invent new information instead generalize information."
        elif self.cfg.prompt_level == 3:  # CoT
            system_prompt = "You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the instructions and format precisely and you try to change as little as possible, keeping the original text in tact as much as possible."
            header = f"Below I give you a set of comments from a user as well as personal inferences made about the user that can be made from the comments. Your task is to anonymize the comments to prevent these inferences, while keeping the rest of the text intact.For inference, I'll give you examples of texts with same piis and different values. You can use these examples to help you anonymize the comments, maybe change the style. The examples are: {style_examples}."
            footer = "In a first step briefly explain the changes you will make. In a new line write a single # and then return the anonymized text. Only change relevant parts and keep everything else as is. Make sure you keep the original meaning, i.e. do not invent new information."

           
        comments = profile.get_latest_comments().comments
        comment_string = "\n".join([str(c) for c in comments])
        previous_inferences = profile.get_latest_comments().predictions[
            self.model.config.name
        ]
        inference_string = ""
        try:
            for key, inf in previous_inferences.items():
                if key == "full_answer":
                    continue
                if "guess" not in inf:
                    continue
                inference_string += f"Type: {key}\n"
                inference_string += f"Inference: {inf['inference']}\n"
                inference_string += f"Guess: {inf['guess']}\n"
        except Exception as e:
            # Fall back to full answer
            inference_string = previous_inferences["full_answer"]

        intermediate = f"\n\n {comment_string}\n\nInferences:\n\n{inference_string}"

        prompt = Prompt(
            system_prompt=system_prompt,
            header=header,
            intermediate=intermediate,
            footer=footer,
            target=(
                profile.get_relevant_pii()[0]
                if len(profile.get_relevant_pii()) > 0
                else ""
            ),
            original_point=profile,  # type: ignore
            gt=profile.get_relevant_pii(),  # type: ignore
            answer="",
            shots=[],
            id=profile.username,  # type: ignore
        )

        return [prompt]
class LLMCodingAnonymizer(LLMFullAnonymizer):

    def _create_anon_prompt(self, profile: Profile) -> List[Prompt]:
        type_str = ""
        if os.path.exists(self.cfg.summary_path):
            with open(self.cfg.summary_path, "r",encoding="utf-8") as f:
                feature_insights = json.load(f)
        else:
            feature_insights = {}    
        pii_insights=""   
        for pii in profile.get_relevant_pii():

            insights=feature_insights.get(pii, {}).get("insights", [])
            if len(insights) >0:
                insight="\n".join(insights)
                pii_insights += f"\nFor {pii}:\nThe insights for anonymization:\n{insight}\n"
            else:
                pii_insights += f"\nFor {pii}:\nNo insights available.\n"    
            
        if self.cfg.prompt_level == 1:  # Naive
            system_prompt = "You are an helpful assistant that helps users to anonymize their comments."
            header = "Below I give you a set of comments from a user as well as personal inferences made about the user that can be made from the comments. Your task is to anonymize the comments."
            footer = "In a new line write first a single # and then return the anonymized text."
        elif self.cfg.prompt_level == 2:  # Better prompt and explanation
            system_prompt = "You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the instructions and format precisely and you try to change as little as possible, keeping the original text in tact as much as possible."
            header = "Below I give you a set of comments from a user as well as personal inferences made about the user that can be made from the comments. Your task is to anonymize the comments to prevent these inferences, while keeping the rest of the text intact."
            footer = "In a new line write first a single # and then return the anonymized text. Only change relevant parts and keep everything else as is. Make sure you keep the original meaning, i.e. do not invent new information instead generalize information."
        elif self.cfg.prompt_level == 3:  # CoT
            system_prompt = "You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the instructions and format precisely and you try to change as little as possible, keeping the original text in tact as much as possible."
            header1 = f"""Below I give you a set of comments from a user as well as personal inferences made about the user that can be made from the comments. Your task is to anonymize the comments to prevent these inferences, while keeping the rest of the text intact.
            For reference, I'll give you some insights to help you anonimize the comments. 
            The comments may have already been anonymized, so I'll also provide you with the original comments and some utility scores (all ranging from 0-1). (If the scores are > 0.9, it means that utility is well preserved, if the score is between 0.65 and 0.9, it means the utility is acceptable ,if the scores are < 0.65, it means that utility is not well preserved).
            If you find the utility scores are not so high, maybe you should not change the current comments too much. 
            Note that the utility scores are not always accurate, so you should also consider the comments themselves and make changes on the current comments rather than the original comments.
            
            

            """
            header2=f"""Below I give you a set of comments from a user as well as personal inferences made about the user that can be made from the comments. Your task is to anonymize the comments to prevent these inferences, while keeping the rest of the text intact.
            For reference, I'll give you some insights to help you anonimize the comments. 
            
            

            """
            footer = "In a first step briefly explain the changes you will make. In a new line write a single # and then return the anonymized text. Only change relevant parts and keep everything else as is. Make sure you keep the original meaning, i.e. do not invent new information."

           
        comments = profile.get_latest_comments().comments
        res={}
        model_utility = profile.get_latest_comments().utility
        if model_utility:
            key = next(iter(model_utility))
            model_utility=model_utility[key]
        if "bleu" in model_utility:
            res["bleu"] = model_utility["bleu"]
        if "rouge" in model_utility:
            res["rouge1"] = model_utility["rouge"][0]["rouge1"].fmeasure
            res["rougeL"] = model_utility["rouge"][0]["rougeL"].fmeasure
        utility=f"bleu: {res.get('bleu', 1)}\nrouge1: {res.get('rouge1', 1)}\nrougeL: {res.get('rougeL', 1)}\n"    
        comment_string = "\n".join([str(c) for c in comments])
        ori=profile.get_original_comments().comments
        original_comment_string = "\n".join([str(c) for c in ori])
        previous_inferences = profile.get_latest_comments().predictions[
            self.model.config.name
        ]
        inference_string = ""
        try:
            for key, inf in previous_inferences.items():
                if key == "full_answer":
                    continue
                if "guess" not in inf:
                    continue
                inference_string += f"Type: {key}\n"
                inference_string += f"Inference: {inf['inference']}\n"
                inference_string += f"Guess: {inf['guess']}\n"
        except Exception as e:
            # Fall back to full answer
            inference_string = previous_inferences["full_answer"]
        if len(profile.comments)==2 or len(profile.comments)==3 or len(profile.comments)==5:
            header=header1
            intermediate = f"\n\nOriginal comments:\n{original_comment_string}\n\nCurrent comments:\n{comment_string}\n \nInferences:\n\n{inference_string}\n Utility scores:\n\n{utility}\n\nInsights:\n\n{pii_insights}"
        else:
            header=header2
            intermediate = f"\n\nCurrent comments:\n{comment_string}\n \nInferences:\n\n{inference_string}\nInsights:\n\n{pii_insights}"
        prompt = Prompt(
            system_prompt=system_prompt,
            header=header,
            intermediate=intermediate,
            footer=footer,
            target=(
                profile.get_relevant_pii()[0]
                if len(profile.get_relevant_pii()) > 0
                else ""
            ),
            original_point=profile,  # type: ignore
            gt=profile.get_relevant_pii(),  # type: ignore
            answer="",
            shots=[],
            id=profile.username,  # type: ignore
        )

        return [prompt]       
    def anonymize_profiles(self, profiles: List[Profile]) -> Iterator[Profile]:

        prompts = []
        for profile in profiles:
            prompts.extend(self._create_anon_prompt(profile))

        for i, res in enumerate(
            self.model.predict_multi(
                prompts, max_workers=self.cfg.max_workers, timeout=120
            )
        ):
            prompt, answer = res
            # for i, prompt in enumerate(prompts):
            #     answer = self.model.predict(prompt)

            op = prompt.original_point
            assert isinstance(op, Profile)
            print(f"{i}".center(50, "="))
            print(prompt.get_prompt())
            op.print_review_pii()
            print(f"{self.model.config.name}\n" + answer)

            typed_comments = self.filter_and_align_comments(answer, op)
            output_entry = {
                "username": op.username,
                "answer": answer,
                "length": len(op.comments)-1,
            }

            with open(f"ano_outputs.jsonl", "a") as f:
                f.write(json.dumps(output_entry) + "\n")
                f.flush()
            print(typed_comments)

            op.comments.append(AnnotatedComments(typed_comments, op.review_pii, {}, {}))

            yield op     
class LLMCodingAnonymizer1(LLMFullAnonymizer):

    def _create_anon_prompt(self, profile: Profile) -> List[Prompt]:
        type_str = ""
        if os.path.exists(self.cfg.summary_path):
            with open(self.cfg.summary_path, "r",encoding="utf-8") as f:
                feature_insights = json.load(f)
        else:
            feature_insights = {}    
        pii_insights=""   
        for pii in profile.get_relevant_pii():

            insights=feature_insights.get(pii, {}).get("insights", [])
            if len(insights) >0:
                insight="\n".join(insights)
                pii_insights += f"\nFor {pii}:\nThe insights for anonymization:\n{insight}\n"
            else:
                pii_insights += f"\nFor {pii}:\nNo insights available.\n"    
            
        if self.cfg.prompt_level == 1:  # Naive
            system_prompt = "You are an helpful assistant that helps users to anonymize their comments."
            header = "Below I give you a set of comments from a user as well as personal inferences made about the user that can be made from the comments. Your task is to anonymize the comments."
            footer = "In a new line write first a single # and then return the anonymized text."
        elif self.cfg.prompt_level == 2:  # Better prompt and explanation
            system_prompt = "You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the instructions and format precisely and you try to change as little as possible, keeping the original text in tact as much as possible."
            header = "Below I give you a set of comments from a user as well as personal inferences made about the user that can be made from the comments. Your task is to anonymize the comments to prevent these inferences, while keeping the rest of the text intact."
            footer = "In a new line write first a single # and then return the anonymized text. Only change relevant parts and keep everything else as is. Make sure you keep the original meaning, i.e. do not invent new information instead generalize information."
        elif self.cfg.prompt_level == 3:  # CoT
            system_prompt = "You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the instructions and format precisely and you try to change as little as possible, keeping the original text in tact as much as possible."
            header1 = f"""Below I give you a set of comments from a user as well as personal inferences made about the user that can be made from the comments. Your task is to anonymize the comments to prevent these inferences, while keeping the rest of the text intact.
            For reference, I'll give you some insights to help you anonimize the comments. 
            The comments may have already been anonymized, so I'll also provide you with the original comments and some utility scores (all ranging from 0-1). (If the scores are > 0.9, it means that utility is well preserved, if the score is between 0.65 and 0.9, it means the utility is acceptable ,if the scores are < 0.65, it means that utility is not well preserved).
            If you find the utility scores are not so high, maybe you should not change the current comments too much. 
            Note that the utility scores are not always accurate, so you should also consider the comments themselves and make changes on the current comments rather than the original comments.
            
            

            """
            header2=f"""Below I give you a set of comments from a user as well as personal inferences made about the user that can be made from the comments. Your task is to anonymize the comments to prevent these inferences, while keeping the rest of the text intact.
            For reference, I'll give you some insights to help you anonimize the comments. 
            
            

            """
            footer = "In a first step briefly explain the changes you will make. In a new line write a single # and then return the anonymized text. Only change relevant parts and keep everything else as is. Make sure you keep the original meaning, i.e. do not invent new information."

           
        comments = profile.get_latest_comments().comments
        res={}
        model_utility = profile.get_latest_comments().utility
        if model_utility:
            key = next(iter(model_utility))
            model_utility=model_utility[key]
        if "bleu" in model_utility:
            res["bleu"] = model_utility["bleu"]
        if "rouge" in model_utility:
            if isinstance(model_utility["rouge"][0]["rouge1"], list):
                res["rouge1"] = model_utility["rouge"][0]["rouge1"][2]
                res["rougeL"] = model_utility["rouge"][0]["rougeL"][2]
            else:
                res["rouge1"] = model_utility["rouge"][0]["rouge1"].fmeasure
                res["rougeL"] = model_utility["rouge"][0]["rougeL"].fmeasure    
        utility=f"bleu: {res.get('bleu', 1)}\nrouge1: {res.get('rouge1', 1)}\nrougeL: {res.get('rougeL', 1)}\n"    
        comment_string = "\n".join([str(c) for c in comments])
        ori=profile.get_original_comments().comments
        original_comment_string = "\n".join([str(c) for c in ori])
        previous_inferences = profile.get_latest_comments().predictions[
            self.model.config.args["attack_model"]
        ]
        inference_string = ""
        try:
            for key, inf in previous_inferences.items():
                if key == "full_answer":
                    continue
                if "guess" not in inf:
                    continue
                inference_string += f"Type: {key}\n"
                inference_string += f"Inference: {inf['inference']}\n"
                inference_string += f"Guess: {inf['guess']}\n"
        except Exception as e:
            # Fall back to full answer
            inference_string = previous_inferences["full_answer"]

        header=header1
        intermediate = f"\n\nOriginal comments:\n{original_comment_string}\n\nCurrent comments:\n{comment_string}\n \nInferences:\n\n{inference_string}\n Utility scores:\n\n{utility}\n\nInsights:\n\n{pii_insights}"

        prompt = Prompt(
            system_prompt=system_prompt,
            header=header,
            intermediate=intermediate,
            footer=footer,
            target=(
                profile.get_relevant_pii()[0]
                if len(profile.get_relevant_pii()) > 0
                else ""
            ),
            original_point=profile,  # type: ignore
            gt=profile.get_relevant_pii(),  # type: ignore
            answer="",
            shots=[],
            id=profile.username,  # type: ignore
        )

        return [prompt]       
    def anonymize_profiles(self, profiles: List[Profile]) -> Iterator[Profile]:

        prompts = []
        for profile in profiles:
            prompts.extend(self._create_anon_prompt(profile))

        for i, res in enumerate(
            self.model.predict_multi(
                prompts, max_workers=self.cfg.max_workers, timeout=120
            )
        ):
            prompt, answer = res
            # for i, prompt in enumerate(prompts):
            #     answer = self.model.predict(prompt)

            op = prompt.original_point
            assert isinstance(op, Profile)
            print(f"{i}".center(50, "="))
            print(prompt.get_prompt())
            op.print_review_pii()
            print(f"{self.model.config.name}\n" + answer)

            typed_comments = self.filter_and_align_comments(answer, op)
           
            print(typed_comments)

            op.comments.append(AnnotatedComments(typed_comments, op.review_pii, {}, {}))

            yield op                 
class LLMCodingAnonymizer2(LLMFullAnonymizer):

    def _create_anon_prompt(self, profile: Profile) -> List[Prompt]:
        type_str = ""
        if os.path.exists(self.cfg.summary_path):
            with open(self.cfg.summary_path, "r",encoding="utf-8") as f:
                feature_insights = json.load(f)
        else:
            feature_insights = {}    
        pii_insights=""   
        for pii in profile.get_relevant_pii():

            insights=feature_insights.get(pii, {}).get("insights", [])
            if len(insights) >0:
                insight="\n".join(insights)
                pii_insights += f"\nFor {pii}:\nThe insights for anonymization:\n{insight}\n"
            else:
                pii_insights += f"\nFor {pii}:\nNo insights available.\n"    
            
        if self.cfg.prompt_level == 1:  # Naive
            system_prompt = "You are an helpful assistant that helps users to anonymize their comments."
            header = "Below I give you a set of comments from a user as well as personal inferences made about the user that can be made from the comments. Your task is to anonymize the comments."
            footer = "In a new line write first a single # and then return the anonymized text."
        elif self.cfg.prompt_level == 2:  # Better prompt and explanation
            system_prompt = "You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the instructions and format precisely and you try to change as little as possible, keeping the original text in tact as much as possible."
            header = "Below I give you a set of comments from a user as well as personal inferences made about the user that can be made from the comments. Your task is to anonymize the comments to prevent these inferences, while keeping the rest of the text intact."
            footer = "In a new line write first a single # and then return the anonymized text. Only change relevant parts and keep everything else as is. Make sure you keep the original meaning, i.e. do not invent new information instead generalize information."
        elif self.cfg.prompt_level == 3:  # CoT
            system_prompt = "You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the instructions and format precisely and you try to change as little as possible, keeping the original text in tact as much as possible."
            header1 = f"""Below I give you a set of comments from a user as well as personal inferences made about the user that can be made from the comments. Your task is to anonymize the comments to prevent these inferences, while keeping the rest of the text intact.
            For reference, I'll give you some insights to help you anonimize the comments. 
            The comments may have already been anonymized, so I'll also provide you with the original comments and some utility scores (all ranging from 0-1). (If the scores are > 0.9, it means that utility is well preserved, if the score is between 0.65 and 0.9, it means the utility is acceptable ,if the scores are < 0.65, it means that utility is not well preserved).
            If you find the utility scores are not so high, maybe you should not change the current comments too much. 
            Note that the utility scores are not always accurate, so you should also consider the comments themselves and make changes on the current comments rather than the original comments.
            
            

            """
            header2=f"""Below I give you a set of comments from a user as well as personal inferences made about the user that can be made from the comments. Your task is to anonymize the comments to prevent these inferences, while keeping the rest of the text intact.
            For reference, I'll give you some insights to help you anonimize the comments. 
            
            

            """
            footer = "In a first step briefly explain the changes you will make. In a new line write a single # and then return the anonymized text. Only change relevant parts and keep everything else as is. Make sure you keep the original meaning, i.e. do not invent new information."

           
        comments = profile.get_latest_comments().comments
        res={}
        model_utility = profile.get_latest_comments().utility
        if model_utility:
            key = next(iter(model_utility))
            model_utility=model_utility[key]
        if "bleu" in model_utility:
            res["bleu"] = model_utility["bleu"]
        if "rouge" in model_utility:
            res["rouge1"] = model_utility["rouge"][0]["rouge1"].fmeasure
            res["rougeL"] = model_utility["rouge"][0]["rougeL"].fmeasure
        utility=f"bleu: {res.get('bleu', 1)}\nrouge1: {res.get('rouge1', 1)}\nrougeL: {res.get('rougeL', 1)}\n"    
        average_utility = (res.get('bleu', 1)+res.get('rouge1', 1)+res.get('rougeL', 1))/3
        if len(profile.comments)<2:
            pre_utility=1
        else:    
            Res={}
            model_utility = profile.comments[-2].utility
            if model_utility:
                key = next(iter(model_utility))
                model_utility=model_utility[key]
            if "bleu" in model_utility:
                Res["bleu"] = model_utility["bleu"]
            if "rouge" in model_utility:
                Res["rouge1"] = model_utility["rouge"][0]["rouge1"].fmeasure
                Res["rougeL"] = model_utility["rouge"][0]["rougeL"].fmeasure   
            pre_utility = (Res.get('bleu', 1)+Res.get('rouge1', 1)+Res.get('rougeL', 1))/3
        comment_string = "\n".join([str(c) for c in comments])
        ori=profile.get_original_comments().comments
        original_comment_string = "\n".join([str(c) for c in ori])
        previous_inferences = profile.get_latest_comments().predictions[
            self.model.config.args["attack_model"]
        ]
        inference_string = ""
        try:
            for key, inf in previous_inferences.items():
                if key == "full_answer":
                    continue
                if "guess" not in inf:
                    continue
                inference_string += f"Type: {key}\n"
                inference_string += f"Inference: {inf['inference']}\n"
                inference_string += f"Guess: {inf['guess']}\n"
        except Exception as e:
            # Fall back to full answer
            inference_string = previous_inferences["full_answer"]

        if (pre_utility-average_utility)>0.075:
            header=header1
            intermediate = f"\n\nOriginal comments:\n{original_comment_string}\n\nCurrent comments:\n{comment_string}\n \nInferences:\n\n{inference_string}\n Utility scores:\n\n{utility}\n\nInsights:\n\n{pii_insights}"
        else:
            header=header2
            intermediate = f"\n\nCurrent comments:\n{comment_string}\n \nInferences:\n\n{inference_string}\nInsights:\n\n{pii_insights}"
        prompt = Prompt(
            system_prompt=system_prompt,
            header=header,
            intermediate=intermediate,
            footer=footer,
            target=(
                profile.get_relevant_pii()[0]
                if len(profile.get_relevant_pii()) > 0
                else ""
            ),
            original_point=profile,  # type: ignore
            gt=profile.get_relevant_pii(),  # type: ignore
            answer="",
            shots=[],
            id=profile.username,  # type: ignore
        )

        return [prompt]       
    def anonymize_profiles(self, profiles: List[Profile]) -> Iterator[Profile]:

        prompts = []
        for profile in profiles:
            prompts.extend(self._create_anon_prompt(profile))

        for i, res in enumerate(
            self.model.predict_multi(
                prompts, max_workers=self.cfg.max_workers, timeout=120
            )
        ):
            prompt, answer = res
            # for i, prompt in enumerate(prompts):
            #     answer = self.model.predict(prompt)

            op = prompt.original_point
            assert isinstance(op, Profile)
            print(f"{i}".center(50, "="))
            print(prompt.get_prompt())
            op.print_review_pii()
            print(f"{self.model.config.name}\n" + answer)

            typed_comments = self.filter_and_align_comments(answer, op)
            # output_entry = {
            #     "username": op.username,
            #     "answer": answer,
            #     "length": len(op.comments)-1,
            # }

            # with open(f"ano_outputs.jsonl", "a") as f:
            #     f.write(json.dumps(output_entry) + "\n")
            #     f.flush()
            print(typed_comments)

            op.comments.append(AnnotatedComments(typed_comments, op.review_pii, {}, {}))

            yield op                             