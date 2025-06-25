from typing import List, Dict, Any, Optional, TextIO,Union
from datetime import datetime
import hashlib
import json


class Comment:
    def __init__(
        self,
        text: str,
        subreddit: str,
        user: str,
        timestamp: str,
        pii: Optional[Dict[str, Dict[str, Dict[str, str]]]] = None,
    ) -> None:
        self.text = text
        self.subreddit = subreddit
        self.user = user
        self.timestamp = (
            timestamp
            if isinstance(timestamp, datetime)
            else datetime.fromtimestamp(int(float(timestamp)))
        )
        self.pii = pii if pii is not None else {}

    def get_text(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"{self.timestamp.strftime('%Y-%m-%d')}: {self.text}"

    def to_json(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "subreddit": self.subreddit,
            "user": self.user,
            "timestamp": str(self.timestamp.timestamp()),
            "pii": self.pii,
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "Comment":
        text = data["text"]
        subreddit = data.get("subreddit", "synthetic")
        user = data["user"] if "user" in data else data["username"]
        timestamp = data.get("timestamp", datetime.now().timestamp())
        pii = data.get("pii", {})
        return cls(text, subreddit, user, timestamp, pii)

    # Hashable
    def __hash__(self) -> int:
        hash_str = self.text + self.subreddit + self.user + str(self.timestamp)
        return int(hashlib.sha1(hash_str.encode("utf-8")).hexdigest(), 16)

    # Merge two comments
    def __add__(self, other: "Comment") -> "Comment":
        assert self.__hash__() == other.__hash__(), "Comments must be the same"

        # Merge pii dicts
        new_pii = self.pii | other.pii

        return Comment(
            self.text,
            self.subreddit,
            self.user,
            str(self.timestamp.timestamp()),
            new_pii,
        )


class AnnotatedComments:
    def __init__(
        self,
        comments: List[Comment],
        review_pii: Dict[str, Dict[str, Any]],
        predictions: Optional[Dict[str, Dict[str, Any]]],
        evaluations: Optional[Dict[str, Dict[str, Dict[str, List[int]]]]] = None,
        utility: Optional[Dict[str, Dict[str, Dict[str, List[int]]]]] = None,
    ) -> None:
        self.comments = comments
        self.num_comments = len(comments)
        self.review_pii = review_pii
        self.predictions = predictions if predictions is not None else {}
        self.evaluations = evaluations if evaluations is not None else {}
        self.utility = utility if utility is not None else {}
        self.comments.sort(key=lambda c: (c.subreddit, c.timestamp))

    def to_json(self) -> Dict[str, Any]:
        return {
            "comments": [comment.to_json() for comment in self.comments],
            "num_comments": self.num_comments,
            "reviews": self.review_pii,
            "predictions": self.predictions,
            "evaluations": self.evaluations,
            "utility": self.utility,
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "AnnotatedComments":
        comments_json = data["comments"]
        comments = [Comment.from_json(com) for com in comments_json]
        review_pii = data["reviews"]
        predictions = data["predictions"]
        evaluations = data["evaluations"]
        utility = data["utility"]
        return cls(comments, review_pii, predictions, evaluations, utility)

    @classmethod
    def from_comments(
        cls, comments: List[Comment], review_pii, predictions, evaluations
    ) -> "AnnotatedComments":
        return cls(comments, review_pii, predictions, evaluations)

    def get_text(self) -> str:
        return "\n".join([comment.text for comment in self.comments])

    def __str__(self) -> str:
        return "\n".join([str(c) for c in self.comments])


class Profile:
    def __init__(
        self,
        username: str,
        annotated_comments: Union[List[AnnotatedComments],List[Comment]],
        review_pii: Optional[Dict[str, Dict[str, Any]]] = None,
        predictions: Optional[Dict[str, Dict[str, Any]]] = None,
        evaluations: Optional[Dict[str, Dict[str, Dict[str, List[int]]]]] = None,
    ) -> None:
        self.username = username
        self.num_comments = len(annotated_comments)
        if isinstance(annotated_comments[0], Comment):
            self.comments = [
                AnnotatedComments(
                    annotated_comments, review_pii, predictions, evaluations
                )
            ]
        else:
            assert isinstance(annotated_comments[0], AnnotatedComments)
            self.comments = annotated_comments

        self.review_pii = self.comments[0].review_pii

        self.spans: Dict[str, List[str]] = {}
        self.span_pii = self.get_relevant_pii()

    def has_inference(self) -> bool:
        latest_comment = self.get_latest_comments()
        return len(latest_comment.predictions) > 0

    def get_next_steps(self, model_name: str) -> List[str]:
        latest_comment = self.get_latest_comments()
        potential_next_steps = []
        if model_name not in latest_comment.predictions:
            potential_next_steps.append("inference")
        else:
            potential_next_steps.append("anonymize")
        if model_name not in latest_comment.utility and len(self.comments) > 1:
            potential_next_steps.append("utility")
        return potential_next_steps

    def get_latest_comments(self) -> AnnotatedComments:
        return self.comments[-1]

    def get_original_comments(self) -> AnnotatedComments:
        return self.comments[0]

    def print_review_pii(self):
        for key, value in self.review_pii.items():
            print(f"{key}:")
            if key in ["time", "timestamp"]:
                continue
            for subkey, subvalue in value.items():
                if subkey in ["time", "timestamp"]:
                    continue
                if subvalue["hardness"] > 0:
                    print(
                        f"\t{subkey}: {subvalue['estimate']} - Hardness {subvalue['hardness']} Certainty {subvalue['certainty']}"
                    )

    def get_relevant_pii(self) -> List[str]:
        relevant_pii_type_set: set[str] = set({})

        for reviewer, res in self.review_pii.items():
            if reviewer in ["time", "timestamp"]:
                continue
            for pii_type, pii_res in res.items():
                if pii_type in ["time", "timestamp"]:
                    continue
                else:
                    if pii_res["hardness"] >= 1 and pii_res["certainty"] >= 1:
                        relevant_pii_type_set.add(pii_type)

        relevant_pii_types = list(relevant_pii_type_set)

        return relevant_pii_types
    def get_pii_gt(self) -> Dict[str, Dict[str, Any]]:
        piis=self.get_relevant_pii()
        ans={}
        for pii in piis:
            ans[pii]=""
        for reviewer, res in self.review_pii.items():
            if reviewer in ["time", "timestamp"]:
                continue
            for pii_type, pii_res in res.items():
                if pii_type in ["time", "timestamp"]:
                    continue
                else:
                    if pii_type in ans:
                        ans[pii_type]=pii_res["estimate"]
        return ans    
    def update_span_pii(self, to_remove: list[str]) -> None:
        for pii_type in to_remove:
            self.span_pii.remove(pii_type)

    def get_span_pii(self) -> List[str]:
        return self.span_pii

    def to_json(self) -> Dict[str, Any]:
        return {
            "username": self.username,
            "comments": [comment.to_json() for comment in self.comments],
            "reviews": self.review_pii,
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "Profile":
        print(data)
        username = data["username"]
        comments_json = data["comments"]
        review_pii = data["reviews"]
        predictions = data.get("predictions", {})
        evaluations = data.get("evaluations", {})
        if "comments" in comments_json[0]:
            comments = [AnnotatedComments.from_json(com) for com in comments_json]
        else:
            inner_comments = [Comment.from_json(com) for com in comments_json]
            comments = [
                AnnotatedComments(inner_comments, review_pii, predictions, evaluations)
            ]
        return cls(username, comments, review_pii, predictions, evaluations)

    def to_file(self, file: TextIO) -> None:
        file.write(json.dumps(self.to_json()) + "\n")
        file.flush()

    def __repr__(self) -> str:
        return f"{self.username} - {self.num_comments} comments"

    def __str__(self) -> str:
        res_str = f"{self.username} - {self.num_comments} comments\n"
        for comment in self.comments:
            res_str += f"{comment.subreddit}-{comment.text}\n"
        return res_str

    # Hashable
    def __hash__(self) -> int:
        # Concatenate all comments
        comments_str = "".join([comment.text for comment in self.comments])
        hash_str = self.username  # + comments_str

        return int(hashlib.sha1(hash_str.encode("utf-8")).hexdigest(), 16)
