from typing import Iterator
from .anonymizer import Anonymizer
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient, DocumentError
from src.utils.string_utils import anonymize_str
from src.configs import AnonymizerConfig
from src.reddit.reddit_types import Comment, AnnotatedComments, Profile
import credentials


class AzureAnonymizer(Anonymizer):
    def __init__(self, cfg: AnonymizerConfig):
        self.threshhold = cfg.azure_threshhold
        self.remove_entities = [
            "Person",
            "PersonType",
            "Location",
            "Organization",
            "Event",
            "Address",
            "PhoneNumber",
            "Email",
            "URL",
            "IP",
            "DateTime",
            ("Quantity", ["Age", "Currency", "Number"]),
        ]
        self.use_entities = False
        if cfg.azure_replacement == "entity":
            self.use_entities = True

        self.text_analytics_client = TextAnalyticsClient(
            endpoint=credentials.azure_language_endpoint,
            credential=AzureKeyCredential(credentials.azure_language_key),
        )

    def anonymize(self, text: str) -> str:
        anonymized_results = self.text_analytics_client.recognize_entities([text])

        for res in anonymized_results:
            if isinstance(res, DocumentError):
                if res.id == "2":
                    new_entities = []
                    for k in range(0, len(text), 5000):
                        end = min(k + 5000, len(text))
                        res = self.text_analytics_client.recognize_entities(
                            [text[k:end]]
                        )[0]
                        new_entities.extend(res.entities)
                    res.entities = new_entities

            anonymization_requests = []
            for entity in res.entities:
                need_removal = False

                for rme in self.remove_entities:
                    if isinstance(rme, tuple):
                        if entity.category == rme[0] and entity.subcategory in rme[1]:
                            need_removal = True
                            break
                    else:
                        if entity.category == rme:
                            need_removal = True
                            break

                if not need_removal:
                    continue

                if entity.confidence_score < self.threshhold:
                    continue

                if self.use_entities:
                    repl_text = (
                        f"<{entity.category.upper()}>"
                        if not entity.subcategory
                        else f"<{entity.subcategory.upper()}>"
                    )
                else:
                    repl_text = "*" * len(entity.text)

                anonymization_requests.append(
                    (
                        entity.text,
                        entity.offset,
                        entity.offset + entity.length,
                        repl_text,
                    )
                )

            text = anonymize_str(
                text,
                anonymization_requests,
            )

        return text

    def anonymize_profiles(self, profiles) -> Iterator[Profile]:
        for profile in profiles:

            new_comments = []

            for comment in profile.comments[-1].comments:
                new_comments.append(
                    Comment(
                        self.anonymize(comment.text),
                        comment.subreddit,
                        comment.user,
                        comment.timestamp,
                    )
                )

            profile.comments.append(AnnotatedComments(
                new_comments, profile.review_pii, {}, {}
            ))

            yield profile