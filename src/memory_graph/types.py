from enum import StrEnum


class NodeType(StrEnum):
    PERSON = "Person"
    RELATIONSHIP = "Relationship"
    EVENT = "Event"
    MEMORY = "Memory"
    PREFERENCE = "Preference"
    OCCASION = "Occasion"
    GIFT_ARTIFACT = "GiftArtifact"


class EdgeType(StrEnum):
    RELATES_TO = "RELATES_TO"
    PARTICIPATED_IN = "PARTICIPATED_IN"
    RECALLS = "RECALLS"
    PREFERS = "PREFERS"
    INTENDED_FOR = "INTENDED_FOR"
    GENERATED_FROM = "GENERATED_FROM"
