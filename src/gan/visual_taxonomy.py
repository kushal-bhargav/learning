from __future__ import annotations

VISUAL_GIFT_ARTIFACT_TYPES = (
    "greeting_card",
    "gift_wrap",
    "keepsake_print",
    "poster",
    "gift_tag",
    "sticker",
    "invitation",
    "decorative_motif",
    "other",
)


def normalize_artifact_type(value: object) -> str:
    text = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "card": "greeting_card",
        "greeting": "greeting_card",
        "greetingcard": "greeting_card",
        "wrap": "gift_wrap",
        "wrapping": "gift_wrap",
        "wrapping_paper": "gift_wrap",
        "print": "keepsake_print",
        "keepsake": "keepsake_print",
        "tag": "gift_tag",
        "label": "gift_tag",
        "motif": "decorative_motif",
    }
    normalized = aliases.get(text, text)
    return normalized if normalized in VISUAL_GIFT_ARTIFACT_TYPES else "other"


def artifact_description(artifact_type: str) -> str:
    return {
        "greeting_card": "a synthetic greeting card image for a personalized gift",
        "gift_wrap": "a repeatable gift wrap pattern for a personalized gift",
        "keepsake_print": "a keepsake print illustration for a meaningful gift",
        "poster": "a personalized poster-style gift illustration",
        "gift_tag": "a small printable gift tag design",
        "sticker": "a cheerful sticker-style gift design",
        "invitation": "a celebratory invitation or announcement card design",
        "decorative_motif": "a decorative motif for a gift artifact",
        "other": "a visual gift artifact",
    }[normalize_artifact_type(artifact_type)]
