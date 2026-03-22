import random

# Recommended color palettes based on skin tone
COLOR_RULES = {

    "fair": [
        "navy", "emerald", "burgundy",
        "deep purple", "charcoal"
    ],

    "medium": [
        "olive", "mustard", "teal",
        "cream", "maroon"
    ],

    "dark": [
        "white", "pastel blue", "lavender",
        "light grey", "peach"
    ]
}


def recommend_colors(skin_tone, k=3):
    """
    Returns color recommendations based on detected skin tone
    """

    if skin_tone not in COLOR_RULES:
        return []

    palette = COLOR_RULES[skin_tone]

    return random.sample(palette, min(k, len(palette)))