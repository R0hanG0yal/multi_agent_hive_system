from __future__ import annotations

import difflib
import re

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "our",
    "the",
    "to",
    "was",
    "we",
    "what",
    "which",
    "with",
    "you",
    "your",
}

BEVERAGE_KEYWORDS = {
    "beverage",
    "chai",
    "coffee",
    "cocoa",
    "drink",
    "drinks",
    "espresso",
    "juice",
    "latte",
    "smoothie",
    "tea",
}

DRINK_CONTEXT_KEYWORDS = BEVERAGE_KEYWORDS | {
    "cold",
    "drink",
    "drinks",
    "hot",
    "rain",
    "raining",
    "rainy",
    "sip",
    "weather",
}

RECIPE_CONTEXT_KEYWORDS = {
    "cook",
    "cooking",
    "dish",
    "ingredients",
    "make",
    "prepare",
    "recipe",
}

PREFERENCE_PATTERNS = (
    re.compile(
        r"\bi\s+(?:really\s+)?(?:like|love|prefer)\s+([a-z][a-z\s]{0,40}?)(?:\s+(?:because|when|if|during|on|but|however|and i)\b|$)"
    ),
    re.compile(
        r"\bmy\s+favorite\s+(?:drink|beverage)\s+is\s+([a-z][a-z\s]{0,40}?)(?:\s+(?:because|when|if|during|on|but|however|and i)\b|$)"
    ),
)

IDENTITY_PATTERNS = (
    re.compile(
        r"\b(?:i am|i m|i'm|my name is|this is|call me)\s+([a-z][a-z\s]{0,40}?)(?:\s+(?:and|but|because|if|when|from|working|studying|student|founder|developer|engineer)\b|$)"
    ),
)

NON_NAME_TOKENS = {
    "am",
    "a",
    "an",
    "the",
    "allergic",
    "asthma",
    "business",
    "chai",
    "coffee",
    "cold",
    "developer",
    "diabetes",
    "diabetic",
    "doctor",
    "drink",
    "engineer",
    "fever",
    "flu",
    "founder",
    "happy",
    "hungry",
    "hypertension",
    "kidney",
    "liver",
    "patient",
    "pricing",
    "research",
    "sad",
    "sick",
    "startup",
    "student",
    "tea",
    "thirsty",
}

HEALTH_CONDITION_ALIASES = {
    "diabetes": (
        "diabetes",
        "diabetic",
        "prediabetes",
        "prediabetic",
        "high sugar",
        "blood sugar",
    ),
    "hypertension": (
        "hypertension",
        "high blood pressure",
        "high bp",
        "bp high",
    ),
    "heart_condition": (
        "heart disease",
        "heart condition",
        "cardiac",
        "arrhythmia",
        "palpitations",
        "heart failure",
    ),
    "kidney_disease": (
        "kidney disease",
        "kidney problem",
        "kidney failure",
        "ckd",
        "renal disease",
        "dialysis",
    ),
    "gerd_acidity": (
        "acidity",
        "acid reflux",
        "gerd",
        "reflux",
        "gastritis",
        "ulcer",
        "stomach ulcer",
        "heartburn",
        "indigestion",
    ),
    "cold_flu": (
        "common cold",
        "cough",
        "fever",
        "flu",
        "sore throat",
        "throat infection",
        "viral",
    ),
    "liver_condition": (
        "liver disease",
        "liver problem",
        "fatty liver",
        "hepatitis",
        "cirrhosis",
    ),
    "thyroid_condition": (
        "thyroid",
        "hypothyroid",
        "hyperthyroid",
    ),
    "pcos": (
        "pcos",
        "pcod",
        "polycystic ovary",
    ),
    "asthma_allergy": (
        "asthma",
        "allergy",
        "allergies",
        "sinus",
    ),
}

GENERIC_HEALTH_PATTERNS = (
    re.compile(
        r"\b(?:i have|i am|i m|i'm|even i have|as i am|i am a|i'm a|suffering from|living with)\s+([a-z][a-z\s]{2,50}?)(?:\s+(?:patient|problem|condition|disease))?(?:\b|$)"
    ),
)

KITCHEN_CONSTRAINT_ALIASES = {
    "no_pan": ("no pan", "without pan", "dont have pan", "do not have pan", "don't have pan"),
    "no_oven": ("no oven", "without oven", "dont have oven", "do not have oven", "don't have oven"),
    "no_stove": ("no stove", "without stove", "dont have stove", "do not have stove", "don't have stove", "no gas stove"),
    "no_microwave": ("no microwave", "without microwave", "dont have microwave", "do not have microwave", "don't have microwave"),
    "no_blender": ("no blender", "without blender", "dont have blender", "do not have blender", "don't have blender", "no mixer"),
    "no_pressure_cooker": (
        "no pressure cooker",
        "without pressure cooker",
        "dont have pressure cooker",
        "do not have pressure cooker",
        "don't have pressure cooker",
    ),
    "no_steamer": ("no steamer", "without steamer", "dont have steamer", "do not have steamer", "don't have steamer"),
}

INGREDIENT_AVOIDANCE_PATTERNS = (
    re.compile(
        r"\b(?:allergic to|allergy to|without|avoid|skip|exclude|dont use|don't use|do not use|dont suggest|don't suggest|do not suggest|dont add|don't add|do not add|dont like|don't like|do not like|hate|dislike)\s+(?:me\s+|my\s+|any\s+|some\s+|the\s+)?([a-z][a-z\s,]{0,40}?)(?:\s+(?:recipe|version|dish|please|for|in|while|when|because|so|but|next|instead|according|as)\b|$)"
    ),
    re.compile(
        r"\bno\s+([a-z][a-z\s]{0,20}?)(?:\s+(?:recipe|version|dish|please|for|in|while|when|because|so|but|next|instead)\b|$)"
    ),
    re.compile(
        r"\bi\s+(?:am\s+)?(?:allergic\s+to|allergic)\s+([a-z][a-z\s,]{0,30}?)(?:\s+(?:please|so|but|because|and|next|instead)\b|$)"
    ),
)

INGREDIENT_AVOIDANCE_BLOCKLIST = {
    "pan",
    "oven",
    "stove",
    "microwave",
    "blender",
    "mixer",
    "pressure cooker",
    "cooker",
    "steamer",
    "recipe",
    "version",
    "dish",
    "suggest",
    "it",
    "them",
    "that",
    "this",
}

DOMAIN_ALIASES = {
    "food": {
        "beverage",
        "beverages",
        "cook",
        "cooking",
        "drink",
        "drinks",
        "food",
        "kitchen",
        "meal",
        "meals",
        "recipe",
        "recipes",
    },
    "business": {
        "business",
        "customer",
        "customers",
        "market",
        "pricing",
        "roi",
        "startup",
        "startups",
        "venture",
    },
    "coding": {
        "code",
        "coder",
        "coding",
        "developer",
        "programming",
        "python",
        "script",
        "software",
    },
    "research": {
        "analysis",
        "benchmark",
        "experiment",
        "experiments",
        "research",
        "study",
        "studies",
    },
    "memory": {
        "history",
        "memory",
        "profile",
        "recall",
    },
}

MEMORY_SHARING_PATTERNS = (
    re.compile(
        r"\b(?:use|apply|share|allow)\s+(?:my\s+|the\s+)?([a-z][a-z\s]{0,30}?)(?:\s+(?:knowledge|memory|notes|context|preferences|preference|history))?\s+(?:for|in|to|with)\s+([a-z][a-z\s]{0,30}?)(?:\s+(?:knowledge|memory|notes|context|preferences|preference|history))?\b"
    ),
    re.compile(
        r"\b(?:let|allow)\s+([a-z][a-z\s]{0,30}?)\s+(?:agent|router)\s+(?:use|access)\s+(?:my\s+|the\s+)?([a-z][a-z\s]{0,30}?)(?:\s+(?:knowledge|memory|notes|context|preferences|preference|history))?\b"
    ),
)


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).strip()


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in normalize_text(text).split()
        if token and token not in STOPWORDS
    ]


def token_set(text: str) -> set[str]:
    return set(tokenize(text))


def overlap_score(left: set[str] | tuple[str, ...], right: set[str] | tuple[str, ...]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    shared = len(left_set & right_set)
    union = len(left_set | right_set)
    return shared / union if union else 0.0


def unique_preserve(items: list[str] | tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def has_rain_signal(tokens: set[str] | list[str] | tuple[str, ...]) -> bool:
    return any(token.startswith("rain") for token in tokens)


def has_drink_signal(tokens: set[str] | list[str] | tuple[str, ...]) -> bool:
    token_list = set(tokens)
    return bool(token_list & DRINK_CONTEXT_KEYWORDS)


def has_recipe_signal(tokens: set[str] | list[str] | tuple[str, ...]) -> bool:
    token_list = set(tokens)
    return bool(token_list & RECIPE_CONTEXT_KEYWORDS)


def extract_user_preference(text: str) -> dict[str, tuple[str, ...] | str] | None:
    normalized = normalize_text(text)
    tokens = tokenize(text)

    for pattern in PREFERENCE_PATTERNS:
        match = pattern.search(normalized)
        if not match:
            continue

        item_tokens = [
            token for token in match.group(1).split()
            if token and token not in STOPWORDS
        ][:3]
        if not item_tokens:
            return None

        keywords = list(item_tokens)
        if any(token in BEVERAGE_KEYWORDS for token in item_tokens):
            keywords.append("drink")
        if has_rain_signal(tokens):
            keywords.append("rainy")
        keywords.append("preference")

        return {
            "item": " ".join(item_tokens),
            "keywords": tuple(unique_preserve(keywords)),
        }

    return None


def extract_user_identity(text: str) -> dict[str, tuple[str, ...] | str] | None:
    normalized = normalize_text(text)

    for pattern in IDENTITY_PATTERNS:
        match = pattern.search(normalized)
        if not match:
            continue

        raw_tokens = [token for token in match.group(1).split() if token]
        if raw_tokens and raw_tokens[0] in {"a", "an", "the"}:
            raw_tokens = raw_tokens[1:]
        name_tokens = raw_tokens[:3]
        if not name_tokens:
            return None
        if any(token in NON_NAME_TOKENS or token.isdigit() for token in name_tokens):
            return None
        if all(len(token) <= 2 for token in name_tokens):
            return None

        display_name = " ".join(token.capitalize() for token in name_tokens)
        keywords = tuple(unique_preserve(["name", *name_tokens, "profile"]))
        return {
            "name": display_name,
            "keywords": keywords,
        }

    return None


def extract_health_conditions(text: str) -> tuple[str, ...]:
    normalized = normalize_text(text)
    tokens = normalized.split()
    conditions: list[str] = []

    for canonical, aliases in HEALTH_CONDITION_ALIASES.items():
        if any(alias in normalized for alias in aliases):
            conditions.append(canonical)

    if not conditions:
        for canonical, aliases in HEALTH_CONDITION_ALIASES.items():
            for alias in aliases:
                alias_tokens = alias.split()
                if len(alias_tokens) == 1:
                    alias_threshold = 0.8 if len(alias) >= 6 else 0.9
                    if any(
                        difflib.SequenceMatcher(None, token, alias).ratio() >= alias_threshold
                        for token in tokens
                    ):
                        conditions.append(canonical)
                        break
                else:
                    window_size = len(alias_tokens)
                    for index in range(len(tokens) - window_size + 1):
                        window = " ".join(tokens[index:index + window_size])
                        if difflib.SequenceMatcher(None, window, alias).ratio() >= 0.88:
                            conditions.append(canonical)
                            break
                    if canonical in conditions:
                        break

    if conditions:
        return tuple(unique_preserve(conditions))

    for pattern in GENERIC_HEALTH_PATTERNS:
        match = pattern.search(normalized)
        if not match:
            continue
        snippet = " ".join(match.group(1).split()[:4]).strip()
        if snippet and snippet not in {"happy", "sad", "hungry", "thirsty"}:
            return ("generic_condition",)

    return ()


def extract_health_topic_candidate(text: str) -> str | None:
    normalized = normalize_text(text)
    tokens = normalized.split()

    for aliases in HEALTH_CONDITION_ALIASES.values():
        for alias in aliases:
            if alias in normalized:
                return alias

    for aliases in HEALTH_CONDITION_ALIASES.values():
        for alias in aliases:
            alias_tokens = alias.split()
            if len(alias_tokens) == 1:
                alias_threshold = 0.8 if len(alias) >= 6 else 0.9
                if any(
                    difflib.SequenceMatcher(None, token, alias).ratio() >= alias_threshold
                    for token in tokens
                ):
                    return alias
            else:
                window_size = len(alias_tokens)
                for index in range(len(tokens) - window_size + 1):
                    window = " ".join(tokens[index:index + window_size])
                    if difflib.SequenceMatcher(None, window, alias).ratio() >= 0.88:
                        return alias

    for pattern in GENERIC_HEALTH_PATTERNS:
        match = pattern.search(normalized)
        if not match:
            continue
        snippet = " ".join(match.group(1).split()[:4]).strip()
        if snippet and snippet not in {"happy", "sad", "hungry", "thirsty"}:
            return snippet

    return None


def extract_kitchen_constraints(text: str) -> tuple[str, ...]:
    normalized = normalize_text(text)
    constraints: list[str] = []
    for canonical, aliases in KITCHEN_CONSTRAINT_ALIASES.items():
        matched = False
        for alias in aliases:
            alias_normalized = normalize_text(alias)
            if alias_normalized and alias_normalized in normalized:
                matched = True
                break

            alias_tokens = alias_normalized.split()
            if not alias_tokens:
                continue

            pattern = None
            if alias_tokens[0] in {"no", "without"} and len(alias_tokens) > 1:
                prefix = r"\s+".join(re.escape(token) for token in alias_tokens[:1])
                noun = r"\s+".join(re.escape(token) for token in alias_tokens[1:])
                pattern = rf"\b{prefix}\s+(?:a\s+|an\s+)?{noun}\b"
            elif "have" in alias_tokens and alias_tokens.index("have") < len(alias_tokens) - 1:
                split_index = alias_tokens.index("have") + 1
                prefix = r"\s+".join(re.escape(token) for token in alias_tokens[:split_index])
                noun = r"\s+".join(re.escape(token) for token in alias_tokens[split_index:])
                pattern = rf"\b{prefix}\s+(?:a\s+|an\s+)?{noun}\b"
            else:
                joined_tokens = r"\s+".join(re.escape(token) for token in alias_tokens)
                pattern = rf"\b{joined_tokens}\b"

            if pattern and re.search(pattern, normalized):
                matched = True
                break

        if matched:
            constraints.append(canonical)
    return tuple(unique_preserve(constraints))


def extract_ingredient_avoidances(text: str) -> tuple[str, ...]:
    normalized = normalize_text(text)
    found: list[str] = []

    for pattern in INGREDIENT_AVOIDANCE_PATTERNS:
        for match in pattern.finditer(normalized):
            snippet = match.group(1).strip()
            if not snippet:
                continue
            candidates = re.split(r"\s*(?:,|and|or|/)\s*", snippet)
            for candidate in candidates:
                cleaned_tokens = [
                    token
                    for token in candidate.split()
                    if token not in STOPWORDS
                    and token
                    not in {
                        "recipe",
                        "version",
                        "dish",
                        "suggest",
                        "suggestion",
                        "make",
                        "with",
                        "without",
                    }
                ][:3]
                if not cleaned_tokens:
                    continue
                cleaned = " ".join(cleaned_tokens)
                if cleaned in INGREDIENT_AVOIDANCE_BLOCKLIST:
                    continue
                found.append(cleaned)

    return tuple(unique_preserve(found))


def classify_task_difficulty(text: str) -> str:
    normalized = normalize_text(text)
    tokens = normalized.split()
    hard_keywords = {"analyze", "compare", "design", "evaluate", "optimize", "plan"}
    medium_keywords = {"build", "debug", "estimate", "explain", "prepare", "recipe", "suggest"}

    if len(tokens) > 12 or any(keyword in tokens for keyword in hard_keywords):
        return "hard"
    if len(tokens) >= 6 or any(keyword in tokens for keyword in medium_keywords):
        return "medium"
    return "easy"


def _resolve_domain_alias(text: str) -> str | None:
    normalized = normalize_text(text)
    tokens = set(normalized.split())
    ranked_aliases = sorted(
        (
            (domain, alias)
            for domain, aliases in DOMAIN_ALIASES.items()
            for alias in aliases
        ),
        key=lambda item: len(item[1]),
        reverse=True,
    )
    for domain, alias in ranked_aliases:
        if alias in normalized or alias in tokens:
            return domain
    return None


def extract_memory_sharing_preferences(text: str) -> tuple[tuple[str, str], ...]:
    normalized = normalize_text(text)
    shares: list[tuple[str, str]] = []
    for pattern_index, pattern in enumerate(MEMORY_SHARING_PATTERNS):
        for match in pattern.finditer(normalized):
            first = _resolve_domain_alias(match.group(1))
            second = _resolve_domain_alias(match.group(2))
            if not first or not second or first == second:
                continue
            if pattern_index == 0:
                shares.append((first, second))
            else:
                shares.append((second, first))
    return tuple(unique_preserve([f"{left}:{right}" for left, right in shares]))


def parse_memory_sharing_preferences(text: str) -> tuple[tuple[str, str], ...]:
    shares = []
    for item in extract_memory_sharing_preferences(text):
        left, right = item.split(":", 1)
        shares.append((left, right))
    return tuple(shares)
