from __future__ import annotations

from dataclasses import dataclass
import re

from .live_knowledge import InternetKnowledgeRetriever, KnowledgeRetriever, LiveKnowledge
from .models import AgentProposal, MemoryRecord
from .utils import (
    BEVERAGE_KEYWORDS,
    clamp,
    extract_health_conditions,
    extract_health_topic_candidate,
    extract_ingredient_avoidances,
    extract_kitchen_constraints,
    extract_user_identity,
    extract_user_preference,
    has_drink_signal,
    has_rain_signal,
    has_recipe_signal,
    normalize_text,
    overlap_score,
    token_set,
    tokenize,
    unique_preserve,
)
from .vector import VectorEncoder

DOMAIN_KEYWORDS: dict[str, set[str]] = {
    "food": {
        "burger",
        "beverage",
        "breakfast",
        "chai",
        "coffee",
        "cook",
        "cocoa",
        "dal",
        "disease",
        "diabetes",
        "dinner",
        "dish",
        "drink",
        "drinks",
        "food",
        "gerd",
        "health",
        "hot",
        "hypertension",
        "idli",
        "ingredient",
        "kidney",
        "liver",
        "lunch",
        "meal",
        "noodles",
        "oats",
        "pakoda",
        "pantry",
        "pasta",
        "poha",
        "protein",
        "rain",
        "rainy",
        "raining",
        "recipe",
        "rice",
        "roll",
        "sambar",
        "sambhar",
        "sandwich",
        "snack",
        "sugar",
        "tea",
        "upma",
        "vegetarian",
        "wrap",
    },
    "business": {
        "business",
        "channel",
        "customer",
        "delivery",
        "market",
        "margin",
        "monetization",
        "partner",
        "partners",
        "pilot",
        "pitch",
        "pricing",
        "revenue",
        "roi",
        "startup",
        "strategy",
        "subscription",
    },
    "coding": {
        "bug",
        "code",
        "coding",
        "debug",
        "develop",
        "error",
        "function",
        "implement",
        "programming",
        "python",
        "script",
        "software",
    },
    "research": {
        "ab",
        "baseline",
        "benchmark",
        "bias",
        "compare",
        "dataset",
        "design",
        "evaluate",
        "evaluation",
        "experiment",
        "fine",
        "hypothesis",
        "learning",
        "metric",
        "model",
        "paper",
        "research",
        "reinforcement",
        "risk",
        "study",
        "synthetic",
        "test",
        "training",
    },
    "memory": {
        "earlier",
        "called",
        "name",
        "lean",
        "previous",
        "recall",
        "remember",
        "remind",
        "what",
        "which",
    },
}

DOMAIN_PROTOTYPE_TEXT = {
    "food": "food meal recipe drink breakfast lunch dinner snack vegetarian protein kitchen ingredients cooking health beverage tea coffee rice noodles sandwich",
    "business": "business startup pricing market margin pilot roi partner customer revenue channel strategy subscription venture sales",
    "coding": "coding software python script function debug bug error algorithm implement programming developer code",
    "research": "research experiment hypothesis metric baseline dataset evaluation benchmark compare model study paper reinforcement learning analysis",
    "memory": "memory recall remember earlier previous remind profile history name what was which",
}

_ROUTER_ENCODER = VectorEncoder()
DOMAIN_PROTOTYPE_VECTORS = {
    domain: _ROUTER_ENCODER.encode(f"{DOMAIN_PROTOTYPE_TEXT[domain]} {' '.join(sorted(keywords))}")
    for domain, keywords in DOMAIN_KEYWORDS.items()
}

MEMORY_CUES = ("remember", "earlier", "previous", "recall", "remind me", "what was", "which")
IDEA_CUES = ("idea", "ideas", "startup", "business", "venture")
HOT_BEVERAGES = {"chai", "coffee", "cocoa", "espresso", "latte", "tea"}
RECIPE_CUES = ("recipe", "how to make", "how do i make", "how can i make", "how to cook", "prepare")
_RECIPE_TYPOS = {"reciepe", "recipie", "reipe", "recpe", "receipe", "recipee", "recpie", "reciep"}
RECIPE_FILLER_TERMS = {
    "dish",
    "exact",
    "food",
    "give",
    "make",
    "meal",
    "plan",
    "please",
    "prepare",
    "quick",
    "recipe",
    "show",
    "simple",
    "tell",
    "want",
} | _RECIPE_TYPOS
RECIPE_SUBJECT_IGNORE_TERMS = {
    "a",
    "an",
    "the",
    "is",
    "me",
    "my",
    "of",
    "some",
}
RECIPE_SUBJECT_STOP_TERMS = {
    "after",
    "at",
    "before",
    "because",
    "but",
    "for",
    "from",
    "if",
    "in",
    "into",
    "on",
    "since",
    "so",
    "today",
    "tomorrow",
    "until",
    "using",
    "when",
    "while",
    "with",
    "without",
}
LIVE_LOOKUP_CUES = (
    "what is",
    "what are",
    "tell me about",
    "explain",
    "define",
    "describe",
    "overview of",
    "information on",
    "info on",
)
RECIPE_HEADWORDS = {
    "burger",
    "chai",
    "coffee",
    "curry",
    "dal",
    "fritter",
    "golgappa",
    "idli",
    "noodles",
    "oats",
    "pakoda",
    "pasta",
    "puri",
    "poha",
    "rice",
    "roll",
    "salad",
    "sambar",
    "sambhar",
    "sandwich",
    "tea",
    "upma",
    "wrap",
}
ASSEMBLED_RECIPE_HEADS = {"burger", "roll", "sandwich", "wrap"}
BATTER_FRY_RECIPE_HEADS = {"bhaji", "fritter", "pakoda"}
SIMMER_RECIPE_HEADS = {"dal", "noodles", "oats", "pasta", "poha", "rice", "upma"}
BREW_RECIPE_HEADS = {"chai", "coffee", "cocoa", "tea"}
HEALTH_PRIORITIES = (
    "kidney_disease",
    "gerd_acidity",
    "hypertension",
    "heart_condition",
    "diabetes",
    "cold_flu",
    "liver_condition",
    "thyroid_condition",
    "pcos",
    "asthma_allergy",
    "generic_condition",
)

HEALTH_GUIDANCE = {
    "diabetes": {
        "drink": "an unsweetened coffee, plain chai without sugar, or ginger tea",
        "avoid": "sugary syrups, sweetened premixes, regular soda, and dessert-style shakes",
        "reason": "keeping added sugar low matters for blood sugar control",
    },
    "hypertension": {
        "drink": "a light tea, decaf coffee, or warm ginger water",
        "avoid": "energy drinks and very strong caffeinated drinks",
        "reason": "too much caffeine can temporarily raise blood pressure",
    },
    "heart_condition": {
        "drink": "a mild tea, decaf coffee, or plain warm water",
        "avoid": "energy drinks and heavy caffeine loading",
        "reason": "stimulant-heavy drinks can be hard on the heart",
    },
    "gerd_acidity": {
        "drink": "warm ginger tea, chamomile, or plain warm water",
        "avoid": "strong coffee, fizzy drinks, very acidic drinks, and mint-heavy drinks",
        "reason": "those can worsen reflux or acidity symptoms",
    },
    "kidney_disease": {
        "drink": "only a clinician-approved amount of plain water or a mild unsweetened drink",
        "avoid": "large fluid loads and packaged sports or energy drinks",
        "reason": "kidney conditions can come with specific fluid and mineral limits",
    },
    "cold_flu": {
        "drink": "warm ginger, tulsi, or plain herbal tea",
        "avoid": "very icy drinks if they irritate your throat",
        "reason": "warm fluids are usually easier when you feel unwell",
    },
    "liver_condition": {
        "drink": "plain water, unsweetened tea, or black coffee if your doctor allows it",
        "avoid": "alcohol, energy drinks, and very sugary drinks",
        "reason": "gentler drinks are usually the safer direction",
    },
    "thyroid_condition": {
        "drink": "a simple unsweetened tea or coffee in moderation",
        "avoid": "very sugary and highly caffeinated drinks if they make symptoms worse",
        "reason": "simple drinks are easier to fit around medication and routine",
    },
    "pcos": {
        "drink": "an unsweetened tea, black coffee, or cinnamon-ginger tea",
        "avoid": "sugary cafe drinks and sweetened bottled beverages",
        "reason": "lower-sugar choices are usually the safer baseline",
    },
    "asthma_allergy": {
        "drink": "a warm herbal drink or plain tea",
        "avoid": "anything that personally triggers irritation or allergy symptoms",
        "reason": "warm, gentle drinks are often better tolerated",
    },
    "generic_condition": {
        "drink": "a mild unsweetened warm drink",
        "avoid": "very sugary, very caffeinated, or energy drinks until you know your restrictions",
        "reason": "different conditions can have very different drink limits",
    },
}


def _budget_label(normalized_query: str) -> str:
    if "1 lakh" in normalized_query or "one lakh" in normalized_query:
        if any(
            phrase in normalized_query
            for phrase in ("less than", "under", "below", "within", "budget")
        ):
            return "under INR 1 lakh"
        return "around INR 1 lakh"
    return "a lean starting budget"


def _is_idea_query(normalized_query: str, tokens: set[str]) -> bool:
    return (
        any(cue in normalized_query for cue in IDEA_CUES)
        and any(word in tokens for word in {"idea", "ideas", "startup", "business", "venture"})
    )


def _future_signal(tokens: set[str], normalized_query: str) -> bool:
    return any(
        token in tokens
        for token in {
            "ai",
            "automation",
            "climate",
            "deeptech",
            "ev",
            "future",
            "futuristic",
            "robotics",
            "saas",
            "solar",
            "tech",
        }
    ) or "future ready" in normalized_query


def _budget_signal(tokens: set[str], normalized_query: str) -> bool:
    return "budget" in tokens or "lakh" in tokens or "under" in normalized_query or "less than" in normalized_query


def encode_query_vector(query: str) -> dict[str, float]:
    tokens = token_set(query)
    query_vector = _ROUTER_ENCODER.encode(query)
    raw_scores: dict[str, float] = {}
    token_count = max(len(tokens), 1)
    for domain, keywords in DOMAIN_KEYWORDS.items():
        lexical_overlap = len(tokens & keywords) / token_count
        vector_similarity = VectorEncoder.cosine_similarity(query_vector, DOMAIN_PROTOTYPE_VECTORS[domain])
        raw_scores[domain] = max(0.0, (0.72 * vector_similarity) + (0.28 * lexical_overlap))
    total = sum(raw_scores.values()) or 1.0
    return {domain: round(score / total, 3) for domain, score in raw_scores.items()}


def _focus_terms(tokens: set[str], hits: tuple[str, ...], defaults: tuple[str, ...]) -> list[str]:
    return unique_preserve(list(hits) + [token for token in tokens if len(token) > 3] + list(defaults))[:4]


def _memory_snippet(records: list[MemoryRecord], domain: str | None = None) -> str | None:
    filtered = [
        record for record in records
        if (domain is None or record.domain == domain)
        and not record.key.startswith("feedback:")
        and not record.key.startswith("profile:")
    ]
    if not filtered:
        return None
    return " | ".join(record.summary for record in filtered[:2])


def _is_drink_query(normalized_query: str, tokens: set[str]) -> bool:
    return has_drink_signal(tokens) or "what should i drink" in normalized_query


def _is_recipe_query(normalized_query: str, tokens: set[str]) -> bool:
    return (
        "recipe" in tokens
        or bool(tokens & _RECIPE_TYPOS)
        or "cook" in tokens
        or "make" in tokens
        or any(cue in normalized_query for cue in RECIPE_CUES)
    )


def _is_meal_suggestion_query(normalized_query: str, tokens: set[str]) -> bool:
    if not any(cue in normalized_query for cue in _FOOD_LIVE_CUES):
        return False
    food_intent_tokens = {
        "breakfast",
        "dinner",
        "dish",
        "drink",
        "drinks",
        "eat",
        "food",
        "have",
        "lunch",
        "meal",
        "protein",
        "recipe",
        "snack",
        "vegetarian",
    }
    return bool(tokens & food_intent_tokens) or any(
        phrase in normalized_query
        for phrase in {
            "what should i eat",
            "what should i have",
            "food for",
            "meal for",
            "dish for",
            "snack for",
            "breakfast for",
            "lunch for",
            "dinner for",
        }
    )


def _clean_recipe_subject(candidate: str) -> str | None:
    normalized_candidate = normalize_text(candidate).replace("idlisambhar", "idli sambhar").replace("idlisambar", "idli sambar")
    filtered: list[str] = []
    for token in normalized_candidate.split():
        if token in RECIPE_FILLER_TERMS or token in RECIPE_SUBJECT_IGNORE_TERMS:
            continue
        if filtered and token in RECIPE_SUBJECT_STOP_TERMS:
            break
        filtered.append(token)
    if not filtered:
        return None
    if "idli" in filtered and ("sambar" in filtered or "sambhar" in filtered):
        return "idli sambhar"
    return " ".join(filtered[:4])


def _recipe_subject(query: str, focus_terms: list[str]) -> str | None:
    normalized = normalize_text(query)
    # Normalize common recipe typos in the query text for pattern matching
    for typo in _RECIPE_TYPOS:
        normalized = normalized.replace(typo, "recipe")
    subject_patterns = (
        r"(?:recipe\s+(?:of|for)\s+)([a-z][a-z\s]{0,40})",
        r"(?:how to make|how do i make|how can i make|how to cook|prepare)\s+([a-z][a-z\s]{0,40})",
        r"(?:give(?: me)?(?: an?| some)?\s+)([a-z][a-z\s]{0,40})\s+recipe",
        r"([a-z][a-z\s]{0,40})\s+recipe",
    )
    for pattern in subject_patterns:
        match = re.search(pattern, normalized)
        if not match:
            continue
        subject = _clean_recipe_subject(match.group(1))
        if subject:
            return subject

    filtered_terms = [
        term
        for term in tokenize(query)
        if term not in RECIPE_FILLER_TERMS
    ]
    if len(filtered_terms) >= 2 and filtered_terms[-1] in RECIPE_HEADWORDS:
        return " ".join(filtered_terms[-2:])
    if filtered_terms:
        return filtered_terms[-1]
    for term in focus_terms:
        if term not in RECIPE_FILLER_TERMS:
            return term
    return None


def _title_case_subject(subject: str | None) -> str:
    if not subject:
        return "home-style dish"
    return subject


def _recipe_head(subject: str | None) -> str | None:
    if not subject:
        return None
    return subject.split()[-1]


def _is_live_lookup_query(normalized_query: str) -> bool:
    return any(cue in normalized_query for cue in LIVE_LOOKUP_CUES)


# Food-specific cues that should trigger a live lookup even without
# "what is" / "explain" style phrasing.
_FOOD_LIVE_CUES = (
    "suggest", "recommend", "give me", "tell me",
    "what to eat", "what to have", "what should i eat",
    "what should i have", "dish for", "meal for",
    "food for", "dinner for", "lunch for", "breakfast for",
    "snack for", "dish at", "meal at",
)


def _extract_food_subject(query: str) -> str | None:
    """Extract the food subject from suggestion-style queries.

    Examples:
        'suggest me dish for night' -> 'night dinner'
        'recommend a meal for morning' -> 'morning breakfast'
        'what should I eat at night' -> 'night dinner'
    """
    normalized = normalize_text(query)
    # Try to find a specific dish name first
    food_patterns = (
        r"(?:suggest|recommend|give me|tell me)[^a-z]*(?:a |an |some )?([a-z][a-z\s]{2,30})(?:\s+(?:for|at|in)\s+.*)?",
        r"(?:for|at|in)\s+(?:the )?([a-z]+)\s*$",
    )
    time_context = ""
    if any(w in normalized for w in ("night", "evening", "dinner")):
        time_context = "dinner"
    elif any(w in normalized for w in ("morning", "breakfast")):
        time_context = "breakfast"
    elif any(w in normalized for w in ("lunch", "afternoon")):
        time_context = "lunch"
    elif any(w in normalized for w in ("snack", "evening snack")):
        time_context = "snack"

    # Extract the dish keyword
    tokens = [t for t in tokenize(query) if t not in RECIPE_FILLER_TERMS and len(t) > 2]
    dish_words = [t for t in tokens if t not in {"night", "morning", "evening", "afternoon", "lunch", "dinner", "breakfast", "snack", "suggest", "recommend"}]

    if dish_words:
        subject = " ".join(dish_words[:2])
        if time_context:
            return f"{subject} {time_context} recipe"
        return f"{subject} recipe"
    if time_context:
        return f"{time_context} recipes"
    return None


def _live_lookup_subject(query: str) -> str | None:
    normalized = normalize_text(query)
    subject_patterns = (
        r"(?:what is|what are|tell me about|explain|define|describe|overview of|information on|info on)\s+([a-z0-9][a-z0-9\s]{0,80})",
    )
    for pattern in subject_patterns:
        match = re.search(pattern, normalized)
        if not match:
            continue
        subject = " ".join(
            token
            for token in match.group(1).split()
            if token not in {"a", "an", "the"}
        ).strip()
        if subject:
            return subject
    return None


def _preferred_beverage_from_memory(records: list[MemoryRecord]) -> str | None:
    ordered_preferences = ("coffee", "chai", "tea", "cocoa", "espresso", "latte", "juice", "smoothie")
    for record in records:
        record_tokens = set(record.keywords) | set(tokenize(record.query)) | set(tokenize(record.summary))
        for beverage in ordered_preferences:
            if beverage in record_tokens:
                return beverage
    return None


def _health_conditions_from_memory(records: list[MemoryRecord]) -> tuple[str, ...]:
    found: list[str] = []
    for record in records:
        if record.key.startswith("health:"):
            for condition in record.keywords:
                if condition in HEALTH_GUIDANCE:
                    found.append(condition)
    return tuple(unique_preserve(found))


def _ordered_conditions(conditions: tuple[str, ...]) -> list[str]:
    ordered: list[str] = []
    for item in HEALTH_PRIORITIES:
        if item in conditions:
            ordered.append(item)
    for item in conditions:
        if item not in ordered:
            ordered.append(item)
    return ordered


def _remembered_name_from_memory(records: list[MemoryRecord]) -> str | None:
    for record in records:
        if not record.key.startswith("profile:"):
            continue
        prefix = "profile note: your name is "
        if record.summary.startswith(prefix):
            return record.summary[len(prefix):].strip()
    return None


def _feedback_notes_from_memory(records: list[MemoryRecord], domain: str | None = None) -> tuple[str, ...]:
    notes: list[str] = []
    prefix = "feedback note: "
    for record in records:
        if not record.key.startswith("feedback:"):
            continue
        if domain is not None and record.domain != domain:
            continue
        if record.summary.startswith(prefix):
            notes.append(record.summary[len(prefix):].strip())
        else:
            notes.append(record.summary.strip())
    return tuple(unique_preserve([note for note in notes if note]))


def _exact_feedback_notes_from_memory(
    records: list[MemoryRecord],
    query: str,
    domain: str | None = None,
) -> tuple[str, ...]:
    query_normalized = normalize_text(query)
    notes: list[str] = []
    prefix = "feedback note: "
    for record in records:
        if not record.key.startswith("feedback:"):
            continue
        if domain is not None and record.domain != domain:
            continue
        if normalize_text(record.query) != query_normalized:
            continue
        if record.summary.startswith(prefix):
            notes.append(record.summary[len(prefix):].strip())
        else:
            notes.append(record.summary.strip())
    return tuple(unique_preserve([note for note in notes if note]))


def _preferred_dish_from_memory(records: list[MemoryRecord], query_normalized: str) -> str | None:
    query_time = None
    if any(w in query_normalized for w in ("morning", "breakfast")):
        query_time = "morning"
    elif any(w in query_normalized for w in ("afternoon", "lunch")):
        query_time = "afternoon"
    elif any(w in query_normalized for w in ("night", "evening", "dinner")):
        query_time = "night"

    for record in records:
        if record.key.startswith("feedback:") or record.key.startswith("food:"):
            # Look for "i like to have X", "prefer X"
            match = re.search(r"(?:like to have|prefer) ([a-z\s]+?)(?: at | for |$)", record.summary, re.IGNORECASE)
            if match:
                dish = match.group(1).strip()
                summary_lower = record.summary.lower()
                memory_time = None
                if any(w in summary_lower for w in ("morning", "breakfast")):
                    memory_time = "morning"
                elif any(w in summary_lower for w in ("afternoon", "lunch")):
                    memory_time = "afternoon"
                elif any(w in summary_lower for w in ("night", "evening", "dinner")):
                    memory_time = "night"
                
                # If memory specifies a time and query asks for a DIFFERENT time, skip this dish!
                if memory_time and query_time and memory_time != query_time:
                    continue
                return dish
    return None

def _avoided_ingredients_from_memory(records: list[MemoryRecord]) -> tuple[str, ...]:
    found: list[str] = []
    prefix = "ingredient note: avoid "
    feedback_prefix = "feedback note: "
    for record in records:
        if record.key.startswith("ingredient:"):
            if record.summary.startswith(prefix):
                candidates = [item.strip() for item in record.summary[len(prefix):].split(",")]
                for candidate in candidates:
                    if candidate:
                        found.append(candidate)
        elif record.key.startswith("feedback:"):
            # Also extract ingredient avoidances from feedback text
            text = record.summary
            if text.startswith(feedback_prefix):
                text = text[len(feedback_prefix):]
            extracted = extract_ingredient_avoidances(text)
            found.extend(extracted)
            # Also check the original query stored with the feedback
            if record.query:
                extracted_query = extract_ingredient_avoidances(record.query)
                found.extend(extracted_query)
    return tuple(unique_preserve(found))


def _kitchen_constraints_from_memory(records: list[MemoryRecord]) -> tuple[str, ...]:
    found: list[str] = []
    for record in records:
        if record.key.startswith("kitchen:"):
            for keyword in record.keywords:
                if keyword.startswith("no_"):
                    found.append(keyword)
    return tuple(unique_preserve(found))


def _recipe_constraint_note(constraints: tuple[str, ...], subject_label: str, head: str | None) -> str:
    if not constraints:
        return ""

    notes: list[str] = []
    if "no_pan" in constraints:
        if subject_label == "idli sambhar" or head in {"idli", "sambar", "sambhar"}:
            notes.append("Since you do not have a pan, steam the idli in an idli stand or cooker and make the sambhar in a deep pot or pressure cooker instead of a shallow pan.")
        elif head in {"sandwich", "burger", "roll", "wrap"}:
            notes.append("Since you do not have a pan, skip toasting and serve it as a fresh assembled version.")
        else:
            notes.append("Since you do not have a pan, use a pot, cooker, or steamer if that works for the dish.")
    if "no_oven" in constraints:
        notes.append("Do not use oven steps; keep it stovetop, steamed, or no-cook.")
    if "no_stove" in constraints:
        notes.append("Since you do not have a stove, choose a no-cook or electric-appliance version if possible.")
    if "no_microwave" in constraints:
        notes.append("Skip microwave shortcuts and use manual prep instead.")
    if "no_pressure_cooker" in constraints:
        notes.append("Do not rely on a pressure cooker; simmer it in a regular pot instead.")
    if "no_steamer" in constraints and (head == "idli" or subject_label == "idli sambhar"):
        notes.append("If you do not have a steamer, use a cooker with a rack or choose an instant batter cup version.")
    return "Constraint note: " + " ".join(notes)


def _recipe_ingredient_note(avoided_ingredients: tuple[str, ...]) -> str:
    if not avoided_ingredients:
        return ""

    notes: list[str] = []
    labels = ", ".join(avoided_ingredients)
    notes.append(f"Ingredient adjustment: this version avoids {labels}.")
    if "onion" in avoided_ingredients:
        notes.append("Use tomato, cabbage, pumpkin, curry leaves, or a pinch of hing for flavor instead of onion.")
    if "garlic" in avoided_ingredients:
        notes.append("Skip garlic and lean on ginger, hing, pepper, or herbs for aroma.")
    if "milk" in avoided_ingredients or "dairy" in avoided_ingredients:
        notes.append("Use water or an unsweetened plant-based milk where needed.")
    if "sugar" in avoided_ingredients:
        notes.append("Keep it unsweetened or use a sugar-free option only if it suits your needs.")
    return " ".join(notes)


_RECIPE_DETAIL_VERBS = (
    "blend",
    "drizzle",
    "knead",
    "mix",
    "rest",
    "toast",
    "fry",
    "grill",
    "boil",
    "roast",
    "saute",
    "spread",
    "steam",
    "roll",
    "shape",
    "stuff",
    "bake",
    "cook",
    "simmer",
)

_RECIPE_INGREDIENT_TRAILING_NOISE = {
    "with",
    "and",
    "roughly",
    "about",
    "approximately",
    "approx",
    "using",
    "little",
}

_RECIPE_INGREDIENT_TIME_WORDS = {
    "hour",
    "hours",
    "hr",
    "hrs",
    "minute",
    "minutes",
    "min",
    "mins",
    "second",
    "seconds",
}

_RECIPE_GERUND_LEADS = {
    "blending": "Blend",
    "drizzling": "Drizzle",
    "kneading": "Knead",
    "mixing": "Mix",
    "resting": "Rest",
    "toasting": "Toast",
    "frying": "Fry",
    "grilling": "Grill",
    "boiling": "Boil",
    "roasting": "Roast",
    "sauteing": "Saute",
    "spreading": "Spread",
    "steaming": "Steam",
    "rolling": "Roll",
}

_RECIPE_STEP_LEADING_NOISE = {"and", "then", "next"}


def _feedback_recipe_steps(note: str) -> list[str]:
    normalized = " ".join(note.strip().split())
    if not normalized:
        return []
    chunks = re.split(r",\s*|\.\s*|\bthen\b", normalized)
    steps: list[str] = []
    for chunk in chunks:
        cleaned = chunk.strip(" .")
        if not cleaned:
            continue
        words = cleaned.split()
        while words and words[0].lower() in _RECIPE_STEP_LEADING_NOISE:
            words.pop(0)
        cleaned = " ".join(words)
        if not cleaned:
            continue
        if not any(verb in cleaned.lower() for verb in _RECIPE_DETAIL_VERBS):
            continue
        first_word = cleaned.split()[0].lower()
        if first_word in _RECIPE_GERUND_LEADS:
            cleaned = _RECIPE_GERUND_LEADS[first_word] + cleaned[len(first_word):]
        else:
            cleaned = cleaned[0].upper() + cleaned[1:]
        steps.append(cleaned)
    return steps[:6]


def _feedback_recipe_ingredients(note: str) -> list[str]:
    unit_pattern = re.compile(
        r"\b\d+(?:/\d+)?(?:\s*-\s*\d+(?:/\d+)?)?\s*(?:cup|cups|tbsp|tsp|teaspoon|teaspoons|tablespoon|tablespoons|ml|l|gram|grams|g)\s+[a-z][a-z\s-]{0,25}",
        re.IGNORECASE,
    )
    count_pattern = re.compile(
        r"\b\d+\s+(?:small|medium|large|big|fresh|ripe|chopped|sliced|diced|grated|boiled|dry)?\s*[a-z][a-z\s-]{0,25}",
        re.IGNORECASE,
    )
    def _trim_tokens(tokens: list[str]) -> list[str]:
        for index, token in enumerate(tokens):
            if token.lower() in _RECIPE_INGREDIENT_TRAILING_NOISE and index >= 2:
                return tokens[:index]
        while tokens and tokens[-1].lower() in _RECIPE_INGREDIENT_TRAILING_NOISE:
            tokens.pop()
        return tokens

    matches = []
    unit_spans: list[tuple[int, int]] = []
    for match in unit_pattern.finditer(note):
        cleaned = " ".join(match.group(0).strip(" .,").split())
        tokens = _trim_tokens(cleaned.split())
        if any(token.lower() in _RECIPE_INGREDIENT_TIME_WORDS for token in tokens):
            continue
        cleaned = " ".join(tokens)
        if cleaned:
            matches.append(cleaned)
            unit_spans.append(match.span())

    for match in count_pattern.finditer(note):
        start, end = match.span()
        if any(not (end <= unit_start or start >= unit_end) for unit_start, unit_end in unit_spans):
            continue
        cleaned = " ".join(match.group(0).strip(" .,").split())
        tokens = _trim_tokens(cleaned.split())
        if any(token.lower() in _RECIPE_INGREDIENT_TIME_WORDS for token in tokens):
            continue
        cleaned = " ".join(tokens)
        if cleaned:
            matches.append(cleaned)
    return unique_preserve(matches)[:6]


def _specific_recipe_from_feedback(subject_label: str, feedback_notes: tuple[str, ...]) -> str | None:
    for note in feedback_notes:
        steps = _feedback_recipe_steps(note)
        ingredients = _feedback_recipe_ingredients(note)
        if not steps:
            continue
        if not ingredients and not any(char.isdigit() for char in note):
            continue

        ingredients_line = ", ".join(ingredients) if ingredients else "use the quantities from your earlier tested method"
        step_lines = "\n".join(
            f"{index}. {step.rstrip('.') + '.'}"
            for index, step in enumerate(steps, start=1)
        )
        return (
            f"Simple {subject_label} recipe:\n"
            f"Ingredients: {ingredients_line}.\n"
            "Steps:\n"
            f"{step_lines}\n"
            "Tip: this version follows the detailed method you shared earlier, so keep the timing, texture, and heat steady for the best result."
        )
    return None


# ---------------------------------------------------------------------------
# Generic feedback constraint extraction & application
# ---------------------------------------------------------------------------

# Map of user-phrased concepts -> concrete tokens/patterns to strip or replace
_CURRENCY_ALIASES: dict[str, tuple[str, ...]] = {
    "indian currency": ("INR", "₹", "Rs", "Rs.", "rupee", "rupees", "lakh", "crore"),
    "inr": ("INR", "₹", "Rs", "Rs.", "rupee", "rupees"),
    "rupee": ("INR", "₹", "Rs", "Rs.", "rupee", "rupees", "lakh", "crore"),
    "dollar": ("USD", "$", "dollar", "dollars"),
    "euro": ("EUR", "€", "euro", "euros"),
}

_FEEDBACK_NEGATIVE_RE = re.compile(
    r"\b(?:dont|don't|do not|never|stop|avoid|skip|no|without|"
    r"dont give|don't give|do not give|dont use|don't use|do not use|"
    r"dont suggest|don't suggest|do not suggest|dont show|don't show|do not show|"
    r"dont include|don't include|do not include|remove|exclude|"
    r"dont want|don't want|do not want)\s+"
    r"(?:me\s+|my\s+|any\s+|the\s+|in\s+|with\s+)?"
    r"([a-z][a-z\s,/]{0,60}?)(?:\s*$|\s+(?:please|because|so|but|as|and|next|instead|anymore)\b)",
)

# Common misspellings of verbs that appear between the negative word and the
# actual constraint noun.  When the compound form (e.g. "dont suggest") isn't
# matched because of a typo, the verb leaks into the captured group.  We strip
# these verbs (and an optional trailing "me") so only the real item remains.
_FEEDBACK_VERB_NOISE_RE = re.compile(
    r"^(?:suggest|sugest|sugget|suuggest|suggesst|"
    r"give|giv|givee|"
    r"show|shw|shoow|"
    r"use|ues|uuse|"
    r"include|includ|inlcude|"
    r"want|wnat|waant|"
    r"add|ad|aad)"
    r"(?:\s+me)?\s+",
)


def _extract_feedback_constraints(note: str) -> list[str]:
    """Extract all 'things to avoid' from a feedback note — generic, not ingredient-specific."""
    normalized = normalize_text(note)
    constraints: list[str] = []
    for match in _FEEDBACK_NEGATIVE_RE.finditer(normalized):
        raw = match.group(1).strip()
        # Strip leaked verb typos (e.g. "suuggest me onion" → "onion")
        raw = _FEEDBACK_VERB_NOISE_RE.sub("", raw).strip()
        # split on commas / 'and' / 'or'
        parts = re.split(r"\s*(?:,|/|\band\b|\bor\b)\s*", raw)
        for part in parts:
            cleaned = part.strip()
            if cleaned and len(cleaned) > 1 and cleaned not in {"it", "them", "that", "this", "me"}:
                constraints.append(cleaned)
    return constraints


# Positive preference regex: catches affirmative requests like
# "give me in dollars", "use USD", "show in euros", "convert to dollar"
_FEEDBACK_POSITIVE_RE = re.compile(
    r"\b(?:give|show|use|convert|switch|change|display|put|write|tell)"
    r"(?:\s+(?:me|it|them|this|that))?"
    r"\s+(?:in|to|into|using|with)\s+"
    r"([a-z][a-z\s,/]{0,40}?)(?:\s*$|\s+(?:please|because|so|but|as|and|next|instead)\b)",
)


def _extract_positive_preferences(note: str) -> list[str]:
    """Extract affirmative preferences like 'give me in dollars' → ['dollars']."""
    normalized = normalize_text(note)
    preferences: list[str] = []
    for match in _FEEDBACK_POSITIVE_RE.finditer(normalized):
        raw = match.group(1).strip()
        parts = re.split(r"\s*(?:,|/|\band\b|\bor\b)\s*", raw)
        for part in parts:
            cleaned = part.strip()
            if cleaned and len(cleaned) > 1 and cleaned not in {"it", "them", "that", "this", "me"}:
                preferences.append(cleaned)
    return preferences


def _apply_currency_feedback(response: str, constraints: list[str]) -> str:
    """If a currency-related constraint is found, replace matching amounts."""
    for constraint in constraints:
        tokens_to_strip: list[str] = []
        for alias, symbols in _CURRENCY_ALIASES.items():
            if alias in constraint or any(s.lower() in constraint for s in symbols):
                tokens_to_strip.extend(symbols)
        if tokens_to_strip:
            for symbol in set(tokens_to_strip):
                # Replace "INR 35k-70k" → "$420-$840" style or just strip the symbol
                response = re.sub(
                    rf"\b{re.escape(symbol)}\s*(\d[\d,kK]*(?:\s*[-–]\s*\d[\d,kK]*)?)" ,
                    r"\1",
                    response,
                )
                response = response.replace(symbol, "")
    return response


# Approximate conversion rates for currency substitution
_CURRENCY_CONVERSIONS: dict[str, tuple[str, float]] = {
    "dollar": ("$", 0.012),
    "dollars": ("$", 0.012),
    "usd": ("$", 0.012),
    "euro": ("€", 0.011),
    "euros": ("€", 0.011),
    "eur": ("€", 0.011),
    "pound": ("£", 0.0095),
    "pounds": ("£", 0.0095),
    "gbp": ("£", 0.0095),
}


def _convert_inr_amounts(response: str, target_symbol: str, rate: float) -> str:
    """Replace INR amounts like 'INR 35k-70k' with converted equivalents like '$420-$840'."""
    def _convert_value(val_str: str) -> str:
        val_str = val_str.strip().replace(",", "")
        multiplier = 1
        if val_str.lower().endswith("k"):
            multiplier = 1000
            val_str = val_str[:-1]
        try:
            num = float(val_str) * multiplier * rate
            if num >= 1000:
                return f"{target_symbol}{num/1000:.1f}k"
            return f"{target_symbol}{int(num)}"
        except ValueError:
            return val_str

    def _replace_range(m: re.Match) -> str:
        low = _convert_value(m.group(1))
        high = _convert_value(m.group(2))
        return f"{low}-{high}"

    def _replace_single(m: re.Match) -> str:
        return _convert_value(m.group(1))

    # Match "INR 35k-70k" or "INR 50,000" patterns (with various INR aliases)
    inr_pattern = r"(?:INR|₹|Rs\.?|rupees?)\s*"
    # Range pattern: INR 35k-70k
    response = re.sub(
        inr_pattern + r"(\d[\d,]*[kK]?)\s*[-–]\s*(\d[\d,]*[kK]?)",
        _replace_range,
        response,
        flags=re.IGNORECASE,
    )
    # Single value: INR 50000
    response = re.sub(
        inr_pattern + r"(\d[\d,]*[kK]?)",
        _replace_single,
        response,
        flags=re.IGNORECASE,
    )
    return response


def _apply_positive_currency_preference(response: str, preferences: list[str]) -> str:
    """If user asked for a specific currency (e.g. 'give me in dollars'), convert INR amounts."""
    for pref in preferences:
        pref_lower = pref.lower().strip()
        for currency_key, (symbol, rate) in _CURRENCY_CONVERSIONS.items():
            if currency_key in pref_lower:
                response = _convert_inr_amounts(response, symbol, rate)
                return response
    return response


def _apply_generic_keyword_scrub(response: str, constraints: list[str]) -> str:
    """Last-resort pass: for each constraint, find and remove/flag occurrences in the response."""
    for constraint in constraints:
        # Skip very short or very generic words
        words = [w for w in constraint.split() if len(w) > 2]
        if not words:
            continue
        phrase = r"\s+".join(re.escape(w) for w in words)
        # Try to remove lines or inline mentions that contain the phrase
        response = re.sub(
            rf"(?i)\b{phrase}(s|ed|ing|er)?\b",
            "[removed per feedback]",
            response,
        )
    return response


def _apply_feedback_memory(response: str, feedback_notes: tuple[str, ...]) -> str:
    if not feedback_notes:
        return response

    summaries: list[str] = []
    for note in feedback_notes:
        avoided_ingredients = extract_ingredient_avoidances(note)
        if avoided_ingredients:
            summaries.append(f"avoiding {', '.join(avoided_ingredients)}")
            continue

        constraints = extract_kitchen_constraints(note)
        if constraints:
            readable = ", ".join(item.replace("_", " ") for item in constraints)
            summaries.append(readable)
            continue

        positive = _extract_positive_preferences(note)
        if positive:
            summaries.append(", ".join(positive))
            continue

        generic = _extract_feedback_constraints(note)
        if generic:
            summaries.append(", ".join(generic))
            continue

        summaries.append(" ".join(note.split()))

    summary_text = "; ".join(unique_preserve([summary for summary in summaries if summary]))
    if not summary_text:
        return response
    return f"{response}\nI also adjusted this using your earlier feedback: {summary_text}."


@dataclass
class SpecialistAgent:
    name: str
    domain: str
    defaults: tuple[str, ...]
    knowledge_retriever: KnowledgeRetriever | None = None

    def propose(
        self,
        query: str,
        recalled_memory: list[MemoryRecord],
        allow_live: bool = False,
    ) -> AgentProposal:
        normalized = normalize_text(query)
        tokens = token_set(query)
        hits = tuple(sorted(tokens & DOMAIN_KEYWORDS[self.domain]))[:4]
        vector = encode_query_vector(query)
        base_confidence = 0.18 + (0.45 * vector.get(self.domain, 0.0))
        live_response = self._build_live_response(
            query=query,
            normalized=normalized,
            tokens=tokens,
            allow_live=allow_live,
        )
        if live_response is not None:
            base_confidence += 0.22
        if self.domain == "food":
            if _is_drink_query(normalized, tokens) or extract_health_conditions(query):
                base_confidence += 0.24
            if _is_recipe_query(normalized, tokens):
                base_confidence += 0.26
            elif _is_meal_suggestion_query(normalized, tokens):
                base_confidence += 0.26
        if self.domain == "business":
            if _is_idea_query(normalized, tokens):
                base_confidence += 0.24
            elif tokens & {"business", "customer", "market", "margin", "pricing", "revenue", "roi", "startup", "subscription"}:
                base_confidence += 0.2
        if self.domain == "coding" and tokens & {"bug", "code", "coding", "debug", "error", "function", "python", "script", "software"}:
            base_confidence += 0.18
        if self.domain == "research" and tokens & {"baseline", "compare", "dataset", "evaluate", "experiment", "hypothesis", "metric", "model", "research"}:
            base_confidence += 0.18
        query_embedding = _ROUTER_ENCODER.encode(query)
        memory_relevant = any(
            overlap_score(tokens, record.keywords) >= 0.08
            or (record.query and normalize_text(record.query) == normalized)
            or (
                record.vector
                and VectorEncoder.cosine_similarity(query_embedding, record.vector) >= 0.34
            )
            for record in recalled_memory
        )
        confidence = clamp(base_confidence + (0.12 if memory_relevant else 0.0), 0.05, 0.92)
        focus_terms = _focus_terms(tokens, hits, self.defaults)
        base_response = live_response or self._build_base_response(
            query,
            normalized,
            tokens,
            focus_terms,
            recalled_memory,
        )
        memory_response = self._build_memory_response(
            query,
            normalized,
            tokens,
            focus_terms,
            recalled_memory,
            live_response,
        )
        rationale = f"{self.name} matched {', '.join(hits) or 'domain priors'}"
        return AgentProposal(
            agent_name=self.name,
            domain=self.domain,
            confidence=round(confidence, 4),
            keywords_hit=hits,
            base_response=base_response,
            memory_response=memory_response,
            rationale=rationale,
        )

    def _build_base_response(
        self,
        query: str,
        normalized: str,
        tokens: set[str],
        focus_terms: list[str],
        recalled_memory: list[MemoryRecord],
    ) -> str:
        joined = ", ".join(focus_terms[:3])
        if self.domain == "food":
            health_conditions = extract_health_conditions(query)
            preference = extract_user_preference(query)
            if preference is not None and health_conditions:
                return self._build_health_aware_drink_response(
                    tokens=tokens,
                    health_conditions=health_conditions,
                    preferred_beverage=str(preference["item"]),
                    from_memory=False,
                )
            if preference is not None:
                rainy_note = " for rainy days" if has_rain_signal(tokens) else ""
                return (
                    f"Noted: I will remember that you like {preference['item']}{rainy_note}. "
                    "When you ask for a drink suggestion later, I will use that preference."
                )
            if _is_drink_query(normalized, tokens):
                if health_conditions:
                    return self._build_health_aware_drink_response(
                        tokens=tokens,
                        health_conditions=health_conditions,
                        preferred_beverage=None,
                        from_memory=False,
                    )
                if has_rain_signal(tokens):
                    return (
                        "Food suggestion: since it is raining, go for a hot coffee or chai. "
                        "If you want one simple pick, hot coffee is a strong choice."
                    )
                return (
                    "Food suggestion: choose a comforting drink such as coffee, tea, or chai "
                    "based on your mood and the weather."
                )
            if _is_recipe_query(normalized, tokens):
                return self._build_recipe_response(
                    query,
                    focus_terms,
                    recalled_memory,
                    from_memory=False,
                )
            if _is_meal_suggestion_query(normalized, tokens):
                return self._build_meal_suggestion(
                    query, normalized, tokens, recalled_memory, from_memory=False,
                )
            return (
                "Food plan: deliver a quick meal recommendation around "
                f"{joined}, with simple prep, protein balance, and a budget-aware swap."
            )
        if self.domain == "business":
            if _is_idea_query(normalized, tokens):
                return self._build_business_ideas_response(normalized, tokens)
            return (
                "Business plan: shape a pilot around "
                f"{joined}, validate margin early, and package the ROI story for customers and partners."
            )
        if self.domain == "coding":
            return (
                "Coding plan: implement a script or function around "
                f"{joined}, handle edge cases and bugs, and keep the logic clean."
            )
        return (
            "Research plan: define a hypothesis around "
            f"{joined}, keep a strong baseline, and track the core metric in a lightweight experiment."
        )

    def _build_meal_suggestion(
        self,
        query: str,
        normalized: str,
        tokens: set[str],
        recalled_memory: list[MemoryRecord],
        from_memory: bool,
    ) -> str:
        food_subject = _extract_food_subject(query) or "meal"
        # strip " recipe" or " recipes" from the end for a cleaner sentence
        clean_subject = food_subject.replace(" recipes", "").replace(" recipe", "")
        
        # Override with a preferred dish from memory if one exists and we are heavily influenced by memory
        if from_memory:
            preferred_dish = _preferred_dish_from_memory(recalled_memory, normalized)
            if preferred_dish:
                clean_subject = preferred_dish

        memory_note = " I kept your earlier food preferences in mind." if from_memory else ""
        
        avoided = _avoided_ingredients_from_memory(recalled_memory)
        avoid_str = ""
        if avoided and from_memory:
            avoid_str = f" Make sure not to use any {', '.join(avoided)}."
        
        if "breakfast" in food_subject or "morning" in normalized:
            plan = f"Food suggestion: for breakfast, {clean_subject} is a great choice. Go for something light and energetic."
        elif "lunch" in food_subject or "afternoon" in normalized:
            plan = f"Food suggestion: for lunch, {clean_subject} works well. Keep it balanced with good protein."
        elif "dinner" in food_subject or "night" in normalized or "evening" in normalized:
            plan = f"Food suggestion: for dinner, {clean_subject} is ideal. Keep it light and easy to digest."
        elif "snack" in food_subject:
            plan = f"Food suggestion: a quick {clean_subject} is perfect. Keep it simple and quick to prepare."
        else:
            plan = f"Food suggestion: {clean_subject} sounds like a great idea. I can give you a specific recipe for it if you want."
            
        return f"{plan}{avoid_str}{memory_note}"

    def _build_memory_response(
        self,
        query: str,
        normalized: str,
        tokens: set[str],
        focus_terms: list[str],
        recalled_memory: list[MemoryRecord],
        live_response: str | None,
    ) -> str | None:
        snippet = _memory_snippet(recalled_memory, self.domain)
        feedback_notes = _feedback_notes_from_memory(recalled_memory, self.domain)
        exact_feedback_notes = _exact_feedback_notes_from_memory(recalled_memory, query, self.domain)
        if not snippet and not feedback_notes:
            return None
        if self.domain == "food" and _is_drink_query(normalized, tokens):
            health_conditions = extract_health_conditions(query)
            query_health_is_generic = health_conditions == ("generic_condition",)
            if not health_conditions or query_health_is_generic:
                health_conditions = _health_conditions_from_memory(recalled_memory)
            if health_conditions and health_conditions != ("generic_condition",):
                preferred = _preferred_beverage_from_memory(recalled_memory)
                response = self._build_health_aware_drink_response(
                    tokens=tokens,
                    health_conditions=health_conditions,
                    preferred_beverage=preferred,
                    from_memory=True,
                )
                return _apply_feedback_memory(response, feedback_notes)
            preferred = _preferred_beverage_from_memory(recalled_memory)
            if preferred:
                rainy_note = " while it is raining" if has_rain_signal(tokens) else ""
                hot_prefix = "hot " if preferred in HOT_BEVERAGES else ""
                response = (
                    f"You told me you like {preferred}, so {hot_prefix}{preferred}{rainy_note} "
                    "is the best match for this moment."
                )
                return _apply_feedback_memory(response, feedback_notes)
            if exact_feedback_notes:
                response = self._build_base_response(
                    query,
                    normalized,
                    tokens,
                    focus_terms,
                    recalled_memory,
                )
                return _apply_feedback_memory(response, exact_feedback_notes)
            return None
        if self.domain == "food" and _is_recipe_query(normalized, tokens):
            response = self._build_recipe_response(
                query,
                focus_terms,
                recalled_memory,
                from_memory=True,
            )
            subject = _recipe_subject(query, focus_terms)
            subject_label = _title_case_subject(subject)
            if subject_label in {"idli sambar", "idli sambhar", "idlisambhar", "idlisambar"}:
                subject_label = "idli sambhar"
            feedback_specific_recipe = _specific_recipe_from_feedback(subject_label, exact_feedback_notes)
            if feedback_specific_recipe:
                return response
            return _apply_feedback_memory(response, feedback_notes)
        if self.domain == "food" and _is_meal_suggestion_query(normalized, tokens):
            response = self._build_meal_suggestion(
                query,
                normalized,
                tokens,
                recalled_memory,
                from_memory=True,
            )
            return _apply_feedback_memory(response, feedback_notes)
        if live_response is not None:
            snippet_text = f"\nMemory note: related earlier notes were {snippet}." if snippet else ""
            response = f"{live_response}{snippet_text}"
            return _apply_feedback_memory(response, feedback_notes)
        if self.domain == "business" and _is_idea_query(normalized, tokens):
            snippet_text = f"Business ideas with memory: build on {snippet}.\n" if snippet else ""
            response = f"{snippet_text}{self._build_business_ideas_response(normalized, tokens)}"
            return _apply_feedback_memory(response, feedback_notes)
            
        joined = ", ".join(focus_terms[:3])
        if snippet:
            response = (
                f"{self.domain.title()} plan with memory: build on {snippet}. "
                f"Next, reinforce {joined} so the answer stays consistent with earlier wins."
            )
        else:
            response = self._build_base_response(query, normalized, tokens, focus_terms, recalled_memory)
            
        return _apply_feedback_memory(response, feedback_notes)

    def _build_live_response(
        self,
        query: str,
        normalized: str,
        tokens: set[str],
        allow_live: bool,
    ) -> str | None:
        if not allow_live or self.knowledge_retriever is None:
            return None

        if self.domain == "food" and _is_drink_query(normalized, tokens):
            health_conditions = extract_health_conditions(query)
            if not health_conditions or health_conditions == ("generic_condition",):
                health_topic = extract_health_topic_candidate(query)
                if health_topic:
                    live_health = self.knowledge_retriever.lookup_health(health_topic)
                    if live_health is not None:
                        warm_note = (
                            "Safer drink baseline: choose a mild warm unsweetened drink or plain water while you verify any condition-specific restrictions."
                            if has_rain_signal(tokens)
                            else "Safer drink baseline: choose plain water or a mild unsweetened drink until you confirm any condition-specific restrictions."
                        )
                        return (
                            f"Live health context from {live_health.source_name} on {live_health.title}:\n"
                            f"{live_health.summary}\n"
                            f"{warm_note}\n"
                            f"Source: {live_health.url}"
                        )
            return None

        if not _is_live_lookup_query(normalized):
            return None

        if not tokens & DOMAIN_KEYWORDS[self.domain]:
            return None

        subject = _live_lookup_subject(query)
        if not subject:
            return None
        context = self.knowledge_retriever.lookup(subject, self.domain)
        if context is None:
            return None
        return self._format_live_context(context)

    def _format_live_context(self, context: LiveKnowledge) -> str:
        if self.domain == "research":
            return (
                f"Live research context from {context.source_name} on {context.title}:\n"
                f"{context.summary}\n"
                "Use this as the concept grounding, then define the hypothesis, baseline, and evaluation metric.\n"
                f"Source: {context.url}"
            )
        if self.domain == "business":
            return (
                f"Live business context from {context.source_name} on {context.title}:\n"
                f"{context.summary}\n"
                "Use this as the baseline concept before deciding pricing, pilot structure, or market approach.\n"
                f"Source: {context.url}"
            )
        return (
            f"Live food context from {context.source_name} on {context.title}:\n"
            f"{context.summary}\n"
            "If you want, I can turn this into a recipe, ingredient list, or meal suggestion next.\n"
            f"Source: {context.url}"
        )

    def _build_business_ideas_response(self, normalized: str, tokens: set[str]) -> str:
        budget = _budget_label(normalized)
        future_ready = _future_signal(tokens, normalized)
        budget_sensitive = _budget_signal(tokens, normalized)

        idea_one_budget = "INR 35k-70k" if budget_sensitive else "INR 50k-90k"
        idea_two_budget = "INR 45k-90k" if budget_sensitive else "INR 60k-100k"
        idea_three_budget = "INR 25k-60k" if budget_sensitive else "INR 40k-80k"
        future_line = "These lean toward AI, automation, and climate-adjacent demand." if future_ready else "These lean toward recurring revenue and sectors with long-term demand."

        return (
            f"Startup ideas {budget}:\n"
            f"1. AI automation studio for local businesses.\n"
            f"   Budget: {idea_one_budget} for outreach, demos, landing page, and basic tooling.\n"
            "   Why it works: you sell workflow automations for clinics, coaching centers, and retailers before building heavy software.\n"
            "2. EV and solar service marketplace for tier-two cities.\n"
            f"   Budget: {idea_two_budget} for field validation, simple booking site, partner onboarding, and local ads.\n"
            "   Why it works: it rides future energy adoption while staying asset-light.\n"
            "3. Niche micro-SaaS for student career prep or admissions workflows.\n"
            f"   Budget: {idea_three_budget} for MVP, distribution content, and pilot customer support.\n"
            "   Why it works: it can start as a service-plus-software model and evolve into recurring subscription revenue.\n"
            f"Best fit for your prompt: start with idea 1 or 3 if you want faster execution. {future_line}"
        )

    def _build_health_aware_drink_response(
        self,
        tokens: set[str],
        health_conditions: tuple[str, ...],
        preferred_beverage: str | None,
        from_memory: bool,
    ) -> str:
        ordered_conditions = _ordered_conditions(health_conditions)
        primary = ordered_conditions[0]
        guidance = HEALTH_GUIDANCE.get(primary, HEALTH_GUIDANCE["generic_condition"])
        lead = guidance["drink"]

        if preferred_beverage and primary not in {"gerd_acidity", "kidney_disease"}:
            if primary in {"diabetes", "pcos"} and preferred_beverage in HOT_BEVERAGES:
                lead = f"an unsweetened {preferred_beverage}"
            elif primary in {"hypertension", "heart_condition"} and preferred_beverage in {"coffee", "espresso"}:
                lead = f"a lighter or decaf version of {preferred_beverage}"
            elif primary != "generic_condition":
                lead = f"{preferred_beverage}, adjusted to stay gentle for your health needs"

        weather_note = " Since it is raining, a warm option makes the most sense." if has_rain_signal(tokens) else ""
        memory_note = " I remembered this from your earlier health or preference notes." if from_memory else ""
        condition_label = ", ".join(condition.replace("_", " ") for condition in ordered_conditions)

        return (
            f"Health-aware drink suggestion: choose {lead}.{weather_note}\n"
            f"Health factors noticed: {condition_label}.\n"
            f"Precautions: avoid {guidance['avoid']}.\n"
            f"Why: {guidance['reason']}.{memory_note}\n"
            "This is general guidance only, so if your doctor has given you specific fluid, sugar, caffeine, or medication rules, follow those first."
        )

    def _build_recipe_response(
        self,
        query: str,
        focus_terms: list[str],
        recalled_memory: list[MemoryRecord],
        from_memory: bool,
    ) -> str:
        subject = _recipe_subject(query, focus_terms)
        subject_label = _title_case_subject(subject)
        head = _recipe_head(subject)
        if subject_label in {"idli sambar", "idli sambhar", "idlisambhar", "idlisambar"}:
            subject_label = "idli sambhar"

        constraints = tuple(
            unique_preserve(
                list(_kitchen_constraints_from_memory(recalled_memory))
                + list(extract_kitchen_constraints(query))
            )
        )
        avoided_ingredients = tuple(
            unique_preserve(
                list(_avoided_ingredients_from_memory(recalled_memory))
                + list(extract_ingredient_avoidances(query))
            )
        )
        exact_feedback_notes = _exact_feedback_notes_from_memory(recalled_memory, query, self.domain)
        avoided_tokens = {
            token
            for ingredient in avoided_ingredients
            for token in ingredient.split()
        }
        avoid_onion = "onion" in avoided_tokens
        avoid_tomato = "tomato" in avoided_tokens
        avoid_garlic = "garlic" in avoided_tokens
        avoid_milk = "milk" in avoided_tokens or "dairy" in avoided_tokens
        avoid_sugar = "sugar" in avoided_tokens
        avoid_butter = "butter" in avoided_tokens
        avoid_chaat_masala = "chaat masala" in avoided_ingredients or (
            "chaat" in avoided_tokens and "masala" in avoided_tokens
        )

        constraint_note = _recipe_constraint_note(constraints, subject_label, head)
        ingredient_note = _recipe_ingredient_note(avoided_ingredients)
        recipe_notes = "\n".join(note for note in (ingredient_note, constraint_note) if note)
        memory_note = "\nI also kept your earlier food context in mind." if from_memory else ""

        feedback_specific_recipe = _specific_recipe_from_feedback(subject_label, exact_feedback_notes)
        if feedback_specific_recipe:
            base = feedback_specific_recipe
            if recipe_notes:
                base += f"\n{recipe_notes}"
            return f"{base}{memory_note}"

        if subject_label == "idli sambhar":
            sambhar_vegetables: list[str] = []
            if not avoid_onion:
                sambhar_vegetables.append("1 chopped onion")
            if not avoid_tomato:
                sambhar_vegetables.append("1 chopped tomato")
            if not sambhar_vegetables:
                sambhar_vegetables.append("1/2 cup chopped pumpkin or carrot")
            vegetable_line = ", ".join(sambhar_vegetables)
            simmer_items: list[str] = []
            if not avoid_onion:
                simmer_items.append("onion")
            if not avoid_tomato:
                simmer_items.append("tomato")
            if not simmer_items:
                simmer_items.append("pumpkin or carrot")
            simmer_line = ", ".join(simmer_items)
            base = (
                "Simple idli sambhar recipe:\n"
                "Ingredients for idli: 2 cups idli batter and a little oil for greasing.\n"
                f"Ingredients for sambhar: 1/2 cup toor dal, {vegetable_line}, 1 to 2 tbsp sambhar powder, 1/4 tsp turmeric, salt, tamarind water, and a simple tempering.\n"
                "Steps:\n"
                "1. Pour the idli batter into greased idli moulds and steam for 10 to 12 minutes.\n"
                "2. Cook the dal until soft.\n"
                f"3. Add {simmer_line}, turmeric, sambhar powder, salt, and tamarind water, then simmer.\n"
                "4. Add the cooked dal and boil for a few more minutes.\n"
                "5. Finish with a tempering of mustard seeds, curry leaves, and dried chili if you like.\n"
                "6. Serve the hot idli with sambhar.\n"
                "Tip: the sambhar should be pourable, not too thick."
            )
            if recipe_notes:
                base += f"\n{recipe_notes}"
            return f"{base}{memory_note}"

        if head == "rice":
            base = (
                f"Simple {subject_label} recipe:\n"
                "Ingredients: 1 cup rice, 2 cups water, 1/2 tsp salt, and 1 tsp oil or ghee if you want.\n"
                "Steps:\n"
                "1. Rinse the rice 2 or 3 times until the water is less cloudy.\n"
                "2. Add rice, water, salt, and oil to a pan.\n"
                "3. Bring it to a boil, then cover and cook on low heat for 12 to 15 minutes.\n"
                "4. Turn off the heat and let it rest covered for 5 minutes.\n"
                "5. Fluff with a fork and serve.\n"
                "Tip: use a 1:2 rice-to-water ratio for regular white rice."
            )
            if recipe_notes:
                base += f"\n{recipe_notes}"
            return f"{base}{memory_note}"

        if head in {"golgappa", "puri"}:
            base = (
                f"Simple {subject_label} recipe:\n"
                "Ingredients: 1 cup semolina, 2 tbsp maida or atta, 1 tbsp oil, a pinch of salt, about 1/3 cup warm water, and oil for frying.\n"
                "Steps:\n"
                "1. Mix semolina, maida or atta, salt, and oil in a bowl.\n"
                "2. Add warm water little by little and knead into a stiff dough.\n"
                "3. Cover and rest the dough for 20 to 30 minutes.\n"
                "4. Roll it thin, cut small discs, and keep them covered.\n"
                "5. Fry on medium-high heat until the puris puff and turn crisp.\n"
                "6. Cool fully before filling and serving.\n"
                "Tip: keep the dough firm and the rolled discs thin, or the puris may not puff properly."
            )
            if recipe_notes:
                base += f"\n{recipe_notes}"
            return f"{base}{memory_note}"

        if head in BATTER_FRY_RECIPE_HEADS:
            filling = "sliced onion or potato"
            if subject and subject != head:
                filling = subject.replace(head, "").strip() or filling
            if avoid_onion and "onion" in filling:
                filling = filling.replace("sliced onion or potato", "sliced potato or spinach").replace("onion", "potato")
            base = (
                f"Simple {subject_label} recipe:\n"
                f"Ingredients: 1 cup besan, 1 cup {filling}, 1 green chili, a little coriander, 1/2 tsp turmeric, 1/2 tsp chili powder, salt, water, and oil for frying.\n"
                "Steps:\n"
                "1. Mix besan, salt, turmeric, chili powder, and coriander in a bowl.\n"
                f"2. Add the {filling} and mix well.\n"
                "3. Add a little water at a time to make a thick coating batter.\n"
                "4. Heat oil on medium flame.\n"
                "5. Drop small portions into the oil and fry until golden and crisp.\n"
                "6. Remove onto paper and serve hot with chutney or chai.\n"
                f"Tip: keep the batter thick so the {head or 'pakoda'} stays crisp."
            )
            if recipe_notes:
                base += f"\n{recipe_notes}"
            return f"{base}{memory_note}"

        if head in ASSEMBLED_RECIPE_HEADS:
            filling = "simple vegetable filling"
            if subject and subject != head:
                filling = subject.replace(head, "").strip() or filling
            seasoning_line = "and a little chaat masala or herbs if you like"
            sprinkle_step = "3. Sprinkle a little salt, pepper, and chaat masala.\n"
            if avoid_chaat_masala:
                seasoning_line = "and a little extra herbs if you like"
                sprinkle_step = "3. Sprinkle a little salt, pepper, and mixed herbs.\n"
            spread_ing = "sauce or chutney" if avoid_butter else "butter or sauce"
            spread_step = "Spread your preferred sauce or chutney" if avoid_butter else "Spread butter and chutney or ketchup"
            base = (
                f"Simple {subject_label} recipe:\n"
                f"Ingredients: 4 bread or wrap pieces, 1 cup {filling}, 1 to 2 tsp {spread_ing}, salt, pepper, {seasoning_line}.\n"
                "Steps:\n"
                f"1. {spread_step} on the bread slices.\n"
                "2. Add the filling in an even layer.\n"
                f"{sprinkle_step}"
                f"4. Close the {head or 'sandwich'} and toast it on a pan or sandwich maker until crisp, or serve it plain.\n"
                "5. Cut and serve hot.\n"
                f"Tip: do not overfill it, or the {head or 'sandwich'} will break while toasting."
            )
            if recipe_notes:
                base += f"\n{recipe_notes}"
            return f"{base}{memory_note}"

        if head in {"tea", "chai"}:
            tea_liquid = "1 cup water, 1/2 cup milk" if not avoid_milk else "1 1/2 cups water or unsweetened plant milk"
            sweetener = "sugar to taste" if not avoid_sugar else "no sugar, or a sugar-free option only if you want"
            if not avoid_milk and not avoid_sugar:
                tea_step = "3. Add milk and sugar, then simmer for 2 more minutes.\n"
            elif avoid_milk and avoid_sugar:
                tea_step = "3. Add your plant milk if using it and keep it unsweetened, then simmer for 2 more minutes.\n"
            elif avoid_milk:
                tea_step = "3. Add your plant milk if using it, sweeten only if you want, and simmer for 2 more minutes.\n"
            else:
                tea_step = "3. Add milk and keep it unsweetened, then simmer for 2 more minutes.\n"
            base = (
                f"Simple {subject_label} recipe:\n"
                f"Ingredients: {tea_liquid}, 1 tsp tea leaves, {sweetener}, and a little ginger or cardamom if you like.\n"
                "Steps:\n"
                "1. Boil the water with ginger or cardamom.\n"
                "2. Add tea leaves and simmer for 1 minute.\n"
                f"{tea_step}"
                "4. Strain and serve hot."
            )
            if recipe_notes:
                base += f"\n{recipe_notes}"
            return f"{base}{memory_note}"

        if head in {"coffee", "cocoa"}:
            base_liquid = "1 cup hot water or milk" if not avoid_milk else "1 cup hot water or unsweetened plant milk"
            sweetener = "and sugar if wanted" if not avoid_sugar else "and no added sugar unless you choose a sugar-free option"
            base = (
                f"Simple {subject_label} recipe:\n"
                f"Ingredients: {base_liquid}, 1 to 2 tsp {head}, {sweetener}.\n"
                "Steps:\n"
                "1. Heat the liquid.\n"
                f"2. Add {head} and stir well.\n"
                "3. Sweeten only if it fits your preference.\n"
                "4. Serve hot."
            )
            if recipe_notes:
                base += f"\n{recipe_notes}"
            return f"{base}{memory_note}"

        if head in SIMMER_RECIPE_HEADS:
            if head in {"noodles", "pasta"}:
                base = (
                    f"Simple {subject_label} recipe:\n"
                    f"Ingredients: 1 cup {subject_label}, 3 cups water, 1 tsp oil or butter, salt, and chopped vegetables or seasoning if you like.\n"
                    "Steps:\n"
                    "1. Bring the water to a boil with a little salt.\n"
                    f"2. Add the {subject_label} and cook until just soft.\n"
                    "3. Drain extra water if needed.\n"
                    "4. Toss with butter, seasoning, and vegetables or sauce.\n"
                    "5. Serve hot.\n"
                    f"Tip: do not overcook the {head}, or it will turn mushy."
                )
                if recipe_notes:
                    base += f"\n{recipe_notes}"
                return f"{base}{memory_note}"
            if head == "dal":
                tempering_line = "4. Add salt and a simple tempering of oil, cumin, garlic, or chili if you like.\n"
                if avoid_garlic:
                    tempering_line = "4. Add salt and a simple tempering of oil, cumin, hing, ginger, or chili if you like.\n"
                base = (
                    f"Simple {subject_label} recipe:\n"
                    "Ingredients: 1 cup dal, 3 cups water, 1/2 tsp turmeric, salt, and 1 to 2 tsp oil or ghee.\n"
                    "Steps:\n"
                    "1. Wash the dal well.\n"
                    "2. Add it to a cooker or pan with water and turmeric.\n"
                    "3. Cook until soft.\n"
                    f"{tempering_line}"
                    "5. Simmer for 2 more minutes and serve hot.\n"
                    "Tip: add more water if you want a thinner dal."
                )
                if recipe_notes:
                    base += f"\n{recipe_notes}"
                return f"{base}{memory_note}"
            if head == "oats":
                oats_liquid = "2 cups water or milk" if not avoid_milk else "2 cups water or unsweetened plant milk"
                base = (
                    f"Simple {subject_label} recipe:\n"
                    f"Ingredients: 1 cup oats, {oats_liquid}, a pinch of salt, and optional fruit, nuts, or spices.\n"
                    "Steps:\n"
                    "1. Heat the liquid in a pan.\n"
                    "2. Add oats and stir well.\n"
                    "3. Cook for 3 to 5 minutes until soft.\n"
                    "4. Add toppings or seasoning.\n"
                    "5. Serve warm.\n"
                    "Tip: stir often so the oats stay creamy."
                )
                if recipe_notes:
                    base += f"\n{recipe_notes}"
                return f"{base}{memory_note}"
            base_liquid = "2 cups water"
            base = (
                f"Simple {subject_label} recipe:\n"
                f"Ingredients: 1 cup {subject_label}, {base_liquid}, 1 to 2 tsp oil or butter, salt, and basic spices if you like.\n"
                "Steps:\n"
                "1. Wash the main ingredient well before cooking.\n"
                "2. Add it to a pan with water, salt, and oil or butter.\n"
                "3. Cook on medium or low heat until soft and fully done.\n"
                "4. Stir once or twice and adjust water if needed.\n"
                "5. Finish with a simple tempering, herbs, or sauce and serve hot.\n"
                f"Tip: keep the heat moderate so the {head} cooks evenly."
            )
            if recipe_notes:
                base += f"\n{recipe_notes}"
            return f"{base}{memory_note}"

        base = (
            f"Simple {subject_label} recipe:\n"
            f"Ingredients: 1 to 2 cups {subject_label}, 1 to 2 tsp oil or butter, salt, pepper or basic spices, and an optional garnish or sauce.\n"
            "Steps:\n"
            "1. Prep the main ingredient into an easy-to-cook form.\n"
            "2. Heat a pan and add oil or butter.\n"
            "3. Cook or assemble the dish with simple seasoning.\n"
            "4. Taste once, adjust the texture or spice, and finish the dish.\n"
            "5. Serve hot or fresh depending on the dish.\n"
            f"Tip: if you want, I can make this recipe more specific with exact ingredients and timing for {subject_label}."
        )
        if recipe_notes:
            base += f"\n{recipe_notes}"
        return f"{base}{memory_note}"


class MemoryAgent:
    name = "memory"
    domain = "memory"

    def propose(
        self,
        query: str,
        recalled_memory: list[MemoryRecord],
        allow_live: bool = False,
    ) -> AgentProposal:
        _ = allow_live
        normalized = normalize_text(query)
        tokens = token_set(query)
        identity = extract_user_identity(query)
        recall_intent = any(cue in normalized for cue in MEMORY_CUES)
        wants_profile_recall = "name" in tokens or "who am i" in normalized
        specialist_overlap = bool(
            tokens
            & (
                DOMAIN_KEYWORDS["food"]
                | DOMAIN_KEYWORDS["business"]
                | DOMAIN_KEYWORDS["coding"]
                | DOMAIN_KEYWORDS["research"]
            )
        )
        specialist_intent = (
            _is_recipe_query(normalized, tokens)
            or _is_drink_query(normalized, tokens)
            or _is_meal_suggestion_query(normalized, tokens)
            or _is_idea_query(normalized, tokens)
            or (specialist_overlap and not recall_intent and not wants_profile_recall and identity is None)
        )
        remembered_name = _remembered_name_from_memory(recalled_memory)
        memory_bonus = 0.2 if recalled_memory and not specialist_intent else 0.04 if recalled_memory else 0.0
        confidence = (
            (0.08 if specialist_intent else 0.2)
            + (0.45 if recall_intent else 0.0)
            + memory_bonus
            + (0.58 if identity is not None else 0.0)
            + (0.18 if wants_profile_recall and remembered_name else 0.0)
        )
        if recall_intent and recalled_memory:
            confidence += 0.18
        confidence = clamp(confidence, 0.05, 0.96)
        if identity is not None:
            name = str(identity["name"])
            base_response = (
                f"Noted: your name is {name}. I will remember that for later questions."
            )
            memory_response = (
                f"Profile memory saved: your name is {name}, and I will use it in later recall-style questions."
            )
        elif remembered_name and wants_profile_recall:
            base_response = f"You told me your name is {remembered_name}."
            memory_response = (
                f"Profile recall with context: your saved name is {remembered_name}."
            )
        elif recalled_memory:
            snippet = _memory_snippet(recalled_memory)
            base_response = f"Memory recall: the strongest prior notes are {snippet}."
            memory_response = (
                f"Memory recall with context: retrieve {snippet} and use it as the authoritative prior answer."
            )
        else:
            base_response = (
                "Memory recall: there is no strong matching note yet, so the system should defer to the best specialist."
            )
            memory_response = None
        return AgentProposal(
            agent_name=self.name,
            domain=self.domain,
            confidence=round(confidence, 4),
            keywords_hit=tuple(sorted(tokens & DOMAIN_KEYWORDS["memory"]))[:4],
            base_response=base_response,
            memory_response=memory_response,
            rationale="memory agent checks for recall cues and shared knowledge overlap",
        )


def build_default_agents(
    knowledge_retriever: KnowledgeRetriever | None = None,
) -> dict[str, SpecialistAgent | MemoryAgent]:
    retriever = knowledge_retriever or InternetKnowledgeRetriever()
    return {
        "food": SpecialistAgent("food", "food", ("meal", "protein", "budget"), knowledge_retriever=retriever),
        "business": SpecialistAgent("business", "business", ("pricing", "pilot", "margin"), knowledge_retriever=retriever),
        "coding": SpecialistAgent("coding", "coding", ("script", "function", "debug"), knowledge_retriever=retriever),
        "research": SpecialistAgent("research", "research", ("hypothesis", "metric", "baseline"), knowledge_retriever=retriever),
        "memory": MemoryAgent(),
    }
