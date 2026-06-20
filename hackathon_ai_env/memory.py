from __future__ import annotations

import math
import time

from .models import MemoryRecord, QueryScenario
from .vector import VectorEncoder
from .utils import (
    clamp,
    extract_health_conditions,
    extract_ingredient_avoidances,
    extract_kitchen_constraints,
    parse_memory_sharing_preferences,
    extract_user_identity,
    extract_user_preference,
    has_drink_signal,
    has_recipe_signal,
    normalize_text,
    overlap_score,
    tokenize,
    unique_preserve,
)


class SharedKnowledgeSpace:
    def __init__(
        self,
        decay: float = 0.92,
        learning_rate: float = 0.35,
        time_decay_days: float = 30.0,
    ) -> None:
        self.decay = decay
        self.learning_rate = learning_rate
        self.time_decay_days = time_decay_days
        self.records: dict[str, MemoryRecord] = {}
        self.step = 0
        self.encoder = VectorEncoder()
        self.sharing_rules: dict[str, set[str]] = {}

    def reset(self) -> None:
        self.records = {}
        self.step = 0
        self.sharing_rules = {}

    def snapshot(self) -> dict[str, float]:
        weights = [record.weight for record in self.records.values()]
        bank_counts: dict[str, int] = {}
        for record in self.records.values():
            bank_counts[record.domain] = bank_counts.get(record.domain, 0) + 1
        return {
            "count": float(len(self.records)),
            "total_weight": round(sum(weights), 4),
            "top_weight": round(max(weights, default=0.0), 4),
            "banks": {domain: float(count) for domain, count in sorted(bank_counts.items())},
        }

    def to_dict(self) -> dict[str, object]:
        records = []
        for record in self.records.values():
            records.append(
                {
                    "key": record.key,
                    "domain": record.domain,
                    "query": record.query,
                    "summary": record.summary,
                    "keywords": list(record.keywords),
                    "weight": record.weight,
                    "last_step": record.last_step,
                    "accesses": record.accesses,
                    "vector": list(record.vector),
                    "updated_at": record.updated_at,
                    "is_user_provided": record.is_user_provided,
                }
            )
        return {
            "decay": self.decay,
            "learning_rate": self.learning_rate,
            "time_decay_days": self.time_decay_days,
            "step": self.step,
            "records": records,
            "sharing_rules": {
                source: sorted(targets)
                for source, targets in sorted(self.sharing_rules.items())
            },
        }

    def load_dict(self, payload: dict[str, object]) -> None:
        self.decay = float(payload.get("decay", self.decay))
        self.learning_rate = float(payload.get("learning_rate", self.learning_rate))
        self.time_decay_days = float(payload.get("time_decay_days", self.time_decay_days))
        self.step = int(payload.get("step", 0))
        self.records = {}

        raw_records = payload.get("records", [])
        if isinstance(raw_records, list):
            for item in raw_records:
                if not isinstance(item, dict):
                    continue
                key = str(item.get("key", "")).strip()
                domain = str(item.get("domain", "")).strip()
                if not key or not domain:
                    continue
                record = MemoryRecord(
                    key=key,
                    domain=domain,
                    query=str(item.get("query", "")),
                    summary=str(item.get("summary", "")),
                    keywords=tuple(str(keyword) for keyword in item.get("keywords", [])),
                    weight=float(item.get("weight", 0.5)),
                    last_step=int(item.get("last_step", 0)),
                    accesses=int(item.get("accesses", 0)),
                    vector=tuple(float(value) for value in item.get("vector", [])),
                    updated_at=float(item.get("updated_at", 0.0)),
                    is_user_provided=bool(item.get("is_user_provided", False)),
                )
                self.records[record.key] = record

        raw_rules = payload.get("sharing_rules", {})
        sharing_rules: dict[str, set[str]] = {}
        if isinstance(raw_rules, dict):
            for source, targets in raw_rules.items():
                if not isinstance(source, str):
                    continue
                if isinstance(targets, (list, tuple, set)):
                    sharing_rules[source] = {str(target) for target in targets if str(target)}
        self.sharing_rules = sharing_rules

    def resolve_shared_domains(
        self,
        target_domain: str | None,
        explicit_shares: tuple[tuple[str, str], ...] = (),
    ) -> tuple[str, ...]:
        if target_domain is None or target_domain == "memory":
            domains = sorted({record.domain for record in self.records.values()})
            return tuple(domain for domain in domains if domain)
        allowed = {target_domain}
        for source, targets in self.sharing_rules.items():
            if target_domain in targets:
                allowed.add(source)
        for source, shared_target in explicit_shares:
            if shared_target == target_domain:
                allowed.add(source)
        return tuple(sorted(allowed))

    def remember_memory_sharing(self, text: str) -> tuple[tuple[str, str], ...]:
        shares = parse_memory_sharing_preferences(text)
        for source, target in shares:
            if source == target:
                continue
            self.sharing_rules.setdefault(source, set()).add(target)
        return shares

    def probability_of_use_matrix(self) -> dict[str, float]:
        """
        Returns a 'Probability of use Matrix' mapping domains to their
        relative usage probability within the shared knowledge space.
        """
        domain_counts: dict[str, float] = {}
        total = 0.0
        for record in self.records.values():
            # Base it on history and accesses
            score = 1.0 + record.accesses + record.weight
            domain_counts[record.domain] = domain_counts.get(record.domain, 0.0) + score
            total += score
            
        if total == 0:
            return {}
            
        return {domain: round(score / total, 4) for domain, score in domain_counts.items()}

    def recall(
        self,
        query: str,
        limit: int = 3,
        target_domain: str | None = None,
        explicit_shares: tuple[tuple[str, str], ...] = (),
    ) -> list[MemoryRecord]:
        query_normalized = normalize_text(query)
        query_tokens = set(tokenize(query))
        query_vector = self.encoder.encode(query)
        allowed_domains = set(self.resolve_shared_domains(target_domain, explicit_shares))
        now = time.time()
        ranked: list[tuple[float, MemoryRecord]] = []
        for record in self.records.values():
            if target_domain is not None and target_domain != "memory" and record.domain not in allowed_domains:
                continue
            freshness = self.decay ** max(self.step - record.last_step, 0)
            if record.updated_at > 0 and self.time_decay_days > 0:
                age_days = max(now - record.updated_at, 0.0) / 86400.0
                time_freshness = math.exp(-age_days / self.time_decay_days)
                freshness *= max(0.25, time_freshness)
            keyword_overlap = overlap_score(query_tokens, record.keywords)
            score = keyword_overlap * (0.7 + 0.3 * record.weight) * freshness
            similarity = 0.0
            if record.vector:
                similarity = VectorEncoder.cosine_similarity(query_vector, record.vector)
                score += (similarity * 0.45 * freshness)
            exact_query_match = bool(record.query and normalize_text(record.query) == query_normalized)
            special_match = False
            if exact_query_match:
                score += 0.14 * freshness
                special_match = True
            if record.key.startswith("preference:") and has_drink_signal(query_tokens):
                score += 0.18 * freshness
                special_match = True
            if record.key.startswith("health:") and has_drink_signal(query_tokens):
                score += 0.22 * freshness
                special_match = True
            if record.key.startswith("kitchen:") and has_recipe_signal(query_tokens):
                score += 0.24 * freshness
                special_match = True
            if record.key.startswith("ingredient:") and has_recipe_signal(query_tokens):
                score += 0.26 * freshness
                special_match = True
            feedback_relevant = exact_query_match or keyword_overlap >= 0.08 or similarity >= 0.24
            if record.key.startswith("feedback:") and feedback_relevant:
                score += 0.28 * freshness
                if has_recipe_signal(query_tokens):
                    score += 0.08 * freshness
                special_match = True
            if record.key.startswith("profile:") and ("name" in query_tokens or "who" in query_tokens):
                score += 0.24 * freshness
                special_match = True
            eligible = special_match or keyword_overlap >= 0.05 or similarity >= 0.24
            if eligible and score >= 0.04:
                ranked.append((score, record))
        ranked.sort(key=lambda item: (-item[0], item[1].key))
        return [record for _, record in ranked[:limit]]

    def mark_access(self, records: list[MemoryRecord]) -> None:
        for record in records:
            current = self.records.get(record.key)
            if current is not None:
                current.accesses += 1

    def _blend_weight(
        self,
        record: MemoryRecord,
        new_vector: tuple[float, ...],
        reward_signal: float,
        floor: float,
    ) -> float:
        similarity = VectorEncoder.cosine_similarity(record.vector, new_vector) if record.vector else 0.0
        novelty = 1.0 - similarity
        blended = (
            (record.weight * self.decay * (0.85 + 0.15 * similarity))
            + (self.learning_rate * max(reward_signal, 0.0) * (0.7 + 0.3 * novelty))
        )
        return clamp(blended, floor, 2.0)

    def integrate(
        self,
        scenario: QueryScenario,
        answer: str,
        final_domain: str,
        reward: float,
        is_user_provided: bool = False,
    ) -> MemoryRecord:
        self.step += 1
        answer_tokens = tokenize(answer)
        summary_terms = unique_preserve(list(scenario.expected_keywords) + answer_tokens)[:6]
        key_terms = unique_preserve(list(scenario.expected_keywords) + tokenize(scenario.query))[:4]
        record_key = f"{final_domain}:{'-'.join(key_terms) or 'general'}"
        summary = f"{final_domain} note: {', '.join(summary_terms) or scenario.query}"
        new_vector = self.encoder.encode(f"{scenario.query} {answer}")
        timestamp = time.time()

        if record_key in self.records:
            record = self.records[record_key]
            record.summary = summary
            record.query = scenario.query
            record.keywords = tuple(summary_terms)
            record.weight = self._blend_weight(record, new_vector, reward, 0.05)
            record.last_step = self.step
            record.vector = new_vector
            record.updated_at = timestamp
            if is_user_provided:
                record.is_user_provided = True
            return record

        record = MemoryRecord(
            key=record_key,
            domain=final_domain,
            query=scenario.query,
            summary=summary,
            keywords=tuple(summary_terms),
            weight=clamp(0.4 + self.learning_rate * max(reward, 0.0), 0.05, 2.0),
            last_step=self.step,
            vector=new_vector,
            updated_at=timestamp,
            is_user_provided=is_user_provided,
        )
        self.records[record_key] = record
        return record

    def remember_preference(self, query: str) -> MemoryRecord | None:
        preference = extract_user_preference(query)
        if preference is None:
            return None

        self.step += 1
        item = str(preference["item"])
        keywords = tuple(preference["keywords"])
        key_terms = unique_preserve(list(keywords))[:4]
        record_key = f"preference:{'-'.join(key_terms) or 'general'}"
        rainy_note = " on rainy days" if "rainy" in keywords else ""
        summary = f"user preference: likes {item}{rainy_note}"
        new_vector = self.encoder.encode(query)
        timestamp = time.time()

        if record_key in self.records:
            record = self.records[record_key]
            record.query = query
            record.summary = summary
            record.keywords = keywords
            record.weight = self._blend_weight(record, new_vector, 0.18, 0.2)
            record.last_step = self.step
            record.vector = new_vector
            record.updated_at = timestamp
            return record

        record = MemoryRecord(
            key=record_key,
            domain="food",
            query=query,
            summary=summary,
            keywords=keywords,
            weight=0.78,
            last_step=self.step,
            vector=new_vector,
            updated_at=timestamp,
        )
        self.records[record_key] = record
        return record

    def remember_health_profile(self, text: str) -> MemoryRecord | None:
        conditions = extract_health_conditions(text)
        if not conditions:
            return None

        self.step += 1
        key_terms = unique_preserve(list(conditions) + ["health", "drink"])[:4]
        record_key = f"health:{'-'.join(key_terms) or 'general'}"
        summary = f"health note: {', '.join(conditions).replace('_', ' ')}"
        new_vector = self.encoder.encode(text)
        timestamp = time.time()

        if record_key in self.records:
            record = self.records[record_key]
            record.query = text
            record.summary = summary
            record.keywords = tuple(unique_preserve(list(conditions) + ["health", "drink", "care"]))
            record.weight = self._blend_weight(record, new_vector, 0.18, 0.2)
            record.last_step = self.step
            record.vector = new_vector
            record.updated_at = timestamp
            return record

        record = MemoryRecord(
            key=record_key,
            domain="food",
            query=text,
            summary=summary,
            keywords=tuple(unique_preserve(list(conditions) + ["health", "drink", "care"])),
            weight=0.82,
            last_step=self.step,
            vector=new_vector,
            updated_at=timestamp,
        )
        self.records[record_key] = record
        return record

    def remember_identity(self, text: str) -> MemoryRecord | None:
        identity = extract_user_identity(text)
        if identity is None:
            return None

        self.step += 1
        name = str(identity["name"])
        keywords = tuple(identity["keywords"])
        record_key = f"profile:{'-'.join(keywords[:3]) or 'name'}"
        summary = f"profile note: your name is {name}"
        new_vector = self.encoder.encode(text)
        timestamp = time.time()

        if record_key in self.records:
            record = self.records[record_key]
            record.query = text
            record.summary = summary
            record.keywords = keywords
            record.weight = self._blend_weight(record, new_vector, 0.18, 0.25)
            record.last_step = self.step
            record.vector = new_vector
            record.updated_at = timestamp
            return record

        record = MemoryRecord(
            key=record_key,
            domain="memory",
            query=text,
            summary=summary,
            keywords=keywords,
            weight=0.84,
            last_step=self.step,
            vector=new_vector,
            updated_at=timestamp,
        )
        self.records[record_key] = record
        return record

    def remember_cooking_constraints(self, text: str) -> MemoryRecord | None:
        constraints = extract_kitchen_constraints(text)
        if not constraints:
            return None

        self.step += 1
        key_terms = unique_preserve(list(constraints) + ["kitchen", "recipe"])[:4]
        record_key = f"kitchen:{'-'.join(key_terms) or 'general'}"
        readable = ", ".join(item.replace("_", " ") for item in constraints)
        summary = f"kitchen note: {readable}"
        keywords = tuple(unique_preserve(list(constraints) + ["kitchen", "recipe", "cook"]))
        new_vector = self.encoder.encode(text)
        timestamp = time.time()

        if record_key in self.records:
            record = self.records[record_key]
            record.query = text
            record.summary = summary
            record.keywords = keywords
            record.weight = self._blend_weight(record, new_vector, 0.18, 0.2)
            record.last_step = self.step
            record.vector = new_vector
            record.updated_at = timestamp
            return record

        record = MemoryRecord(
            key=record_key,
            domain="food",
            query=text,
            summary=summary,
            keywords=keywords,
            weight=0.8,
            last_step=self.step,
            vector=new_vector,
            updated_at=timestamp,
        )
        self.records[record_key] = record
        return record

    def remember_ingredient_preferences(self, text: str) -> MemoryRecord | None:
        ingredients = extract_ingredient_avoidances(text)
        if not ingredients:
            return None

        self.step += 1
        key_terms = unique_preserve(list(ingredients) + ["ingredient", "avoid"])[:4]
        record_key = f"ingredient:{'-'.join(term.replace(' ', '-') for term in key_terms) or 'general'}"
        summary = f"ingredient note: avoid {', '.join(ingredients)}"
        keyword_terms = list(ingredients)
        for ingredient in ingredients:
            keyword_terms.extend(ingredient.split())
        keywords = tuple(unique_preserve(keyword_terms + ["ingredient", "avoid", "recipe", "allergy"]))
        new_vector = self.encoder.encode(text)
        timestamp = time.time()

        if record_key in self.records:
            record = self.records[record_key]
            record.query = text
            record.summary = summary
            record.keywords = keywords
            record.weight = self._blend_weight(record, new_vector, 0.18, 0.25)
            record.last_step = self.step
            record.vector = new_vector
            record.updated_at = timestamp
            return record

        record = MemoryRecord(
            key=record_key,
            domain="food",
            query=text,
            summary=summary,
            keywords=keywords,
            weight=0.84,
            last_step=self.step,
            vector=new_vector,
            updated_at=timestamp,
        )
        self.records[record_key] = record
        return record

    def remember_feedback(self, query: str, notes: str, domain: str) -> MemoryRecord | None:
        note = " ".join(notes.strip().split())
        if not note:
            return None

        self.step += 1
        query_terms = [token for token in tokenize(query) if len(token) > 2][:4]
        note_terms = [token for token in tokenize(note) if len(token) > 2][:5]
        key_terms = unique_preserve(query_terms + note_terms)[:4]
        record_key = f"feedback:{domain}:{'-'.join(key_terms) or 'general'}"
        keywords = tuple(unique_preserve(query_terms + note_terms + [domain, "feedback"]))
        summary = f"feedback note: {note}"
        new_vector = self.encoder.encode(f"{query} {note}")
        timestamp = time.time()

        if record_key in self.records:
            record = self.records[record_key]
            record.query = query
            record.summary = summary
            record.keywords = keywords
            record.weight = self._blend_weight(record, new_vector, 0.2, 0.25)
            record.last_step = self.step
            record.vector = new_vector
            record.updated_at = timestamp
            return record

        record = MemoryRecord(
            key=record_key,
            domain=domain,
            query=query,
            summary=summary,
            keywords=keywords,
            weight=0.88,
            last_step=self.step,
            vector=new_vector,
            updated_at=timestamp,
        )
        self.records[record_key] = record
        return record
