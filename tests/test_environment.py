from __future__ import annotations

import time
import unittest

from hackathon_ai_env.environment import HackathonAIEnvironment
from hackathon_ai_env.live_knowledge import LiveKnowledge
from hackathon_ai_env.memory import SharedKnowledgeSpace
from hackathon_ai_env.models import MemoryRecord
from hackathon_ai_env.scenarios import default_scenarios


class FakeKnowledgeRetriever:
    def lookup(self, topic: str, domain: str) -> LiveKnowledge | None:
        cleaned = topic.strip().lower()
        if domain == "research" and cleaned == "reinforcement learning":
            return LiveKnowledge(
                topic=topic,
                title="Reinforcement learning",
                summary="Reinforcement learning is a machine learning method where an agent learns by taking actions and receiving rewards.",
                source_name="Wikipedia",
                url="https://example.com/reinforcement-learning",
            )
        return None

    def lookup_health(self, topic: str) -> LiveKnowledge | None:
        cleaned = topic.strip().lower()
        if cleaned == "fibromyalgia":
            return LiveKnowledge(
                topic=topic,
                title="Fibromyalgia",
                summary="Fibromyalgia is a chronic condition associated with widespread pain, fatigue, and sleep problems.",
                source_name="MedlinePlus",
                url="https://example.com/fibromyalgia",
            )
        return None


class EnvironmentTest(unittest.TestCase):
    def test_training_populates_q_table(self) -> None:
        env = HackathonAIEnvironment(seed=3)
        summaries = env.train(default_scenarios(), episodes=8)
        self.assertEqual(len(summaries), 8)
        self.assertTrue(env.q_controller.q_table)

    def test_evaluation_reaches_high_accuracy(self) -> None:
        env = HackathonAIEnvironment(seed=3)
        scenarios = default_scenarios()
        env.train(scenarios, episodes=40)
        results = env.evaluate(scenarios)
        accuracy = sum(
            1 for result in results if result.final_agent == result.scenario.expected_domain
        ) / len(results)
        self.assertGreaterEqual(accuracy, 0.8)

    def test_memory_query_recalls_prior_notes(self) -> None:
        env = HackathonAIEnvironment(seed=3)
        scenarios = default_scenarios()
        env.train(scenarios, episodes=40)
        env.prime_memory(scenarios)
        result = env.answer_query("What was the budget-friendly dinner idea we discussed earlier?")
        self.assertEqual(result.final_agent, "memory")
        self.assertTrue(result.recalled_memory_keys)

    def test_business_idea_query_returns_concrete_ideas(self) -> None:
        env = HackathonAIEnvironment(seed=3)
        scenarios = default_scenarios()
        env.train(scenarios, episodes=40)
        result = env.answer_query(
            "Suggest me some startup ideas. My conditions are: it should be less than 1 lakh budget and good for futuristic investment."
        )
        self.assertEqual(result.final_agent, "business")
        self.assertIn("Startup ideas", result.answer)
        self.assertIn("1.", result.answer)
        self.assertIn("Budget:", result.answer)

    def test_feedback_updates_policy_and_memory(self) -> None:
        env = HackathonAIEnvironment(seed=3)
        scenarios = default_scenarios()
        env.train(scenarios, episodes=40)
        before_memory_count = len(env.memory.records)
        result = env.answer_query("Suggest a startup idea under 1 lakh.")
        reward = env.apply_feedback(result, rating=5)
        self.assertGreater(reward.total, 0.7)
        self.assertGreaterEqual(len(env.memory.records), before_memory_count)
        self.assertIn(result.action.key, env.q_controller.q_table[result.state_key])

    def test_preference_statement_guides_followup_drink_query(self) -> None:
        env = HackathonAIEnvironment(seed=3)
        scenarios = default_scenarios()
        env.train(scenarios, episodes=40)
        env.answer_query("I like coffee because it is raining.")
        result = env.answer_query("Today is raining, what should I drink?")
        self.assertEqual(result.final_agent, "food")
        self.assertIn("coffee", result.answer.lower())
        self.assertTrue(result.recalled_memory_keys)

    def test_health_condition_in_query_changes_drink_suggestion(self) -> None:
        env = HackathonAIEnvironment(seed=3)
        scenarios = default_scenarios()
        env.train(scenarios, episodes=40)
        result = env.answer_query(
            "As my mood is happy, what should I drink today if it is raining and I have diabetes?"
        )
        self.assertEqual(result.final_agent, "food")
        self.assertIn("health-aware drink suggestion", result.answer.lower())
        self.assertIn("diabetes", result.answer.lower())
        self.assertIn("avoid", result.answer.lower())

    def test_typo_health_condition_with_preference_still_gives_specific_guidance(self) -> None:
        env = HackathonAIEnvironment(seed=3)
        scenarios = default_scenarios()
        env.train(scenarios, episodes=40)
        result = env.answer_query("i like coffee but i am a dibites patient ewhat to do")
        self.assertEqual(result.final_agent, "food")
        self.assertIn("diabetes", result.answer.lower())
        self.assertIn("coffee", result.answer.lower())
        self.assertIn("unsweetened", result.answer.lower())

    def test_health_condition_in_feedback_is_remembered(self) -> None:
        env = HackathonAIEnvironment(seed=3)
        scenarios = default_scenarios()
        env.train(scenarios, episodes=40)
        first = env.answer_query("What should I drink today if it is raining?")
        env.apply_feedback(first, rating=4, notes="I have hypertension and need safer drink options.")
        second = env.answer_query("It is raining again, what should I drink?")
        self.assertEqual(second.final_agent, "food")
        self.assertIn("health factors noticed", second.answer.lower())
        self.assertIn("hypertension", second.answer.lower())

    def test_name_introduction_is_not_routed_to_business(self) -> None:
        env = HackathonAIEnvironment(seed=3)
        scenarios = default_scenarios()
        env.train(scenarios, episodes=40)
        result = env.answer_query("I am Nomish Agarwal")
        self.assertEqual(result.final_agent, "memory")
        self.assertIn("your name is Nomish Agarwal", result.answer)

    def test_saved_name_can_be_recalled(self) -> None:
        env = HackathonAIEnvironment(seed=3)
        scenarios = default_scenarios()
        env.train(scenarios, episodes=40)
        env.answer_query("I am Nomish Agarwal")
        result = env.answer_query("What is my name?")
        self.assertEqual(result.final_agent, "memory")
        self.assertIn("Nomish Agarwal", result.answer)

    def test_recipe_query_returns_direct_rice_recipe(self) -> None:
        env = HackathonAIEnvironment(seed=3)
        scenarios = default_scenarios()
        env.train(scenarios, episodes=40)
        result = env.answer_query("What is the recipe of rice?")
        self.assertEqual(result.final_agent, "food")
        self.assertIn("Simple rice recipe", result.answer)
        self.assertIn("1 cup rice", result.answer)
        self.assertIn("Steps:", result.answer)

    def test_recipe_query_with_memory_still_returns_recipe(self) -> None:
        env = HackathonAIEnvironment(seed=3)
        scenarios = default_scenarios()
        env.train(scenarios, episodes=40)
        env.prime_memory(scenarios)
        result = env.answer_query("What is the recipe of rice?")
        self.assertEqual(result.final_agent, "food")
        self.assertIn("Simple rice recipe", result.answer)
        self.assertNotIn("Food plan with memory", result.answer)

    def test_pakoda_recipe_prompt_does_not_treat_give_as_subject(self) -> None:
        env = HackathonAIEnvironment(seed=3)
        scenarios = default_scenarios()
        env.train(scenarios, episodes=40)
        result = env.answer_query("give pakoda recipe")
        self.assertEqual(result.final_agent, "food")
        self.assertIn("Simple pakoda recipe", result.answer)
        self.assertNotIn("Simple give recipe", result.answer)

    def test_pakoda_recipe_prompt_with_extra_words_returns_pakoda_recipe(self) -> None:
        env = HackathonAIEnvironment(seed=3)
        scenarios = default_scenarios()
        env.train(scenarios, episodes=40)
        result = env.answer_query("give me a pakoda recipe")
        self.assertEqual(result.final_agent, "food")
        self.assertIn("Simple pakoda recipe", result.answer)
        self.assertIn("besan", result.answer)

    def test_sandwich_recipe_prompt_returns_concrete_recipe(self) -> None:
        env = HackathonAIEnvironment(seed=3)
        scenarios = default_scenarios()
        env.train(scenarios, episodes=40)
        result = env.answer_query("What is the recipe of sandwich?")
        self.assertEqual(result.final_agent, "food")
        self.assertIn("Simple sandwich recipe", result.answer)
        self.assertIn("4 bread or wrap pieces", result.answer)
        self.assertIn("toast it", result.answer)

    def test_noodles_recipe_prompt_uses_general_recipe_engine(self) -> None:
        env = HackathonAIEnvironment(seed=3)
        scenarios = default_scenarios()
        env.train(scenarios, episodes=40)
        result = env.answer_query("How to make noodles?")
        self.assertEqual(result.final_agent, "food")
        self.assertIn("Simple noodles recipe", result.answer)
        self.assertIn("Ingredients:", result.answer)
        self.assertIn("Steps:", result.answer)

    def test_multiword_recipe_subject_is_preserved(self) -> None:
        env = HackathonAIEnvironment(seed=3)
        scenarios = default_scenarios()
        env.train(scenarios, episodes=40)
        result = env.answer_query("How to make paneer sandwich?")
        self.assertEqual(result.final_agent, "food")
        self.assertIn("Simple paneer sandwich recipe", result.answer)
        self.assertIn("paneer", result.answer)

    def test_golgappa_recipe_prompt_returns_specific_recipe(self) -> None:
        env = HackathonAIEnvironment(seed=3)
        scenarios = default_scenarios()
        env.train(scenarios, episodes=40)
        result = env.answer_query("What is the recipe of golgappa?")
        self.assertEqual(result.final_agent, "food")
        self.assertIn("Simple golgappa recipe", result.answer)
        self.assertIn("semolina", result.answer.lower())
        self.assertIn("fry", result.answer.lower())

    def test_unknown_multiword_recipe_subject_is_preserved(self) -> None:
        env = HackathonAIEnvironment(seed=3)
        scenarios = default_scenarios()
        env.train(scenarios, episodes=40)
        result = env.answer_query("How do I make tomato soup?")
        self.assertEqual(result.final_agent, "food")
        self.assertIn("Simple tomato soup recipe", result.answer)
        self.assertIn("Ingredients:", result.answer)
        self.assertIn("Steps:", result.answer)

    def test_unknown_health_condition_can_use_live_health_context(self) -> None:
        env = HackathonAIEnvironment(seed=3, knowledge_retriever=FakeKnowledgeRetriever())
        scenarios = default_scenarios()
        env.train(scenarios, episodes=40)
        result = env.answer_query("I have fibromyalgia, what should I drink?")
        self.assertEqual(result.final_agent, "food")
        self.assertIn("Live health context from MedlinePlus", result.answer)
        self.assertIn("Fibromyalgia", result.answer)
        self.assertIn("Source: https://example.com/fibromyalgia", result.answer)

    def test_general_research_concept_can_use_live_context(self) -> None:
        env = HackathonAIEnvironment(seed=3, knowledge_retriever=FakeKnowledgeRetriever())
        scenarios = default_scenarios()
        env.train(scenarios, episodes=40)
        result = env.answer_query("What is reinforcement learning?")
        self.assertEqual(result.final_agent, "research")
        self.assertIn("Live research context from Wikipedia", result.answer)
        self.assertIn("taking actions and receiving rewards", result.answer)

    def test_live_context_query_reuses_feedback_on_repeat(self) -> None:
        env = HackathonAIEnvironment(seed=3, knowledge_retriever=FakeKnowledgeRetriever())
        scenarios = default_scenarios()
        env.train(scenarios, episodes=40)
        first = env.answer_query("What is reinforcement learning?")
        env.apply_feedback(first, rating=4, notes="Keep the explanation simpler next time.")
        second = env.answer_query("What is reinforcement learning?")
        self.assertEqual(second.final_agent, "research")
        self.assertIn("Live research context from Wikipedia", second.answer)
        self.assertIn("earlier feedback", second.answer.lower())
        self.assertIn("Keep the explanation simpler next time.", second.answer)

    def test_business_feedback_does_not_leak_into_recipe_answer(self) -> None:
        env = HackathonAIEnvironment(seed=3)
        scenarios = default_scenarios()
        env.train(scenarios, episodes=40)
        first = env.answer_query("Suggest a startup idea under 1 lakh.")
        env.apply_feedback(first, rating=4, notes="Use dollars next time.")
        second = env.answer_query("What is the recipe of rice?")
        self.assertEqual(second.final_agent, "food")
        self.assertIn("Simple rice recipe", second.answer)
        self.assertNotIn("Use dollars next time.", second.answer)
        self.assertNotIn("earlier feedback", second.answer.lower())

    def test_idli_sambhar_recipe_prompt_returns_specific_recipe(self) -> None:
        env = HackathonAIEnvironment(seed=3)
        scenarios = default_scenarios()
        env.train(scenarios, episodes=40)
        result = env.answer_query("tell me recipe of idlisambhar")
        self.assertEqual(result.final_agent, "food")
        self.assertIn("Simple idli sambhar recipe", result.answer)
        self.assertIn("Ingredients for idli", result.answer)
        self.assertIn("Ingredients for sambhar", result.answer)

    def test_feedback_about_missing_pan_changes_future_recipe_answer(self) -> None:
        env = HackathonAIEnvironment(seed=3)
        scenarios = default_scenarios()
        env.train(scenarios, episodes=40)
        first = env.answer_query("tell me recipe of idlisambhar")
        env.apply_feedback(first, rating=3, notes="I don't have a pan.")
        second = env.answer_query("tell me recipe of idlisambhar")
        self.assertEqual(second.final_agent, "food")
        self.assertIn("Simple idli sambhar recipe", second.answer)
        self.assertIn("Constraint note:", second.answer)
        self.assertIn("do not have a pan", second.answer)

    def test_onion_allergy_feedback_changes_future_recipe_answer(self) -> None:
        env = HackathonAIEnvironment(seed=3)
        scenarios = default_scenarios()
        env.train(scenarios, episodes=40)
        first = env.answer_query("tell me recipe of idlisambhar")
        env.apply_feedback(first, rating=3, notes="I am allergic to onion so suggest me without onion recipe.")
        second = env.answer_query("tell me recipe of idlisambhar")
        self.assertEqual(second.final_agent, "food")
        self.assertIn("Simple idli sambhar recipe", second.answer)
        self.assertNotIn("1 chopped onion", second.answer)
        self.assertNotIn("Add onion", second.answer)
        self.assertIn("avoids onion", second.answer)

    def test_direct_without_onion_recipe_query_is_adjusted(self) -> None:
        env = HackathonAIEnvironment(seed=3)
        scenarios = default_scenarios()
        env.train(scenarios, episodes=40)
        result = env.answer_query("tell me recipe of idlisambhar without onion")
        self.assertEqual(result.final_agent, "food")
        self.assertIn("Simple idli sambhar recipe", result.answer)
        self.assertNotIn("1 chopped onion", result.answer)
        self.assertNotIn("Add onion", result.answer)
        self.assertIn("avoids onion", result.answer)

    def test_generic_recipe_feedback_is_reused_on_repeat(self) -> None:
        env = HackathonAIEnvironment(seed=3)
        scenarios = default_scenarios()
        env.train(scenarios, episodes=40)
        first = env.answer_query("What is the recipe of sandwich?")
        env.apply_feedback(first, rating=4, notes="Keep it simpler next time.")
        second = env.answer_query("What is the recipe of sandwich?")
        self.assertEqual(second.final_agent, "food")
        self.assertIn("Simple sandwich recipe", second.answer)
        self.assertIn("earlier feedback", second.answer.lower())
        self.assertIn("Keep it simpler next time.", second.answer)

    def test_chaat_masala_feedback_changes_sandwich_recipe_steps(self) -> None:
        env = HackathonAIEnvironment(seed=3)
        scenarios = default_scenarios()
        env.train(scenarios, episodes=40)
        first = env.answer_query("What is the recipe of sandwich?")
        env.apply_feedback(first, rating=4, notes="Please avoid chaat masala in the sandwich recipe.")
        second = env.answer_query("What is the recipe of sandwich?")
        self.assertEqual(second.final_agent, "food")
        self.assertIn("Simple sandwich recipe", second.answer)
        self.assertNotIn("salt, pepper, and a little chaat masala", second.answer.lower())
        self.assertNotIn("sprinkle a little salt, pepper, and chaat masala", second.answer.lower())
        self.assertIn("mixed herbs", second.answer.lower())

    def test_recipe_feedback_does_not_distort_later_drink_query(self) -> None:
        env = HackathonAIEnvironment(seed=3)
        scenarios = default_scenarios()
        env.train(scenarios, episodes=40)
        first = env.answer_query("What is the recipe of sandwich?")
        env.apply_feedback(first, rating=4, notes="Please avoid chaat masala in the sandwich recipe.")
        second = env.answer_query("What should I drink today if it is raining?")
        self.assertEqual(second.final_agent, "food")
        self.assertIn("Food suggestion", second.answer)
        self.assertNotIn("chaat masala", second.answer.lower())
        self.assertNotIn("earlier feedback", second.answer.lower())

    def test_golgappa_feedback_upgrades_generic_recipe_to_specific_method(self) -> None:
        env = HackathonAIEnvironment(seed=3)
        scenarios = default_scenarios()
        env.train(scenarios, episodes=40)
        first = env.answer_query("What is the recipe of golgappa?")
        env.apply_feedback(
            first,
            rating=4,
            notes="kneading 1 cup semolina with 1/4 cup warm oil and roughly 80-100ml warm water, resting the dough for 20 minutes, then frying at medium-high heat.",
        )
        second = env.answer_query("What is the recipe of golgappa?")
        self.assertEqual(second.final_agent, "food")
        self.assertIn("Simple golgappa recipe", second.answer)
        self.assertIn("1 cup semolina", second.answer)
        self.assertIn("1/4 cup warm oil", second.answer)
        self.assertIn("80-100ml warm water", second.answer)
        self.assertIn("Rest the dough for 20 minutes", second.answer)
        self.assertIn("fry at medium-high heat", second.answer.lower())
        self.assertNotIn("1 to 2 cups golgappa", second.answer)
        self.assertNotIn("I also adjusted this using your earlier feedback", second.answer)

    def test_exact_feedback_upgrades_unknown_recipe_query(self) -> None:
        env = HackathonAIEnvironment(seed=3)
        scenarios = default_scenarios()
        env.train(scenarios, episodes=40)
        first = env.answer_query("How do I make tomato soup?")
        env.apply_feedback(
            first,
            rating=4,
            notes="boil 3 tomatoes with 2 cups water, blend until smooth, simmer with salt and pepper for 5 minutes.",
        )
        second = env.answer_query("How do I make tomato soup?")
        self.assertEqual(second.final_agent, "food")
        self.assertIn("Simple tomato soup recipe", second.answer)
        self.assertIn("3 tomatoes", second.answer)
        self.assertIn("2 cups water", second.answer)
        self.assertIn("Blend until smooth", second.answer)
        self.assertIn("simmer with salt and pepper", second.answer.lower())
        self.assertNotIn("I also adjusted this using your earlier feedback", second.answer)

    def test_exact_feedback_upgrades_multiword_recipe_query(self) -> None:
        env = HackathonAIEnvironment(seed=3)
        scenarios = default_scenarios()
        env.train(scenarios, episodes=40)
        first = env.answer_query("How to make masala dosa?")
        env.apply_feedback(
            first,
            rating=4,
            notes="mix 1 cup dosa batter with a little salt, spread thin on a hot tawa, drizzle oil, and cook until crisp.",
        )
        second = env.answer_query("How to make masala dosa?")
        self.assertEqual(second.final_agent, "food")
        self.assertIn("Simple masala dosa recipe", second.answer)
        self.assertIn("Ingredients: 1 cup dosa batter.", second.answer)
        self.assertIn("Spread thin on a hot tawa", second.answer)
        self.assertIn("Drizzle oil", second.answer)
        self.assertIn("Cook until crisp", second.answer)
        self.assertNotIn("I also adjusted this using your earlier feedback", second.answer)

    def test_memory_banks_are_separated_without_explicit_sharing(self) -> None:
        env = HackathonAIEnvironment(seed=3)
        scenarios = default_scenarios()
        env.train(scenarios, episodes=40)
        env.answer_query("I like coffee because it is raining.")
        result = env.answer_query("Suggest pricing for my startup.")
        self.assertEqual(result.final_agent, "business")
        business_bank = result.trace["memory_banks_used"]["business"]
        self.assertFalse(any(key.startswith("preference:") for key in business_bank))

    def test_cross_router_sharing_requires_user_permission(self) -> None:
        env = HackathonAIEnvironment(seed=3)
        scenarios = default_scenarios()
        env.train(scenarios, episodes=40)
        env.answer_query("I like coffee because it is raining.")
        result = env.answer_query("Use food knowledge in business and suggest pricing for my startup.")
        self.assertEqual(result.final_agent, "business")
        business_bank = result.trace["memory_banks_used"]["business"]
        self.assertTrue(any(key.startswith("preference:") for key in business_bank))
        self.assertIn("food->business", result.trace["shared_domains"])

    def test_time_decay_prefers_recent_memory_within_same_bank(self) -> None:
        space = SharedKnowledgeSpace()
        now = time.time()
        vector = space.encoder.encode("coffee drink")
        old_record = MemoryRecord(
            key="food:old",
            domain="food",
            query="coffee drink",
            summary="food note: old coffee drink",
            keywords=("coffee", "drink"),
            weight=0.9,
            last_step=1,
            vector=vector,
            updated_at=now - (45 * 86400),
        )
        new_record = MemoryRecord(
            key="food:new",
            domain="food",
            query="coffee drink",
            summary="food note: new coffee drink",
            keywords=("coffee", "drink"),
            weight=0.9,
            last_step=1,
            vector=vector,
            updated_at=now,
        )
        space.records = {old_record.key: old_record, new_record.key: new_record}
        recalled = space.recall("coffee drink", target_domain="food", limit=2)
        self.assertEqual(recalled[0].key, "food:new")

    def test_reward_includes_task_difficulty_and_trace_logging(self) -> None:
        env = HackathonAIEnvironment(seed=3)
        scenarios = default_scenarios()
        env.train(scenarios, episodes=8)
        result = env.evaluate([scenarios[2]])[0]
        self.assertEqual(result.reward.task_difficulty, "hard")
        self.assertGreater(result.reward.difficulty_bonus, 0.0)
        self.assertIn("query_vector", result.trace)
        self.assertIn("memory_banks_used", result.trace)
        self.assertTrue(result.explanation)


if __name__ == "__main__":
    unittest.main()
