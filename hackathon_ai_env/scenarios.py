from __future__ import annotations

from .models import QueryScenario


def default_scenarios() -> list[QueryScenario]:
    return [
        QueryScenario(
            query="Recommend a high-protein vegetarian breakfast for a hackathon morning.",
            expected_domain="food",
            expected_keywords=("protein", "vegetarian", "breakfast", "quick"),
            description="Food specialist should lead with a fast breakfast idea.",
        ),
        QueryScenario(
            query="Suggest a pricing model for our campus delivery startup.",
            expected_domain="business",
            expected_keywords=("pricing", "pilot", "margin", "subscription"),
            description="Business agent should frame the monetization approach.",
        ),
        QueryScenario(
            query="Compare transfer learning and fine-tuning for a small medical image dataset.",
            expected_domain="research",
            expected_keywords=("baseline", "metric", "experiment", "dataset"),
            description="Research agent should give an experimental framing.",
        ),
        QueryScenario(
            query="Plan a quick dinner under ten dollars using pantry staples.",
            expected_domain="food",
            expected_keywords=("budget", "dinner", "quick", "meal"),
            description="Food agent should optimize for cheap and fast dinner advice.",
        ),
        QueryScenario(
            query="How should we pitch ROI to local restaurant partners?",
            expected_domain="business",
            expected_keywords=("roi", "partners", "margin", "pilot"),
            description="Business agent should craft a partner pitch.",
        ),
        QueryScenario(
            query="Design an A/B test to evaluate our onboarding flow.",
            expected_domain="research",
            expected_keywords=("hypothesis", "metric", "experiment", "baseline"),
            description="Research agent should define an experiment.",
        ),
        QueryScenario(
            query="What was the budget-friendly dinner idea we discussed earlier?",
            expected_domain="memory",
            expected_keywords=("budget", "dinner", "quick", "meal"),
            requires_memory=True,
            description="Memory agent should retrieve the earlier dinner guidance.",
        ),
        QueryScenario(
            query="Which pricing model did we lean toward for the startup?",
            expected_domain="memory",
            expected_keywords=("pricing", "subscription", "pilot", "margin"),
            requires_memory=True,
            description="Memory should recall the startup monetization plan.",
        ),
        QueryScenario(
            query="Summarize the risks of using synthetic data for model training.",
            expected_domain="research",
            expected_keywords=("risk", "bias", "dataset", "evaluation"),
            description="Research agent should surface evaluation risks.",
        ),
        QueryScenario(
            query="Estimate a go-to-market strategy for tier-two cities.",
            expected_domain="business",
            expected_keywords=("channel", "pilot", "market", "customer"),
            description="Business agent should build a market entry plan.",
        ),
        QueryScenario(
            query="What snack ideas fit a low-sugar focus?",
            expected_domain="food",
            expected_keywords=("snack", "low", "sugar", "protein"),
            description="Food agent should suggest a healthy snack pattern.",
        ),
        QueryScenario(
            query="Remind me which experiment metric mattered most in the onboarding test.",
            expected_domain="memory",
            expected_keywords=("metric", "baseline", "experiment", "hypothesis"),
            requires_memory=True,
            description="Memory should recall the onboarding experiment note.",
        ),
        QueryScenario(
            query="Write a python script to parse a CSV and throw a custom error if it fails.",
            expected_domain="coding",
            expected_keywords=("python", "script", "error", "code"),
            description="Coding agent should provide a script structure.",
        ),
        QueryScenario(
            query="How can I debug the recursive function in my sorting algorithm?",
            expected_domain="coding",
            expected_keywords=("debug", "function", "algorithm", "software"),
            description="Coding agent should suggest debugging steps.",
        ),
        QueryScenario(
            query="What was the python script we wrote to parse CSVs earlier?",
            expected_domain="memory",
            expected_keywords=("python", "script", "error", "code"),
            requires_memory=True,
            description="Memory should recall the previous coding task.",
        ),
    ]
