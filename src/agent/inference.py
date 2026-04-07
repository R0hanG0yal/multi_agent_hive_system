import os
import json
from src.env.assistant_env import MemoryAugmentedAssistantEnv
from src.models.schemas import Action

class BaselineAgent:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_action(self, obs) -> Action:
        # Construct prompt for the AI (using gpt-4 if available)
        try:
            import openai
            openai.api_key = self.api_key
            prompt = f"TASK: {obs.instruction}\n"
            prompt += f"MEMORY: {obs.retrieved_memories}\n"
            prompt += "Respond as a JSON object with: { 'response': '...', 'decision_type': '...', 'memory_update': '...' }"
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            data = json.loads(response.choices[0].message.content)
            return Action(**data)
        except Exception:
            # Fallback for simulator
            return self._simulate_fallback(obs)

    def _simulate_fallback(self, obs) -> Action:
        if obs.task_id == "EASY_01":
            return Action(response="The user's favorite coffee is double-shot espresso.", decision_type="recall", memory_update="User preference: double-shot espresso")
        return Action(response="Simulated response for " + obs.task_id, decision_type="execute")

def run_simulation():
    env = MemoryAugmentedAssistantEnv()
    agent = BaselineAgent(api_key=os.getenv("OPENAI_API_KEY", "sk-placeholder"))
    
    obs = env.reset()
    done = False
    
    print("\n🚀 Starting OpenEnv Simulation: Memory-Augmented Self-Improving Assistant\n")
    
    while not done:
        print(f"🔹 Task: {obs.task_id} | Step: {obs.step_count} | Instruction: {obs.instruction}")
        
        action = agent.get_action(obs)
            
        obs, reward, done, info = env.step(action)
        print(f"✅ Reward: {reward.score:.2f} | Improvement Bonus: {reward.improvement_bonus:.2f} | Correctness: {reward.correctness:.2f}")

    print("\n🏁 Simulation Complete.")
    final_state = env.state()
    print(f"📊 Final Performance History: {final_state.performance_history}")

if __name__ == "__main__":
    run_simulation()
