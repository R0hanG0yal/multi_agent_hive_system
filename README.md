# Memory-Augmented Self-Improving Assistant (OpenEnv)

## 🎯 Project Overview
This is a production-grade **OpenEnv** environment designed for AI assistants that must interact with users, utilize memory, and improve their performance over time through feedback loops.

### **Real-World Motivation**
In production AI settings, assistants often lose context between sessions or repeat mistakes. This environment simulates a system where memory retrieval is key to high-reward outcomes, and the assistant is incentivized to improve through structured improvement bonuses.

---

## 🧱 Architecture
- **Environment**: Follows OpenEnv spec (step, eset, state).
- **Memory Module**: Implements both short-term context and long-term storage with retrieval.
- **Self-Improvement**: Tracking performance history to calculate incremental reward bonuses.
- **Grader**: Deterministic scoring based on ground truth and memory utility.

---

## 🧬 Observation & Action Space

### **Observation**
- 	ask_id: Identifier for current goal.
- instruction: User-provided task text.
- etrieved_memories: Relevant context fetched from long-term memory.
- history: Recent interaction steps.

### **Action**
- esponse: Text output to the user.
- decision_type: Classification of the action.
- memory_update: (Optional) New knowledge to persist.

---

## 🧠 Tasks (Difficulty: Easy → Hard)
1. **EASY_01**: Basic user preference recall.
2. **MEDIUM_01**: Date-based reasoning using stored deadlines.
3. **HARD_01**: Complex workflow planning with historical dependency.

---

## 🏆 Reward Design
The environment uses a **dense reward function**:
- **70% Correctness**: Measured against ground truth.
- **30% Memory Utility**: Reward for using retrieved context correctly.
- **Improvement Bonus**: Extra incentive if current performance exceeds historical average.
- **Mistake Penalty**: Penalizes low scores on tasks already attempted.

---

## �� Setup & Execution

### **Run Locally**
1. Install dependencies:
   `ash
   pip install -r requirements.txt
   `
2. Run baseline simulation:
   `ash
   python src/agent/inference.py
   `

### **Docker Deployment**
`ash
docker build -t openenv-assistant .
docker run -e OPENAI_API_KEY=your_key openenv-assistant
`

---

## 📦 OpenEnv Compliance
- openenv.yaml included for registration.
- Compatible with openenv validate.
- Pydantic models used for all interfaces.

Developed by Antigravity AI - Advanced Agentic Coding.
