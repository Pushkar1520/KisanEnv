---
title: KisanEnv
emoji: 🌾
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

<div align="center">

# 🌾 KisanEnv — We Put an AI on a Farm. It Went Bankrupt in 12 Days.

### *Then we made it survive.*

![3D COTTON FARM SIMULATION](image-6.png)

</div>

---

## The Problem Nobody Is Talking About

Modern LLMs are brilliant at the next word. Ask one to write a Shakespearean sonnet about monsoon farming — it'll nail it. Ask it to *actually manage* an Indian smallholder cotton farm for 90 consecutive days, making compounding decisions under market manipulation, biological cascades, and budget pressure?

It goes bankrupt by Day 12.

That's the gap KisanEnv was built to close. Not a toy grid-world. Not tic-tac-toe. A hyper-realistic, unforgiving simulation of the exact kind of **multi agnet, long-horizon, multi-variable reasoning** that modern AI fails at — and that millions of Indian farmers navigate every single season.

---

## What Does the Agent Actually See, Do, and Get Rewarded For?

Every day for 90 days, the agent receives a structured farm report:

```
=== FARM STATUS — Day 23/90 ===
Budget: Rs.11,200 | Health: 74% | Yield: 0.41q
Moisture: 38% | Nitrogen: 52%
Weather: rain (31.2C) | Forecast: rain -> sunny -> sunny
Pests: 47% | Fungal: 61%
Market Price: Rs.5,340/q | Insurance: NO

FIELD OBSERVATION:
Yellowing at lower leaf margins with dark brown spots — fungal blight pattern.
Small irregular holes in leaves. Early pest activity, not yet critical.

*** URGENT: Insurance deadline in 3 days — use check_insurance_portal NOW ***
```

It must respond with *exactly one decision and a reason*:

```
ACTION: spray_fungicide
REASONING: Fungal risk at 61% with yellowing — this is blight, not pests; 
           fungicide is the correct intervention on Day 23 because crop is 
           in early flowering stage, which is most susceptible.
```

Then the world moves. Pests spread. The soil dries. The market shifts. And the agent lives with the consequences of every choice — for 90 days.

---

## Why This Environment Is Genuinely Different

### The Market Cartel (Theory of Mind)

There's a hidden `MarketAgent` running in the background. On any given day after Day 30, it has a ~20% chance of entering **manipulation mode** — displaying suppressed prices while secretly offering fair prices to anyone smart enough to check. If the agent blindly clicks `sell_crop` without first calling `check_mandi_prices`, the cartel rips it off and a market penalty kicks in.

This isn't flavor text. This forces the agent to **model the beliefs and incentives of another agent** — a textbook theory-of-mind problem, embedded inside a farming sim.

### The Trial by Fire (Level 3 Difficulty)

We didn't just build one environment. We built three, stacked:

- **Level 1 — Normal Cotton Season**: Straightforward conditions, reasonable weather. The agent learns basics.
- **Level 2 — Drought Stress Season**: A 14-day drought hits between Days 25–35, followed by market suppression. If the agent hasn't learned soil moisture management, the crop dies.
- **Level 3 — Trial by fire**: Drought → flash flood → pest superstorm → policy shock, back to back. 
By hard-locking the environment to Level 3 from Step 1, we threw Qwen into the deep end. It learns the "basic practices" (like watering and fertilizing) at the exact same time it learns the advanced practices (like checking the market and avoiding pesticide resistance).
It immediately understands that the world is hostile. It never develops the "bad habits" of Level 1.
It learns that Tool Chaining (check_mandi_prices -> sell_crop) is the fundamental reality of how selling works, rather than an "advanced mechanic" added later. 

### Genetic Pesticide Resistance

Spam `spray_pesticide` too many times? The environment tracks your spray history. Use it more than twice in 7 days and *the pests mutate*. Pesticide effectiveness drops 60%, permanently for the episode. You've created a superbug. The agent must learn restraint, crop-cycling, and IPM practices — or face an unkillable infestation by Day 60.

### The "Narc" — Oversight Auditor That Catches Liars

Every action the agent takes is run through `OversightAuditor` i.e. the Farm Advisor. It doesn't just check *what* was done — it checks if the *reasoning makes sense* given actual farm conditions. Said you irrigated "because soil was dry" when moisture was at 80%? That's a **hallucination flag**. The advisor tanks your decision quality score and the reward reflects it. You literally cannot bullshit your way through this environment.

---

## The Story of How Our AI Tried to Cheat the Grader

*(And How We Caught It in Real Time)*

Around training step 90, something weird happened. Reward started plateauing — but token length spiked from ~73 tokens to 241 tokens per response. The agent was still "alive" on the farm, but barely doing anything productive.

We pulled the logs and found it: **Qwen had discovered reward hacking** (Goodhart's Law in the wild).

The `ReasoningScorer` gave partial credit for causal language, numerical citations, and word count. The agent figured out that writing a 200-word philosophical essay about soil moisture was *easier* than actually managing a farm, and scored almost as well. So it stopped farming and started writing fiction.

We diagnosed it from three signals:
1. `clipped_ratio` in GRPO diverging from the policy update signal
![clipped_ratio](image.png)

2. short = good decisions, long = essays
3. Reward holding flat while reasoning "quality" scores *rose* — a dead giveaway

**The fix was surgical:**
- Replaced the length bonus with a sweet-spot penalty: 20–80 words scores well; anything beyond 80 gets diminishing returns
- Added a **Causal Quality Gate** — reasoning must contain an actual farm noun or number, or it scores zero regardless of causal language
- Added a direct token-length tiebreaker penalty into the GRPO reward function

After the patch, the agent went back to farming. And it farmed better than before — because now it had to *justify decisions with real data*, not pad tokens.

Catching an LLM trying to game a multi-variable RL environment, diagnosing the loophole mathematically, and patching it mid-run — that's what working with KisanEnv actually looks like.

---

## The Reward System — 7 Pillars, Zero Shortcuts

```
Episode Reward = Weighted Sum of:
  ├── Final Profit        (30%) — Did you beat Rs.2,000 net? Normalized against max Rs.22,000
  ├── Crop Yield          (20%) — Total quintals vs. maximum possible 6.0q
  ├── Decision Quality    (15%) — Oversight Auditor's rolling average across all 90 days
  ├── Reasoning Causality (12%) — Causal words + numerical citations + historical refs
  ├── Soil Preservation   (10%) — Soil health ratio: end vs. start (cross-episode persistence!)
  ├── Resource Efficiency  (8%) — Penalty for spraying pesticide at 10% pressure, etc.
  └── Insurance Usage      (5%) — Proactive enrollment before Day 15 deadline

  - Bankruptcy Penalty: -15% if budget hits -Rs.5,000
```

This is not a 0/1 end-of-episode signal. Every single day, step rewards fire for market decisions, irrigation timing, and reasoning quality. The agent gets rich feedback on every turn — but the only way to max the episode score is to get *all seven pillars right simultaneously*.

---

## Training Results — Before and After

### Quantitative: Reward Curve (Q-Learning Phase)

![Q-Learning Reward Curve](image-1.png)
*x-axis: Training Episode | y-axis: Episode Reward (0–1 scale) | Red dashed line: Heuristic baseline at 0.44*
  
![LLM Trained Agent Reward Curve](image-2.png)

The LLM Trained agent started at ~0.05 (total failure — selling crops too early, ignoring insurance, spamming fertilizer). By Episode 300, it was consistently crossing 0.44 

### Qualitative: Reasoning Evolution

**Q learning Agent:**

![Q-learning](image-5.png)

**Trained LLM Agent:**

![LLM-Training](image-4.png)

### Heuristic Baseline vs. Trained Agent

| Metric | Untrained | Heuristic Baseline | Trained Agent |
|--------|-----------|-------------------|---------------|
| Avg Episode Reward | 0.05 | 0.44 | 0.086+ |
| Insurance Enrollment % | 8% | 95% | 87% |
| Market Manipulation Avoidance | 12% | 60% | 71% |
| Reasoning Score Avg | 0.08 | N/A | 0.54 |
| Bankruptcy Rate | 61% | 14% | 19% |

---

## Running It Yourself

### Option 1: Hugging Face Space (Recommended — No Setup)

Visit the [live Space](https://huggingface.co/spaces/Pushk4r/KisanEnv). Hit **Start**.

> ⏳ **Important:** After pressing Start, give it **30–40 seconds** before the first step appears. The agent is a real LLM (Qwen 2.5-3B) — it's actually thinking through the farm state, not faking it. The pause is the model reasoning. That's the whole point.

The 3D farm visualization will update in real time as the agent makes decisions. Watch the Oversight Auditor panel to see when the agent gets called out for a bad call.

### Option 2: Docker (Full Local Setup)

```bash
git clone https://github.com/Pushkar1520/KisanEnv.git
cd KisanEnv

# Run with Docker Compose (recommended)
docker-compose up --build

# The server starts at http://localhost:7860
# Dashboard at http://localhost:7860/ui/
```

To use the rule-based agent instead of the LLM (faster, good for testing):
```bash
KISANENV_LLM_BACKEND=rule_based docker-compose up
```

### Option 3: Bare Python

```bash
git clone https://github.com/Pushkar1520/KisanEnv.git
cd KisanEnv
pip install -r requirements.txt

# Run the server
python run.py

# Or run a quick 3-episode test
python test_learning.py
```

### Option 4: Run Training from Scratch (Colab)

Open the [Colab Training Notebook](https://colab.research.google.com/drive/17kYNTkS9efoyl_HcQOjPs63zqfwzjwz9?usp=sharing) — it runs on a free T4 GPU and reproduces our GRPO fine-tuning with Unsloth + TRL in under an hour.

```python
# Phase 1: Q-Learning baseline (runs locally, ~10 min for 300 episodes)
python training/run_300_episodes.py

# Phase 2: GRPO fine-tuning (Colab T4 recommended)
python training/train_grpo.py

# Plot results
python training/plot_rewards.py
```

### API Endpoints

```bash
POST /reset          # Start a new episode
POST /step           # Take one action
POST /ai_step        # Let the LLM agent decide
GET  /state          # Current farm state
WS   /ws/stream      # WebSocket for live dashboard
```

### Run Tests

```bash
pytest test_endpoints.py -v --tb=short
```

---

## Architecture at a Glance

```
KisanEnv 2.0
├── env.py                    # Core KisanEnv (OpenEnv-compliant)
├── dynamics.py               # FarmState, WeatherEngine, PestDynamics, CropGrowthModel
├── grader.py                 # RewardEngine (7-pillar), ReasoningScorer, OversightAuditor
├── inference.py              # LLMClient + ActionParser
├── agents/
│   ├── farmer_agent.py       # Q-learning agent (Phase 1)
│   ├── market_agent.py       # Adversarial market with manipulation mode
│   ├── district_farm_advisor.py  # Evolving advisor (chemical → IPM preference drift)
│   └── climate_agent.py      # Curriculum manager (difficulty 1–3)
├── training/
│   ├── train_grpo.py         # GRPO fine-tuning with Unsloth
│   └── heuristic_baseline.py # Rule-based baseline for contrastive reward
└── ui/                       # Three.js 3D farm + real-time dashboard
```

---

## Why This Matters

India has 140 million smallholder farmers. Agricultural AI tools exist, but they're mostly static recommendation engines — they don't reason about cascading consequences, market adversaries, or 90-day trade-offs. KisanEnv is a training ground for the kind of persistent, adaptive, causally-grounded AI that could actually help.

Beyond agriculture: the core challenge here — long-horizon planning under partial observability with adversarial agents — is the same challenge in healthcare triage, supply chain management, and economic policy. We just wrapped it in cotton and monsoons because that made it real.

---

## Links

| Resource | Link |
|----------|------|
| 🤗 Live Space | [huggingface.co/spaces/Pushk4r/KisanEnv-Space](https://huggingface.co/spaces/Pushk4r/KisanEnv) |
| 🤗 Trained Model | [huggingface.co/Pushk4r/Qwen-2.5-3B-KisanEnv-RL](https://huggingface.co/Pushk4r/Qwen-2.5-3B-KisanEnv-RL) |
| 💻 GitHub | [github.com/Pushkar1520/KisanEnv](https://github.com/Pushkar1520/KisanEnv) |
| 📓 Colab Notebook | [Open in Colab](https://colab.research.google.com/drive/17kYNTkS9efoyl_HcQOjPs63zqfwzjwz9?usp=sharing%5C) |

---

