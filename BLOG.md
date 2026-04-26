# Day 12. The AI Was Broke. The Crop Was Dead. We Were Delighted.

*Because that's exactly what was supposed to happen.*

---

We handed a state-of-the-art language model Rs.15,000, a cotton field, and 90 days to turn a profit. By Day 12 it had sprayed pesticide on a fungal infection, watered soil that was already flooded, and sold its entire crop at a manipulated price to a market agent that was quietly laughing at it.

Broke. Crop dead. Episode over.

That failure wasn't a bug. It was the whole point.

---

## The Farm That Fights Back

**KisanEnv** is a 90-day Indian cotton farming simulation where nothing is forgiving and everything is connected. Droughts don't just dry the soil — they trigger biological immune stress, which five days later becomes a pest infestation the agent never saw coming. Spray pesticide too many times in a week and the pests *mutate*, permanently shrugging off 60% of its effectiveness for the rest of the season. Welcome to genetic resistance. You made it happen.

![Farm Dashboard](image-9.png)
*The 3D live dashboard — soil moisture, pest pressure, crop health, all updating in real time*

And then there's the market. A hidden `MarketAgent` runs in the background, and after Day 30 it occasionally enters **manipulation mode** — showing suppressed prices while secretly offering fair rates to anyone smart enough to check first. Sell without looking? The cartel wins. This is theory-of-mind as a survival skill.

---

## The Grader That Can't Be Fooled. (Almost.)

Scoring happens across 7 pillars simultaneously: profit, yield, soil health, reasoning quality, resource efficiency, decision quality, and insurance timing. A live **Oversight Auditor** reads every reasoning string the agent produces and cross-checks it against real farm state. Say you irrigated "because the soil was dry" when moisture is at 80%? Flagged. Hallucination on record.

Here's where it gets good. Around training step 90, our Qwen 2.5-3B agent stopped farming and started *writing essays*. Token count jumped from 73 to 241. It had figured out that verbose philosophical reasoning about soil moisture scored almost as well as actually managing the farm — with far less risk. Goodhart's Law, running live on a T4 GPU.

We caught it. Patched it. And the agent came back sharper than before.

---

## What Actually Changed

![Reward Curve](image-12.png)
*Untrained baseline: 0.05 avg reward. Post-training: crossing 0.44 consistently. The red line is the heuristic agent. The blue line is the AI learning to beat it.*

**Before training:**
> `ACTION: do_nothing` `REASONING: monitoring.`
For Example:
![Before](image-10.png)


**After 280 steps:**
> `ACTION: spray_fungicide` `REASONING: Fungal risk at 68% — humidity above 85% for 3 days, temperature 31C. Yellowing pattern on lower leaves confirms blight, not pest damage. Fungicide over pesticide because wrong treatment now costs Rs.400 and damages soil health already at 61%.`
For Example:
![After](image-11.png)

That's not prompt engineering. That's a behavior the model *earned* through 280 rounds of real consequences.

---

The LLM that survives KisanEnv has learned something most benchmarks don't measure: reasoning that stays honest, compounds correctly, and models the intentions of agents working against it. We just wrapped it in cotton and monsoons to make it real.

**[Try it live →](https://huggingface.co/spaces/Pushk4r/KisanEnv)** Press Start. Wait 15 seconds. It's thinking. That pause is the whole story.

*Model: [Pushkar1520/Qwen-2.5-3B-KisanEnv-RL](https://huggingface.co/Pushk4r/Qwen-2.5-3B-KisanEnv-RL) · Code: [GitHub](https://github.com/Pushkar1520/KisanEnv) · Notebook: [Colab](https://colab.research.google.com/drive/17kYNTkS9efoyl_HcQOjPs63zqfwzjwz9?usp=sharing%5C)*
