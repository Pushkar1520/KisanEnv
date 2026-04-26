import os
import json
import random
import torch
import numpy as np
from typing import List
try:
    from unsloth import FastLanguageModel
    from trl import GRPOTrainer, GRPOConfig
except ImportError:
    print("Warning: Unsloth or TRL not installed.")
from env import KisanEnv
from inference import ActionParser
from grader import ReasoningScorer
from agents.climate_agent import ClimateAgent
SYSTEM_PROMPT = """You are a farming AI assistant. Respond in EXACTLY this format and nothing else:
ACTION: <one_action_name>
REASONING: <one sentence explaining why>
Do not write anything before ACTION:. Do not repeat the farm status. Do not explain your format."""
def format_prompt(raw_prompt, tokenizer):
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": raw_prompt}
        ],
        tokenize=False,
        add_generation_prompt=True
    )
def reward_function(completions, prompts=None, **kwargs) -> List[float]:
    rewards = []
    for content in completions:
        env = KisanEnv()
        env.climate_agent = ClimateAgent(agent_type="qwen_3b")
        env.reset()
        total = 0.0
        done = False
        _, r, done, info = env.step(content)
        total += r
        if not done:
            from training.heuristic_baseline import HeuristicAgent
            h = HeuristicAgent()
            for _ in range(14):
                if done: break
                h_action = h.decide(env.farm_state.to_dict())
                _, r, done, info = env.step(h_action)
                total += r * 0.6
        if done:
            total = info.get("episode_reward", total)
        parsed = ActionParser.parse(content)
        # Using a very rough approximation for tokenizer encoding since tokenizer isn't passed here
        token_count = len(content.split()) * 1.3  # Approx tokens
        length_penalty = max(0.0, (token_count - 60) * 0.0005)
        total -= length_penalty
        
        if not parsed.get("valid_format", True):
            total -= 0.15
        
        total += ReasoningScorer.score(parsed.get("reasoning", "")) * 0.02
        rewards.append(float(np.clip(total, 0, 1)))
    return rewards
def main():
    model_name = "unsloth/Qwen2.5-3B-Instruct"
    max_seq_length = 1024
    output_dir = "kisanenv-qwen-grpo"
    print(f"Loading {model_name}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",  
        random_state=42,                    
    )
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=5e-5,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=5,                         
        bf16=True,
        fp16=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=8,                       
        max_prompt_length=768,                   
        max_completion_length=80,                
        max_steps=500,                           
        save_steps=50,
        dataloader_num_workers=0,                
        remove_unused_columns=False,           
    )
    from datasets import Dataset
    if os.path.exists("training/prompts.json"):
        print("Loading existing prompts from training/prompts.json...")
        with open("training/prompts.json", "r") as f:
            raw = json.load(f)
    else:
        print("No prompts.json found. Generating from KisanEnv...")
        raw = []
        gen_env = KisanEnv()
        for difficulty in [1, 1, 1, 2, 2, 3]:
            gen_env.climate_agent.force_difficulty(difficulty)
            for ep in range(40):
                obs = gen_env.reset()
                raw.append({"prompt": format_prompt(obs["prompt"], tokenizer)})
                for step in range(15):    
                    action = random.choice([
                        "irrigate_medium", "spray_pesticide", "spray_fungicide",
                        "check_insurance_portal", "do_nothing", "apply_fertilizer_low"
                    ])
                    obs, _, done, _ = gen_env.step(f"ACTION: {action}\nREASONING: monitoring.")
                    if not done:
                        raw.append({"prompt": format_prompt(obs["prompt"], tokenizer)})
                    else:
                        break
        os.makedirs("training", exist_ok=True)
        with open("training/prompts.json", "w") as f:
            json.dump(raw, f)
        print(f"Generated and saved {len(raw)} prompts.")
    dataset = Dataset.from_list(raw)
    print(f"Dataset ready: {len(dataset)} prompts")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_function],
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,                    
    )
    torch.cuda.empty_cache()                    
    print("Starting GRPO training...")
    stats = trainer.train()
    print(f"Training complete. Final loss: {stats.training_loss:.4f}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved LoRA adapter to {output_dir}/")
if __name__ == "__main__":
    main()
