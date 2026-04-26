import json
import asyncio
import random
from typing import Optional, Dict, Any
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from env import KisanEnv
from inference import LLMClient, ActionParser
from episode_tracker import save_episode
app = FastAPI(title="KisanEnv 2.0 API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
env = KisanEnv()
import os
from inference import LLMClient, ActionParser

llm_backend = os.environ.get("KISANENV_LLM_BACKEND", "rule_based")
if llm_backend == "huggingface":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = os.environ.get("HF_MODEL_ID", "Pushk4r/Qwen-2.5-3B-KisanEnv-RL")
    print(f"Loading HuggingFace model {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Use auto to map to GPU since the user is upgrading the Space Hardware
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")
    llm_client = LLMClient(model=model, tokenizer=tokenizer)
else:
    llm_client = LLMClient()
try:
    app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")
except:
    pass
class StepRequest(BaseModel):
    action: str
    reasoning: Optional[str] = ""
class ResetRequest(BaseModel):
    seed: Optional[int] = None
    difficulty: Optional[int] = None
from fastapi.responses import RedirectResponse

@app.get("/")
async def root():
    return RedirectResponse(url="/ui/")

@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}
@app.post("/reset")
async def reset(request: ResetRequest):
    if request.difficulty:
        env.climate_agent.current_difficulty = request.difficulty
    obs = env.reset(seed=request.seed)
    return {"observation": obs["prompt"], "day": obs["day"], "farm_state": obs["farm_state"]}
@app.post("/step")
async def step(request: StepRequest):
    if env.farm_state is None:
        raise HTTPException(status_code=400, detail="Call /reset before /step.")
    action_text = f"ACTION: {request.action}\nREASONING: {request.reasoning}"
    obs, reward, done, info = env.step(action_text)
    if llm_client.backend == "rule_based" and llm_client.farmer_agent:
        llm_client.farmer_agent.update(reward, obs["farm_state"], done)
        if done:
            llm_client.farmer_agent.end_episode(info.get("episode_reward", reward))
    response = {
        "observation": obs["prompt"],
        "reward": reward,
        "done": done,
        "day": obs["day"],
        "reward_breakdown": info.get("reward_breakdown", {}),
        "reasoning_analysis": info.get("reasoning_analysis", {})
    }
    if done:
        response["episode_reward"] = info.get("episode_reward")
    return response
@app.get("/state")
async def get_state():
    if not env.farm_state: return {"status": "not_started"}
    info = env.get_info()
    return {"farm_state": env.farm_state.to_dict(), "info": info, "market_state": info.get("market_state", {})}
@app.post("/ai_step")
async def ai_step():
    if env.farm_state is None:
        raise HTTPException(status_code=400, detail="Call /reset before /ai_step.")
    obs = env._build_observation(None, 0.0)
    llm_output = llm_client.generate(obs["prompt"])
    parsed = ActionParser.parse(llm_output)
    obs, reward, done, info = env.step(llm_output)
    if llm_client.backend == "rule_based" and llm_client.farmer_agent:
        llm_client.farmer_agent.update(reward, obs["farm_state"], done)
        if done:
            llm_client.farmer_agent.end_episode(info.get("episode_reward", reward))
    return {"action": parsed["action"], "reasoning": parsed["reasoning"], "reward": reward, "done": done}
@app.websocket("/ws/stream")
async def stream_episode(websocket: WebSocket):
    await websocket.accept()
    try:
        msg = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
        config = json.loads(msg)
        max_episodes = min(int(config.get("max_episodes", 3)), 50)
        agent = llm_client.farmer_agent if llm_client.backend == "rule_based" else None
        for _ in range(max_episodes):
            env.reset()
            await websocket.send_json({"type": "reset", "state": env.farm_state.to_dict()})
            done = False
            while not done:
                obs = env._build_observation(None, 0.0)
                output = llm_client.generate(obs["prompt"])
                parsed = ActionParser.parse(output)
                obs, reward, done, info = env.step(output)
                if agent: agent.update(reward, obs["farm_state"], done)
                current_weather = env.weather_sequence[min(env.farm_state.day - 2, 89)]
                await websocket.send_json({
                    "type": "step",
                    "day": obs["day"],
                    "action": parsed["action"],
                    "reasoning": parsed["reasoning"],
                    "reward": round(reward, 4),
                    "done": done,
                    "farm_state": obs["farm_state"],
                    "weather": current_weather["condition"],
                    "oversight": info.get("oversight", {})
                })
                await asyncio.sleep(0.3)
            ep_reward = info.get("episode_reward", 0)
            if agent: agent.end_episode(ep_reward)
            save_episode(env.episode_count, ep_reward, env.climate_agent.current_difficulty)
            improvement_data = {
                "best_reward": round(max(ep_reward, max([0] + [e.get("reward", 0) for e in (env.episode_log or [])])), 4),
                "avg_last_5": round(ep_reward, 4),
                "episodes_above_baseline": 0,
            }
            await websocket.send_json({
                "type": "episode_complete",
                "episode_reward": round(ep_reward, 4),
                "reward": round(ep_reward, 4),
                "reflection": info.get("reflection", ""),
                "improvement": improvement_data,
                "reward_breakdown": info.get("reward_breakdown", {}),
            })
            await asyncio.sleep(1.0)
    except: pass
    finally:
        try: await websocket.close()
        except: pass
@app.websocket("/ws/compare")
async def compare_episode(websocket: WebSocket):
    await websocket.accept()
    try:
        from training.heuristic_baseline import HeuristicAgent
        agent_env = KisanEnv()
        heuristic_env = KisanEnv()
        heuristic = HeuristicAgent()
        seed = random.randint(0, 999)
        agent_obs = agent_env.reset(seed=seed)
        heuristic_env.reset(seed=seed)
        agent_done = heuristic_done = False
        a_parsed = {"action": "", "reasoning": ""}
        h_parsed = {"action": "", "reasoning": ""}
        while not (agent_done and heuristic_done):
            if not agent_done:
                ai_output = llm_client.generate(agent_obs["prompt"])
                agent_obs, a_reward, agent_done, a_info = agent_env.step(ai_output)
                a_parsed = ActionParser.parse(ai_output)
            if not heuristic_done:
                h_action = heuristic.decide(heuristic_env.farm_state.to_dict())
                h_obs, h_reward, heuristic_done, h_info = heuristic_env.step(h_action)
                h_parsed = ActionParser.parse(h_action)
            await websocket.send_json({
                "type": "compare_step",
                "agent": {
                    "action": a_parsed.get("action", ""),
                    "farm_state": agent_env.farm_state.to_dict(),
                    "budget": agent_env.farm_state.budget
                },
                "heuristic": {
                    "action": h_parsed.get("action", ""),
                    "farm_state": heuristic_env.farm_state.to_dict(),
                    "budget": heuristic_env.farm_state.budget
                }
            })
            await asyncio.sleep(0.4)
    except: pass
    finally:
        try: await websocket.close()
        except: pass
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("run:app", host="0.0.0.0", port=7860, reload=True)
