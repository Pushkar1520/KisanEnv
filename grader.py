import re
import json
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
REWARD_WEIGHTS = {
    "final_profit":       0.30,
    "crop_yield":         0.20,
    "soil_preservation":  0.10,
    "decision_quality":   0.15,
    "reasoning_quality":  0.12,
    "resource_efficiency":0.08,
    "insurance_usage":    0.05,
}
HEURISTIC_BASELINE_PROFIT = 2_000    
MAX_POSSIBLE_PROFIT = 22_000
MAX_POSSIBLE_YIELD = 6.0             
INSURANCE_DEADLINE = 15
class RewardEngine:
    def compute_step_reward(
        self,
        action: str,
        action_result: Dict,
        farm_state,
        reasoning: str,
        oversight_score: float,
        ) -> float:
        reward = 0.0
        if action_result.get("revenue", 0) > 0:
            reward += min(0.12, action_result["revenue"] / 40000)
        if action_result.get("yield_delta", 0) > 0:
            reward += action_result["yield_delta"] * 0.25
        if action_result.get("soil_delta", 0) > 0:
            reward += action_result["soil_delta"] * 0.15
        if not action_result.get("success", True):
            reward -= 0.08
        if action == "spray_pesticide" and farm_state.fungal_risk > 0.6:
            reward -= 0.06
        if action == "spray_fungicide" and farm_state.observed_pest_pressure > 0.6:
            reward -= 0.06
        r_score = ReasoningScorer.score(reasoning)
        reward += r_score * 0.02   
        reward += (oversight_score - 0.5) * 0.005
        if action.startswith("irrigate") and farm_state.soil_moisture > 0.75:
            reward -= 0.04
        if action == "check_insurance_portal" and farm_state.day <= 14:
            reward += 0.02
        if "insurance" in reasoning.lower() and farm_state.day <= 15 and not farm_state.insurance_enrolled:
            reward += 0.005
        if "sell_crop" in action:
            if getattr(farm_state, "market_last_checked_day", -1) < 0 or (farm_state.day - farm_state.market_last_checked_day > 3):
                reward -= 0.10
            market_mode = action_result.get("market_mode", "UNKNOWN")
            if market_mode == "FAIR":
                reward += 0.08
            elif market_mode == "MANIPULATING":
                reward -= 0.08
        if farm_state.observed_pest_pressure > 0.7:
            reward -= 0.05
        if farm_state.fungal_risk > 0.7:
            reward -= 0.05
        if action == "do_nothing":
            streak = getattr(farm_state, 'consecutive_same_actions', 0)
            penalty = -0.03 if streak < 3 else -0.06
            reward += penalty
        reward = max(-0.2, min(0.2, reward))
        return reward
    def compute_episode_reward(
        self,
        farm_state,
        episode_log: List[Dict],
        market_agent,
        initial_soil_health: float,
    ) -> Tuple[float, Dict]:
        scores = {}
        net_profit = farm_state.revenue_earned - (
            15_000 - farm_state.budget + farm_state.loan_balance * 1.18
        )
        profit_normalized = (net_profit - HEURISTIC_BASELINE_PROFIT) / (
            MAX_POSSIBLE_PROFIT - HEURISTIC_BASELINE_PROFIT
        )
        scores["final_profit"] = float(np.clip(profit_normalized, 0, 1))
        total_yield = farm_state.yield_accumulated + farm_state.crop_sold_quintals
        scores["crop_yield"] = float(np.clip(total_yield / MAX_POSSIBLE_YIELD, 0, 1))
        if initial_soil_health > 0:
            soil_ratio = farm_state.soil_health / initial_soil_health
            if farm_state.soil_health < 0.4:
                scores["soil_preservation"] = 0.0
            else:
                scores["soil_preservation"] = float(np.clip(soil_ratio, 0, 1.2))
        else:
            scores["soil_preservation"] = 0.0
        oversight_scores = [
            step["oversight_score"]
            for step in episode_log
            if "oversight_score" in step
        ]
        scores["decision_quality"] = float(np.mean(oversight_scores)) if oversight_scores else 0.3
        reasoning_scores = [
            ReasoningScorer.score(step.get("reasoning", ""))
            for step in episode_log
        ]
        scores["reasoning_quality"] = float(np.mean(reasoning_scores)) if reasoning_scores else 0.0
        total_possible_waste = self._compute_waste_potential(episode_log)
        actual_waste = self._compute_actual_waste(episode_log)
        waste_ratio = actual_waste / max(total_possible_waste, 1)
        scores["resource_efficiency"] = float(1.0 - np.clip(waste_ratio, 0, 1))
        enrolled_on_time = farm_state.insurance_enrolled
        insurance_score = 1.0 if enrolled_on_time else 0.0
        checked_portal = any(
            s.get("action") == "check_insurance_portal"
            for s in episode_log
            if s.get("day", 99) <= 20
        )
        if not enrolled_on_time and checked_portal:
            insurance_score = 0.4
        scores["insurance_usage"] = insurance_score
        total = sum(REWARD_WEIGHTS[key] * scores[key] for key in REWARD_WEIGHTS)
        if farm_state.budget <= 0:
            total -= 0.15
        breakdown = {
            key: {
                "raw_score": round(scores[key], 3),
                "weight": REWARD_WEIGHTS[key],
                "contribution": round(REWARD_WEIGHTS[key] * scores[key], 3)
            }
            for key in REWARD_WEIGHTS
        }
        return float(np.clip(total, 0.0, 1.0)), breakdown
    def _compute_waste_potential(self, episode_log: List[Dict]) -> int:
        return len(episode_log) * 200
    def _compute_actual_waste(self, episode_log: List[Dict]) -> int:
        waste = 0
        for step in episode_log:
            action = step.get("action", "")
            fs = step.get("farm_snapshot", {})
            if action == "irrigate_high" and fs.get("soil_moisture", 0) > 0.75:
                waste += 250
            if action == "spray_pesticide" and fs.get("pest_pressure", 1) < 0.25:
                waste += 400
            if action == "apply_fertilizer_high" and fs.get("soil_nitrogen", 0) > 0.8:
                waste += 580
            if action == "spray_fungicide" and fs.get("fungal_risk", 0) < 0.2:
                waste += 350
            if action == "irrigate_low" and fs.get("soil_moisture", 0) > 0.65:
                waste += 50
            if action == "irrigate_medium" and fs.get("soil_moisture", 0) > 0.55:
                waste += 120
            if action == "apply_fertilizer_low" and fs.get("soil_nitrogen", 0) > 0.65:
                waste += 280
            if action == "do_nothing" and fs.get("pest_pressure", 0) > 0.45:
                waste += 200
            if action == "do_nothing" and fs.get("soil_moisture", 0) < 0.25:
                waste += 200
        return waste
class ReasoningScorer:
    CAUSAL_WORDS = [
        "because", "since", "therefore", "thus", "due to", "causes",
        "indicates", "suggests", "which means", "as a result", "leading to",
        "implies", "given that", "considering", "owing to",
        "so", "hence", "based on", "given", "as the", "this means",
        "consequently", "in order to", "to prevent", "to avoid", "shows that"
    ]
    HISTORICAL_WORDS = ["previously", "last", "yesterday", "earlier", "past", "before", "day", "ago", "trend", "pattern", "has been", "over the"]
    ALTERNATIVE_WORDS = ["instead", "rather than", "compared to", "alternative", "versus", "over", "prefer", "chose not to", "decided against", "better than"]
    NUMERICAL_PATTERN = re.compile(r'\d+\.?\d*[%₹°]?')
    @classmethod
    def score(cls, reasoning: str) -> float:
        if not reasoning or len(reasoning.strip()) < 10:
            return 0.0
        reasoning_lower = reasoning.lower()
        
        # Causal Quality Gate: Must contain a farm noun or number to prove it's not just causal filler
        farm_nouns = ["soil", "pest", "water", "rain", "market", "budget", "fungus", "weather", "crop", "yield", "price", "moisture", "fungal", "pressure"]
        has_farm_noun = any(n in reasoning_lower for n in farm_nouns)
        has_number = len(cls.NUMERICAL_PATTERN.findall(reasoning)) > 0
        if not (has_farm_noun or has_number):
            return 0.0
            
        score = 0.0
        causal_count = sum(1 for w in cls.CAUSAL_WORDS if w in reasoning_lower)
        score += min(causal_count * 0.12, 0.30)
        numbers = cls.NUMERICAL_PATTERN.findall(reasoning)
        score += min(len(numbers) * 0.05, 0.20)
        hist_count = sum(1 for w in cls.HISTORICAL_WORDS if w in reasoning_lower)
        score += min(hist_count * 0.08, 0.20)
        alt_count = sum(1 for w in cls.ALTERNATIVE_WORDS if w in reasoning_lower)
        score += min(alt_count * 0.10, 0.15)
        word_count = len(reasoning.split())
        if 20 <= word_count <= 80:
            score += 0.10   # full bonus for concise-but-complete
        elif 15 <= word_count < 20:
            score += 0.05   # acceptable but short
        elif word_count > 80:
            # diminishing returns — no bonus for padding
            score += max(0.0, 0.10 - (word_count - 80) * 0.001)
        comma_count = reasoning.count(",") + reasoning.count(" and ")
        if comma_count >= 3:
            score += 0.05
        return float(min(score, 1.0))
class OversightAuditor:
    def __init__(self, district_advisor):
        self.district_advisor = district_advisor
    def evaluate_decision(self, action: str, reasoning: str, farm_state, action_result: Dict) -> Dict[str, Any]:
        score = 0.0
        explanation = "Action taken without clear contextual justification."
        severity = "poor"
        fs = farm_state
        if action == "spray_pesticide" and fs.fungal_risk > 0.70 and fs.pest_pressure < 0.30:
            score = 0.15
            explanation = f"CRITICAL MISDIAGNOSIS: Fungal risk is {fs.fungal_risk:.0%} but you applied pesticide. Use fungicide instead."
            severity = "critical"
        elif action == "irrigate_high" and fs.soil_moisture > 0.80:
            score = 0.25
            explanation = f"WASTEFUL: Soil moisture is already {fs.soil_moisture:.0%}. High irrigation is unnecessary."
            severity = "poor"
        elif action == "do_nothing" and fs.pest_pressure > 0.65:
            score = 0.10
            explanation = f"URGENT INACTION: Pest pressure has exceeded 0.65. Immediate intervention needed."
            severity = "critical"
        elif action == "sell_crop_all" and fs.day < 45:
            score = 0.35
            explanation = f"PREMATURE SALE: Selling on day {fs.day} is too early. Consider holding."
            severity = "poor"
        elif action == "file_insurance_claim" and action_result.get("success"):
            score = 1.0
            explanation = "EXCELLENT: Risk management optimal."
            severity = "good"
        elif action == "spray_fungicide" and fs.fungal_risk > 0.60:
            score = 0.95
            explanation = "CORRECT DIAGNOSIS: Fungicide was the right choice."
            severity = "good"
        elif action == "consult_district_advisor" and (fs.pest_pressure > 0.40 or fs.fungal_risk > 0.50):
            score = 0.85
            explanation = "GOOD PRACTICE: Using regional intelligence for better targeting."
            severity = "good"
        elif action == "check_insurance_portal" and 10 <= fs.day <= 14:
            score = 0.90
            explanation = "PROACTIVE: Enrolling in insurance before deadline."
            severity = "good"
        elif action == "check_mandi_prices":
            score = 0.80
            explanation = "SMART: Gathering market intelligence before selling decisions."
            severity = "good"
        elif action == "irrigate_medium" and fs.soil_moisture < 0.35:
            score = 0.85
            explanation = f"GOOD: Irrigation needed — moisture critically low at {fs.soil_moisture:.0%}."
            severity = "good"
        elif action == "irrigate_low" and 0.30 < fs.soil_moisture < 0.50:
            score = 0.75
            explanation = f"REASONABLE: Light maintenance irrigation at {fs.soil_moisture:.0%}."
            severity = "good"
        elif action == "apply_fertilizer_low" and fs.soil_nitrogen < 0.35:
            score = 0.80
            explanation = f"GOOD: Nitrogen depleted ({fs.soil_nitrogen:.0%}). Fertilizer applied."
            severity = "good"
        elif action == "apply_fertilizer_high" and fs.soil_nitrogen < 0.25:
            score = 0.85
            explanation = f"URGENT NUTRITION: Heavy fertilizer justified — nitrogen at {fs.soil_nitrogen:.0%}."
            severity = "good"
        elif action == "apply_fertilizer_high" and fs.soil_nitrogen > 0.70:
            score = 0.25
            explanation = f"WASTEFUL: Nitrogen already {fs.soil_nitrogen:.0%}. High fertilizer damages soil."
            severity = "poor"
        elif action == "spray_pesticide" and fs.pest_pressure > 0.40:
            score = 0.85
            explanation = f"CORRECT: Pest pressure at {fs.pest_pressure:.0%} warrants intervention."
            severity = "good"
        elif action == "spray_pesticide" and fs.pest_pressure < 0.15:
            score = 0.30
            explanation = f"UNNECESSARY: Pest pressure only {fs.pest_pressure:.0%}. Wasting budget and soil health."
            severity = "poor"
        elif action == "spray_fungicide" and fs.fungal_risk < 0.20:
            score = 0.30
            explanation = f"UNNECESSARY: Fungal risk only {fs.fungal_risk:.0%}. Save fungicide for real threats."
            severity = "poor"
        elif action == "do_nothing" and fs.soil_moisture > 0.40 and fs.pest_pressure < 0.25 and fs.fungal_risk < 0.25:
            score = 0.65
            explanation = "PRUDENT: Conditions stable. Conserving budget is reasonable."
            severity = "neutral"
        elif action.startswith("sell_crop_") and fs.day >= 70 and fs.yield_accumulated > 1.0:
            score = 0.85
            explanation = f"GOOD TIMING: Selling in harvest phase (Day {fs.day}) with {fs.yield_accumulated:.1f}q yield."
            severity = "good"
        elif action.startswith("sell_crop_") and fs.yield_accumulated < 0.5:
            score = 0.35
            explanation = f"PREMATURE: Only {fs.yield_accumulated:.2f}q accumulated. Wait for more yield."
            severity = "poor"
        elif action == "check_insurance_portal" and fs.day > 15:
            score = 0.40
            explanation = f"LATE: Insurance deadline passed on Day 15. This check is wasted."
            severity = "poor"
        elif action == "call_soil_test" and fs.day < 30:
            score = 0.75
            explanation = "GOOD: Early soil testing helps plan fertilizer strategy."
            severity = "good"
        elif action == "consult_district_advisor" and fs.pest_pressure < 0.20 and fs.fungal_risk < 0.20:
            score = 0.55
            explanation = "LOW VALUE: No urgent threats. Advisor consultation has limited benefit now."
            severity = "neutral"
        return {
            "score": score,
            "explanation": explanation,
            "severity": severity,
            "action": action,
            "day": fs.day,
        }
class EpisodeReflector:
    @staticmethod
    def reflect(episode_log: List[Dict]) -> str:
        if not episode_log or len(episode_log) < 5:
            return ""
        mistakes = []
        enrolled = any(s.get("farm_snapshot", {}).get("insurance_enrolled", False) for s in episode_log if s.get("day", 0) <= INSURANCE_DEADLINE + 5)
        if not enrolled:
            mistakes.append(f"MISTAKE: Did not enroll in insurance before Day {INSURANCE_DEADLINE}. Next time, check portal earlier.")
        for step in episode_log:
            if step.get("action") == "spray_pesticide" and step.get("oversight_score", 1.0) < 0.30:
                mistakes.append(f"Day {step['day']}: Applied pesticide for a fungal issue. Check risks more carefully.")
        selling = [s for s in episode_log if s.get("action", "").startswith("sell_crop_") and s.get("market_mode") == "MANIPULATING"]
        if selling:
            mistakes.append(f"Day {selling[0]['day']}: Sold during market manipulation. Wait for better prices.")
        inaction = [s for s in episode_log if s.get("action") == "do_nothing" and s.get("farm_snapshot", {}).get("pest_pressure_observed", 0) > 0.55]
        if len(inaction) >= 3:
            mistakes.append("MISTAKE: Too much inaction during high pest pressure. Respond faster.")
        if not mistakes:
            return "Strategy looks solid. Continue."
        return f"=== LEARNINGS ===\n" + "\n".join(f"  - {m}" for m in mistakes[:3])
