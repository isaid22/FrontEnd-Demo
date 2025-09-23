from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel
from bandit import ThompsonBandit


app = FastAPI(title="Thompson-Bandit",default_response_class=ORJSONResponse,  version="0.1.0")

# ✅ Dev CORS: allow any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # or ["http://localhost:8080"] if you want to be stricter
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,    # keep False if using "*"
)



# --- config -------------------------------------------------------------
ARM_NAMES = [
    "headline_A", "headline_B", "headline_C",
    "headline_D", "headline_E",
    "headline_F", "headline_G",
    "headline_H", "headline_I"
]
ARM_DESCRIPTIONS = {
    # Alice (Purchase)
    "headline_A": {
        "message": "Make home ownership a reality for you Find your dream home with <link>Chase</link>",
        "url": "https://www.chase.com/personal/mortgage/mortgage-purchase"
    },
    "headline_D": {
        "message": "Thinking of buying another home? Let us help you with your <link>Second home or investment properties</link>",
        "url": "https://www.chase.com/personal/mortgage/investment-property?gclid=EAIaIQobChMIv9umnefjjwMV3F1_AB0xliq2EAAYASAAEgJLovD_BwE"
    },
    "headline_E": {
        "message": "Let us help you with your journey of <link>homebuying</link>",
        "url": "https://www.chase.com/personal/mortgage/mortgage-purchase/first-time-homebuyer?gclid=EAIaIQobChMIv9umnefjjwMV3F1_AB0xliq2EAAYASAAEgJLovD_BwE"
    },
    # Bob (Refinance)
    "headline_B": {
        "message": "Take advantage of current interest rate, explore <link>refinancing options</link> for your home",
        "url": "https://www.chase.com/personal/mortgage/mortgage-refinance"
    },
    "headline_F": {
        "message": "Considering refinancing your mortgage? Want to know <link>mrtgage rates?</link>",
        "url": "https://www.chase.com/personal/mortgage/mortgage-rates?gclid=EAIaIQobChMIv9umnefjjwMV3F1_AB0xliq2EAAYASAAEgJLovD_BwE"
    },
    "headline_G": {
        "message": "Check out these resources to enable your journey. Featured <link>calculators and resources</link>",
        "url": "https://www.chase.com/personal/mortgage/calculators-resources?gclid=EAIaIQobChMIv9umnefjjwMV3F1_AB0xliq2EAAYASAAEgJLovD_BwE"
    },
    # Charlie (Home Equity)
    "headline_C": {
        "message": "Curious about how to let your home equity work for you? Learn about <link>home equity</link> solutions",
        "url": "https://www.chase.com/personal/home-equity/customer-service"
    },
    "headline_H": {
        "message": "Know your equity, are you ready to pay it off?  <link>Pay off your HELOC account</link>",
        "url": "https://www.chase.com/personal/home-equity/customer-service/info/pay-off-account"
    },
    "headline_I": {
        "message": "Understanding your options in Home Equity Line of Credit (HELOC) <link>End-of-draw options</link>",
        "url": "https://www.chase.com/personal/home-equity/customer-service/info/end-of-draw-options"
    }
}
bandit = ThompsonBandit(ARM_NAMES)
# ------------------------------------------------------------------------

class ChoiceOut(BaseModel):
    arm_id: str

class RewardIn(BaseModel):
    arm_id: str
    reward: int   # 0 or 1

@app.get("/choose", response_model=ChoiceOut)
def choose():
    """Pick an arm."""
    return ChoiceOut(arm_id=bandit.choose())

@app.post("/reward")
def reward(payload: RewardIn):
    """Log a binary reward."""
    if payload.arm_id not in bandit.arms:
        raise HTTPException(status_code=404, detail="Unknown arm")
    bandit.reward(payload.arm_id, payload.reward)
    # Print arm info after reward update
    arm_state = bandit.state()[payload.arm_id]
    print(f"\nReward Updated: {payload.arm_id}", flush=True)
    print(f"Current Parameters - Alpha: {arm_state['alpha']:.2f}, Beta: {arm_state['beta']:.2f}", flush=True)
    if arm_state['num_pulls'] > 0:
        print(f"Average Reward: {arm_state['average_reward']:.3f} (Total Pulls: {arm_state['num_pulls']})", flush=True)
    return {"status": "ok"}

@app.get("/state")
def state():
    """Debug: current posterior parameters."""
    return bandit.state()

@app.get("/api/recommendation")
def get_recommendation(user_id: str):
    """Get recommendations for a specific user."""
    # Map user_id to interest
    user_interest_map = {
        "user1": ["headline_A", "headline_D", "headline_E"],  # Purchase
        "user2": ["headline_B", "headline_F", "headline_G"],  # Refinance
        "user3": ["headline_C", "headline_H", "headline_I"]   # Home Equity
    }
    arm_ids = user_interest_map.get(user_id, [bandit.choose()])
    recommendations = []
    for arm_id in arm_ids:
        description = ARM_DESCRIPTIONS[arm_id]
        arm_state = bandit.state()[arm_id]
        print(f"\nArm Selected: {arm_id}", flush=True)
        print(f"Current Parameters - Alpha: {arm_state['alpha']:.2f}, Beta: {arm_state['beta']:.2f}", flush=True)
        if arm_state['num_pulls'] > 0:
            print(f"Average Reward: {arm_state['average_reward']:.3f} (Total Pulls: {arm_state['num_pulls']})", flush=True)
        recommendations.append({
            "arm_id": arm_id,
            "message": description["message"],
            "url": description["url"]
        })
    return {
        "user_id": user_id,
        "recommendations": recommendations
    }