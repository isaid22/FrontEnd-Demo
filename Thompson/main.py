from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel
from bandit import ThompsonBandit
from bedrock_access import generate_meassages, get_embeddings_batch
from ranker import rank_by_cosine


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
    # Purchase Mortgage (10 arms)
    "purchase_1", "purchase_2", "purchase_3", "purchase_4", "purchase_5",
    "purchase_6", "purchase_7", "purchase_8", "purchase_9", "purchase_10",
    # Refinance (10 arms)
    "refinance_1", "refinance_2", "refinance_3", "refinance_4", "refinance_5",
    "refinance_6", "refinance_7", "refinance_8", "refinance_9", "refinance_10",
    # Home Equity (10 arms)
    "home_equity_1", "home_equity_2", "home_equity_3", "home_equity_4", "home_equity_5",
    "home_equity_6", "home_equity_7", "home_equity_8", "home_equity_9", "home_equity_10"
]
ARM_DESCRIPTIONS = {
    # Purchase Mortgage (10 arms)
    "purchase_1": {
        "message": "Make home ownership a reality for you Find your dream home with <link>Chase</link>",
        "url": "https://www.chase.com/personal/mortgage/mortgage-purchase"
    },
    "purchase_2": {
        "message": "Thinking of buying another home? Let us help you with your <link>Second home or investment properties</link>",
        "url": "https://www.chase.com/personal/mortgage/investment-property"
    },
    "purchase_3": {
        "message": "Let us help you with your journey of <link>homebuying</link>",
        "url": "https://www.chase.com/personal/mortgage/mortgage-purchase/first-time-homebuyer"
    },
    "purchase_4": {
        "message": "Explore your mortgage options with <link>Chase Home Lending</link>",
        "url": "https://www.chase.com/personal/mortgage/mortgage-purchase"
    },
    "purchase_5": {
        "message": "Get pre-approved for your <link>home purchase</link>",
        "url": "https://www.chase.com/personal/mortgage/mortgage-purchase"
    },
    "purchase_6": {
        "message": "Find the right <link>mortgage rate</link> for your home purchase",
        "url": "https://www.chase.com/personal/mortgage/mortgage-rates"
    },
    "purchase_7": {
        "message": "Calculate your mortgage payments with our <link>calculator</link>",
        "url": "https://www.chase.com/personal/mortgage/calculators-resources"
    },
    "purchase_8": {
        "message": "Learn about <link>down payment assistance</link> programs",
        "url": "https://www.chase.com/personal/mortgage/mortgage-purchase/first-time-homebuyer"
    },
    "purchase_9": {
        "message": "Discover <link>jumbo mortgage</link> options for luxury homes",
        "url": "https://www.chase.com/personal/mortgage/mortgage-purchase"
    },
    "purchase_10": {
        "message": "Get started with your <link>home buying journey</link> today",
        "url": "https://www.chase.com/personal/mortgage/mortgage-purchase"
    },
    
    # Refinance (10 arms)
    "refinance_1": {
        "message": "Take advantage of current interest rate, explore <link>refinancing options</link> for your home",
        "url": "https://www.chase.com/personal/mortgage/mortgage-refinance"
    },
    "refinance_2": {
        "message": "Considering refinancing your mortgage? Want to know <link>mortgage rates?</link>",
        "url": "https://www.chase.com/personal/mortgage/mortgage-rates"
    },
    "refinance_3": {
        "message": "Check out these resources to enable your journey. Featured <link>calculators and resources</link>",
        "url": "https://www.chase.com/personal/mortgage/calculators-resources"
    },
    "refinance_4": {
        "message": "Lower your monthly payments with <link>refinancing</link>",
        "url": "https://www.chase.com/personal/mortgage/mortgage-refinance"
    },
    "refinance_5": {
        "message": "Cash-out refinancing for <link>home improvements</link>",
        "url": "https://www.chase.com/personal/mortgage/mortgage-refinance"
    },
    "refinance_6": {
        "message": "Switch to a <link>fixed-rate mortgage</link> for stability",
        "url": "https://www.chase.com/personal/mortgage/mortgage-refinance"
    },
    "refinance_7": {
        "message": "Reduce your loan term with <link>refinancing</link>",
        "url": "https://www.chase.com/personal/mortgage/mortgage-refinance"
    },
    "refinance_8": {
        "message": "Get a free <link>refinance consultation</link>",
        "url": "https://www.chase.com/personal/mortgage/mortgage-refinance"
    },
    "refinance_9": {
        "message": "Compare current <link>refinance rates</link>",
        "url": "https://www.chase.com/personal/mortgage/mortgage-rates"
    },
    "refinance_10": {
        "message": "Start your <link>refinance application</link> online",
        "url": "https://www.chase.com/personal/mortgage/mortgage-refinance"
    },
    
    # Home Equity (10 arms)
    "home_equity_1": {
        "message": "Curious about how to let your home equity work for you? Learn about <link>home equity</link> solutions",
        "url": "https://www.chase.com/personal/home-equity/customer-service"
    },
    "home_equity_2": {
        "message": "Know your equity, are you ready to pay it off? <link>Pay off your HELOC account</link>",
        "url": "https://www.chase.com/personal/home-equity/customer-service/info/pay-off-account"
    },
    "home_equity_3": {
        "message": "Understanding your options in Home Equity Line of Credit (HELOC) <link>End-of-draw options</link>",
        "url": "https://www.chase.com/personal/home-equity/customer-service/info/end-of-draw-options"
    },
    "home_equity_4": {
        "message": "Access your home's equity with a <link>HELOC</link>",
        "url": "https://www.chase.com/personal/home-equity"
    },
    "home_equity_5": {
        "message": "Fund home improvements with <link>home equity loans</link>",
        "url": "https://www.chase.com/personal/home-equity"
    },
    "home_equity_6": {
        "message": "Compare <link>home equity rates</link> and terms",
        "url": "https://www.chase.com/personal/home-equity"
    },
    "home_equity_7": {
        "message": "Use your equity for <link>debt consolidation</link>",
        "url": "https://www.chase.com/personal/home-equity"
    },
    "home_equity_8": {
        "message": "Calculate your available <link>home equity</link>",
        "url": "https://www.chase.com/personal/home-equity"
    },
    "home_equity_9": {
        "message": "Apply online for a <link>home equity line of credit</link>",
        "url": "https://www.chase.com/personal/home-equity"
    },
    "home_equity_10": {
        "message": "Learn about <link>home equity loan</link> vs HELOC options",
        "url": "https://www.chase.com/personal/home-equity"
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
    recommendations = []
    # Add call to Bedrock to generate more messages
    from bedrock_access import generate_meassages

    user_profile_map = {
        "user1": {
            "cluster_type": "Purchase Mortgage",
            "reasoning": "Recommended based onL high purchase page engagement(1.00), strong buying power (%511,045)" 
        },
        "user2": {
            "cluster_type": "Refinance",
            "reasoning": "Recommended based on existing mortgage indicates refinance potential " 
        },
        "user3": {
            "cluster_type": "HELOC",
            "reasoning": "Recommended based on home equity available for borrowing" 
        }
    }

    profile = user_profile_map.get(user_id, {
        "cluster_type": "Purchase Mortgage",
        "reasoning": "Recommended based onL high purchase page engagement(1.00), strong buying power (%511,045)" 
    }) # if user_id is not found, then default to user1 profile

    user_data = {
        "user_login": [
            {
                "cluster_type": profile["cluster_type"],
                "reasoning": profile["reasoning"]
            }
        ]
    }

    cluster_type = user_data["user_login"][0]["cluster_type"]
    if cluster_type == "HELOC":
        cluster_type = "Home Equity Line of Credit" 
    reasoning = user_data["user_login"][0]["reasoning"]

    # generate user_data

    # import uuid
    # unique_id = str(uuid.uuid4())[:8]
    user_message = (
        f"This message is for user that is interested in {cluster_type}. "
        f"Reasoning: {reasoning}. "
        #f"Request ID: {unique_id} - Generate fresh, unique messages."
    )
    
    print("### USER_MESSAGE and TYPE in main.py:", user_message, type(user_message))

    bedrock_messages_list = generate_meassages(user_data, user_message)
    print("Bedrock Messages List from main.py:")
    for i, message in enumerate(bedrock_messages_list, 1):
        print(f'{i:2d}. {message}', flush=True)
    print()

    # Get embedding vector list
    text_list = [user_message] + bedrock_messages_list # First item is user_message, rest are generated message
    vecs = get_embeddings_batch(text_list, model_id="amazon.titan-embed-text-v1", dimensions=1536)

    print(f"##### Batch count: {len(vecs)}")
    print(f"##### Batch lens: {[len(v) for v in vecs]}")

    
    # Rank the message based on their similarity with user_message 
    # Cosine similarity ranking: use the first vector (concatenated prompt) as reference
    ref_vec = vecs[0]
    cand_vecs = vecs[1:]
    ranked_bedrock_messages_list = rank_by_cosine(ref_vec, cand_vecs, bedrock_messages_list)

    # Print all ranked messages with their cosine similarity scores in descending order
    print("=== RANKED MESSAGES BY COSINE SIMILARITY (Descending Order) ===")
    for i, item in enumerate(ranked_bedrock_messages_list, 1):
        print(f"{i:2d}. Similarity: {item['cosine_similarity']:.4f} | Message: {item['message']}")
    print("=" * 60)




    cluster_headline_map = {
        "Purchase Mortgage": ["purchase_1", "purchase_2", "purchase_3", "purchase_4", "purchase_5", 
                             "purchase_6", "purchase_7", "purchase_8", "purchase_9", "purchase_10"],
        "Refinance": ["refinance_1", "refinance_2", "refinance_3", "refinance_4", "refinance_5",
                      "refinance_6", "refinance_7", "refinance_8", "refinance_9", "refinance_10"],
        "Home Equity Line of Credit": ["home_equity_1", "home_equity_2", "home_equity_3", "home_equity_4", "home_equity_5",
                  "home_equity_6", "home_equity_7", "home_equity_8", "home_equity_9", "home_equity_10"]
    }

    # Assign generated messages to the correct headlines
    print(f"DEBUG: cluster_type = '{cluster_type}'")
    print(f"DEBUG: len(ranked_bedrock_messages_list) = {len(ranked_bedrock_messages_list)}")
    print(f"DEBUG: cluster_type in cluster_headline_map = {cluster_type in cluster_headline_map}")
    
    if cluster_type in cluster_headline_map and len(ranked_bedrock_messages_list) >= 10:
        print(f"DEBUG: Assigning messages to {len(cluster_headline_map[cluster_type])} arms")
        for i, headline_id in enumerate(cluster_headline_map[cluster_type]):
            similarity_score = ranked_bedrock_messages_list[i]["cosine_similarity"]
            message = ranked_bedrock_messages_list[i]["message"]
            print(f"DEBUG: Assigning message {i+1} to {headline_id} - Similarity to user description: {similarity_score:.4f}, Message content: {message[:50]}...")

            # Use the ranked message instead of the original order
            ARM_DESCRIPTIONS[headline_id]["message"] = message
        print("DEBUG: Message assignment completed!")
    else:
        print("DEBUG: Condition failed - using hardcoded messages!")

    # Map user_id to interest
    user_interest_map = {
        "user1": ["purchase_1", "purchase_2", "purchase_3", "purchase_4", "purchase_5", 
                  "purchase_6", "purchase_7", "purchase_8", "purchase_9", "purchase_10"],  # Purchase
        "user2": ["refinance_1", "refinance_2", "refinance_3", "refinance_4", "refinance_5",
                  "refinance_6", "refinance_7", "refinance_8", "refinance_9", "refinance_10"],  # Refinance
        "user3": ["home_equity_1", "home_equity_2", "home_equity_3", "home_equity_4", "home_equity_5",
                  "home_equity_6", "home_equity_7", "home_equity_8", "home_equity_9", "home_equity_10"]   # Home Equity
    }
    arm_ids = user_interest_map.get(user_id, [bandit.choose()])
    
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

    print(f"\nFinal Recommendations for {user_id}:", recommendations, flush=True)
    for recommendation in recommendations:
        print(f"- {recommendation['arm_id']}: {recommendation['message']} ({recommendation['url']})", flush=True)

    return {
        "user_id": user_id,
        "recommendations": recommendations
    }