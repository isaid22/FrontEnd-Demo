from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel
from bandit import ThompsonBandit
from bedrock_access import generate_meassages, get_embeddings_batch
from ranker import rank_by_cosine
import asyncio
import threading
import time


app = FastAPI(title="Thompson-Bandit",default_response_class=ORJSONResponse,  version="0.1.0")

# ✅ Dev CORS: allow any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # or ["http://localhost:8080"] if you want to be stricter
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,    # keep False if using "*"
)

# Global cache for processed messages and processing status
message_cache = {}
processing_status = {}  # user_id -> {"status": "processing"/"complete", "timestamp": time}


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
    "home_equity_6", "home_equity_7", "home_equity_8", "home_equity_9", "home_equity_10",
    # Early Explorer (10 arms)
    "early_explorer_1", "early_explorer_2", "early_explorer_3", "early_explorer_4", "early_explorer_5",
    "early_explorer_6", "early_explorer_7", "early_explorer_8", "early_explorer_9", "early_explorer_10"
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
    },
    
    # Early Explorer (10 arms)
    "early_explorer_1": {
        "message": "Imagine your own space - explore home <link>homeownership</link>",
        "url": "https://www.chase.com/personal/mortgage/education"
    },
    "early_explorer_2": {
        "message": "Build equity and stability through <link>owning your home</link>",
        "url": "https://www.chase.com/personal/mortgage/education"
    },
    "early_explorer_3": {
        "message": "Discover the financial benefits of <link>homeownership</link>",
        "url": "https://www.chase.com/personal/mortgage/education"
    },
    "early_explorer_4": {
        "message": "Start building wealth through <link>home ownership</link>",
        "url": "https://www.chase.com/personal/mortgage/education"
    },
    "early_explorer_5": {
        "message": "Create a foundation for your future with <link>your own home</link>",
        "url": "https://www.chase.com/personal/mortgage/education"
    },
    "early_explorer_6": {
        "message": "Explore the pride and security of <link>owning your home</link>",
        "url": "https://www.chase.com/personal/mortgage/education"
    },
    "early_explorer_7": {
        "message": "Take the first step towards <link>homeownership</link> today",
        "url": "https://www.chase.com/personal/mortgage/education"
    },
    "early_explorer_8": {
        "message": "Invest in your future with the stability of <link>home ownership</link>",
        "url": "https://www.chase.com/personal/mortgage/education"
    },
    "early_explorer_9": {
        "message": "Experience the freedom and control of <link>owning your home</link>",
        "url": "https://www.chase.com/personal/mortgage/education"
    },
    "early_explorer_10": {
        "message": "Make memories in a place you can truly call <link>your own</link>",
        "url": "https://www.chase.com/personal/mortgage/education"
    }
}

# Default messages (original ARM_DESCRIPTIONS that will be shown immediately)
DEFAULT_ARM_DESCRIPTIONS = ARM_DESCRIPTIONS.copy()

bandit = ThompsonBandit(ARM_NAMES)

def process_bedrock_messages_background(user_id: str):
    """Background function to process Bedrock messages for a user."""
    global message_cache, processing_status
    
    try:
        print(f"Starting background processing for {user_id}", flush=True)
        processing_status[user_id] = {"status": "processing", "timestamp": time.time()}
        
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
            },
            "user4": {
                "cluster_type": "Early Explorer",
                "reasoning": "Recommended based on interest in exploring homeownership opportunities" 
            }
        }

        profile = user_profile_map.get(user_id, {
            "cluster_type": "Purchase Mortgage",
            "reasoning": "Recommended based onL high purchase page engagement(1.00), strong buying power (%511,045)" 
        })

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

        user_message = (
            f"This message is for user that is interested in {cluster_type}. "
            f"Reasoning: {reasoning}. "
        )
        
        print(f"### USER_MESSAGE for {user_id} in background:", user_message, flush=True)

        # Call Bedrock API
        bedrock_messages_list = generate_meassages(user_data, user_message)
        print(f"Bedrock Messages List for {user_id}:")
        for i, message in enumerate(bedrock_messages_list, 1):
            print(f'{i:2d}. {message}', flush=True)

        # Get embedding vector list
        text_list = [user_message] + bedrock_messages_list
        vecs = get_embeddings_batch(text_list, model_id="amazon.titan-embed-text-v1", dimensions=1536)

        # Rank the messages
        ref_vec = vecs[0]
        cand_vecs = vecs[1:]
        ranked_bedrock_messages_list = rank_by_cosine(ref_vec, cand_vecs, bedrock_messages_list)

        print(f"=== RANKED MESSAGES FOR {user_id} ===")
        for i, item in enumerate(ranked_bedrock_messages_list, 1):
            print(f"{i:2d}. Similarity: {item['cosine_similarity']:.4f} | Message: {item['message']}")
        print("=" * 60)

        cluster_headline_map = {
            "Purchase Mortgage": ["purchase_1", "purchase_2", "purchase_3", "purchase_4", "purchase_5", 
                                 "purchase_6", "purchase_7", "purchase_8", "purchase_9", "purchase_10"],
            "Refinance": ["refinance_1", "refinance_2", "refinance_3", "refinance_4", "refinance_5",
                          "refinance_6", "refinance_7", "refinance_8", "refinance_9", "refinance_10"],
            "Home Equity Line of Credit": ["home_equity_1", "home_equity_2", "home_equity_3", "home_equity_4", "home_equity_5",
                      "home_equity_6", "home_equity_7", "home_equity_8", "home_equity_9", "home_equity_10"],
            "Early Explorer": ["early_explorer_1", "early_explorer_2", "early_explorer_3", "early_explorer_4", "early_explorer_5",
                              "early_explorer_6", "early_explorer_7", "early_explorer_8", "early_explorer_9", "early_explorer_10"]
        }

        # Create processed messages for this user
        processed_messages = {}
        if cluster_type in cluster_headline_map and len(ranked_bedrock_messages_list) >= 10:
            print(f"DEBUG: Assigning background messages to {len(cluster_headline_map[cluster_type])} arms for {user_id}")
            for i, headline_id in enumerate(cluster_headline_map[cluster_type]):
                similarity_score = ranked_bedrock_messages_list[i]["cosine_similarity"]
                message = ranked_bedrock_messages_list[i]["message"]
                processed_messages[headline_id] = {
                    "message": message,
                    "url": DEFAULT_ARM_DESCRIPTIONS[headline_id]["url"]
                }
            print(f"DEBUG: Background message assignment completed for {user_id}!")
        else:
            print(f"DEBUG: Background condition failed for {user_id} - keeping default messages!")
            # Use default messages for this user
            user_interest_map = {
                "user1": ["purchase_1", "purchase_2", "purchase_3", "purchase_4", "purchase_5", 
                          "purchase_6", "purchase_7", "purchase_8", "purchase_9", "purchase_10"],
                "user2": ["refinance_1", "refinance_2", "refinance_3", "refinance_4", "refinance_5",
                          "refinance_6", "refinance_7", "refinance_8", "refinance_9", "refinance_10"],
                "user3": ["home_equity_1", "home_equity_2", "home_equity_3", "home_equity_4", "home_equity_5",
                          "home_equity_6", "home_equity_7", "home_equity_8", "home_equity_9", "home_equity_10"],
                "user4": ["early_explorer_1", "early_explorer_2", "early_explorer_3", "early_explorer_4", "early_explorer_5",
                          "early_explorer_6", "early_explorer_7", "early_explorer_8", "early_explorer_9", "early_explorer_10"]
            }
            arm_ids = user_interest_map.get(user_id, [])
            for arm_id in arm_ids:
                processed_messages[arm_id] = DEFAULT_ARM_DESCRIPTIONS[arm_id].copy()

        # Store in cache
        message_cache[user_id] = processed_messages
        processing_status[user_id] = {"status": "complete", "timestamp": time.time()}
        print(f"Background processing complete for {user_id}", flush=True)
        
    except Exception as e:
        print(f"Error in background processing for {user_id}: {e}", flush=True)
        processing_status[user_id] = {"status": "error", "timestamp": time.time(), "error": str(e)}
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

@app.get("/api/recommendation/immediate")
def get_immediate_recommendation(user_id: str, background_tasks: BackgroundTasks):
    """Get immediate default recommendations while starting background processing."""
    global message_cache, processing_status
    
    # Always start fresh background processing (clear any existing cache)
    if user_id in message_cache:
        del message_cache[user_id]
    if user_id in processing_status:
        del processing_status[user_id]
    
    # Always start new background processing
    background_tasks.add_task(process_bedrock_messages_background, user_id)
    print(f"Started fresh background processing for {user_id}", flush=True)
    
    # Return default messages immediately
    user_interest_map = {
        "user1": ["purchase_1", "purchase_2", "purchase_3", "purchase_4", "purchase_5", 
                  "purchase_6", "purchase_7", "purchase_8", "purchase_9", "purchase_10"],
        "user2": ["refinance_1", "refinance_2", "refinance_3", "refinance_4", "refinance_5",
                  "refinance_6", "refinance_7", "refinance_8", "refinance_9", "refinance_10"],
        "user3": ["home_equity_1", "home_equity_2", "home_equity_3", "home_equity_4", "home_equity_5",
                  "home_equity_6", "home_equity_7", "home_equity_8", "home_equity_9", "home_equity_10"],
        "user4": ["early_explorer_1", "early_explorer_2", "early_explorer_3", "early_explorer_4", "early_explorer_5",
                  "early_explorer_6", "early_explorer_7", "early_explorer_8", "early_explorer_9", "early_explorer_10"]
    }
    
    arm_ids = user_interest_map.get(user_id, [bandit.choose()])
    recommendations = []
    
    for arm_id in arm_ids:
        description = DEFAULT_ARM_DESCRIPTIONS[arm_id]
        arm_state = bandit.state()[arm_id]
        recommendations.append({
            "arm_id": arm_id,
            "message": description["message"],
            "url": description["url"]
        })

    return {
        "user_id": user_id,
        "recommendations": recommendations,
        "source": "default",
        "processing_status": {"status": "processing", "timestamp": time.time()}
    }

@app.get("/api/recommendation/status")
def get_processing_status(user_id: str):
    """Check if background processing is complete for a user."""
    return {
        "user_id": user_id,
        "status": processing_status.get(user_id, {"status": "not_started"})
    }

@app.get("/api/recommendation/processed")
def get_processed_recommendation(user_id: str):
    """Get processed recommendations if available, otherwise default ones."""
    global message_cache
    
    # Check if processed messages are available
    if user_id in message_cache:
        processed_messages = message_cache[user_id]
        recommendations = []
        
        for arm_id, description in processed_messages.items():
            arm_state = bandit.state()[arm_id]
            recommendations.append({
                "arm_id": arm_id,
                "message": description["message"],
                "url": description["url"]
            })
        
        return {
            "user_id": user_id,
            "recommendations": recommendations,
            "source": "processed",
            "processing_status": processing_status.get(user_id, {"status": "unknown"})
        }
    else:
        # Fall back to default recommendations
        return get_immediate_recommendation(user_id, BackgroundTasks())

@app.get("/api/recommendation")
def get_recommendation(user_id: str):
    """Legacy endpoint - now redirects to processed recommendations."""
    return get_processed_recommendation(user_id)