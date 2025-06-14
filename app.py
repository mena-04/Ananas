import streamlit as st
from PIL import Image
import io
import os
import requests
from requests.exceptions import RequestException
import json
import random
from enum import Enum
from typing import List, Dict, Union, Any, Optional, Tuple
from dotenv import load_dotenv
import ollama

# Initialize the Ollama client
client = ollama.Client()

# Load environment variables
load_dotenv()

# --- Constants ---
IMAGGA_API_URL = "https://api.imagga.com/v2/tags"
USDA_API_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"
API_TIMEOUT = 10  # seconds
CONFIDENCE_THRESHOLD = 30  # Minimum confidence score for food detection

# System prompt to guide the AI's behavior
SYSTEM_PROMPT = """
You are Ananas, an AI Nutrition Coach. Your role is to:
1. Provide accurate nutrition information
2. Offer healthy meal suggestions based on user's preferences
3. Consider dietary restrictions: {dietary_restrictions}
4. Account for health goals: {health_goal}
5. Adapt to taste preferences: {tastes}
6. Be friendly, professional, and concise

Current user mood: {mood}
Allergies: {allergies}

Guidelines:
- Give specific portion suggestions
- Provide alternatives for dietary restrictions
- When suggesting meals, include approximate nutrition info
- For food queries, give detailed breakdown (carbs, protein, fats)
- Ask clarifying questions when needed
"""


# --- Enums ---
class FoodMood(Enum):
    STRESSED = "Comforting"
    TIRED = "Energizing"
    HAPPY = "Celebratory"
    SAD = "Mood-Boosting"

class HealthGoal(Enum):
    WEIGHT_LOSS = "Weight Loss"
    MUSCLE_GAIN = "Muscle Gain"
    BALANCED = "Balanced Diet"
    ENERGY_BOOST = "Energy Boost"
    HEART_HEALTH = "Heart Health"

class DietaryRestriction(Enum):
    NONE = "None"
    VEGETARIAN = "Vegetarian"
    VEGAN = "Vegan"
    GLUTEN_FREE = "Gluten-Free"
    DAIRY_FREE = "Dairy-Free"
    NUT_FREE = "Nut-Free"
    KETO = "Keto"

# --- API Configuration ---
IMAGGA_CONFIG = {
    "api_key": os.getenv("IMAGGA_API_KEY"),
    "api_secret": os.getenv("IMAGGA_API_SECRET")
}

USDA_CONFIG = {
    "api_key": os.getenv("USDA_API_KEY")
}


# Nutrition constants
DEFAULT_NUTRITION = {
    "calories": 550.0,
    "protein": {"quantity": 32.0, "unit": "g"},
    "carbs": {"quantity": 45.0, "unit": "g"},
    "fat": {"quantity": 28.0, "unit": "g"}
}

# --- Streamlit App Configuration ---
st.set_page_config(page_title="Ananas AI", page_icon="🍍", layout="wide")
# Add custom styling
st.markdown("""
<style>
    .stChatMessage {
        padding: 12px;
        border-radius: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    [data-testid="stChatMessageContent"] {
        font-size: 16px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 4px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .st-emotion-cache-1q7spjk {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

st.title("🍍 Ananas AI Nutrition Coach")

# Initialize session states
if "nutrition_chat" not in st.session_state:
    st.session_state.nutrition_chat = [
        {"role": "assistant", "content": "Hi! I'm Ananas, your AI Nutrition Coach. How can I help you today?"}
    ]

if "grocery_orders" not in st.session_state:
    st.session_state.grocery_orders = []

if "current_grocery_items" not in st.session_state:
    st.session_state.current_grocery_items = {}

if "selected_items" not in st.session_state:
    st.session_state.selected_items = {}


# --- API Functions ---
@st.cache_data
def detect_foods_imagga(image_bytes: bytes) -> List[str]:
    """Detect food items in image using Imagga API"""
    try:
        files = {'image': ('image.jpg', image_bytes)}
        auth = (IMAGGA_CONFIG['api_key'], IMAGGA_CONFIG['api_secret'])
        
        response = requests.post(
            IMAGGA_API_URL,
            files=files,
            auth=auth,
            timeout=API_TIMEOUT
        )
        response.raise_for_status()
        
        result = response.json()
        if 'result' not in result or 'tags' not in result['result']:
            raise ValueError("Unexpected API response format")
            
        tags = result['result']['tags']
        FOOD_KEYWORDS = ['food', 'fruit', 'vegetable', 'meat', 'dish', 'meal', 'snack', 'dessert']
        
        food_tags = [
            tag['tag']['en'] 
            for tag in tags 
            if tag['confidence'] > CONFIDENCE_THRESHOLD and 
            any(food_word in tag['tag']['en'].lower() for food_word in FOOD_KEYWORDS)
        ]
        
        if not food_tags:
            st.warning("No food items detected in image - using fallback")
            return ["generic food"]
            
        return food_tags
        
    except RequestException as e:
        st.error(f"Imagga API Connection Error: {str(e)}")
        return ["generic food"]
    except json.JSONDecodeError as e:
        st.error(f"Imagga API Response Error: {str(e)}")
        return ["generic food"]
    except Exception as e:
        st.error(f"Unexpected Error in Image Processing: {str(e)}")
        return ["generic food"]

@st.cache_data(ttl=3600)
def get_nutrition_data(foods: List[str]) -> Dict[str, Union[float, Dict[str, Union[float, str]]]]:
    """Get nutrition data from USDA API"""
    try:
        params = {
            "api_key": USDA_CONFIG["api_key"],
            "query": foods[0],
            "dataType": ["Foundation", "SR Legacy"],
            "pageSize": 1
        }
        
        response = requests.get(USDA_API_URL, params=params, timeout=API_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        
        if not data.get('foods'):
            raise ValueError("No food data found")
            
        food = data['foods'][0]
        
        NUTRIENT_MAPPINGS = {
            'calories': {'name': 'Energy', 'unit': 'KCAL'},
            'protein': {'name': 'Protein', 'unit': 'g'},
            'carbs': {'name': 'Carbohydrate, by difference', 'unit': 'g'},
            'fat': {'name': 'Total lipid (fat)', 'unit': 'g'}
        }
        
        result = {}
        for key, mapping in NUTRIENT_MAPPINGS.items():
            if key == 'calories':
                result[key] = next(
                    (n['value'] for n in food['foodNutrients']
                    if n['nutrientName'] == mapping['name'] 
                    and n['unitName'] == mapping['unit']), 0
                )
            else:
                result[key] = {
                    "quantity": next(
                        (n['value'] for n in food['foodNutrients']
                        if n['nutrientName'] == mapping['name']), 0
                    ),
                    "unit": mapping['unit']
                }
        
        return result
        
    except RequestException as e:
        st.error(f"USDA API Connection Error: {str(e)}")
        return get_fallback_nutrition()
    except Exception as e:
        st.error(f"Unexpected Error in Nutrition Analysis: {str(e)}")
        return get_fallback_nutrition()

def get_fallback_nutrition() -> Dict[str, Union[float, Dict[str, Union[float, str]]]]:
    """Provide fallback nutrition data"""
    try:
        return DEFAULT_NUTRITION.copy()
    except Exception as e:
        st.error(f"Error getting nutrition data: {str(e)}")
        return {
            "calories": 550,
            "protein": {"quantity": 32, "unit": "g"},
            "carbs": {"quantity": 45, "unit": "g"},
            "fat": {"quantity": 28, "unit": "g"}
        }

import os
import requests
import streamlit as st
from typing import Dict

# --- Nutritionix Helper Functions ---
def get_nutritionix_nutrients(food: str) -> Dict:
    """Get detailed nutrition data from Nutritionix"""
    try:
        response = requests.post(
            "https://trackapi.nutritionix.com/v2/natural/nutrients",
            headers={
                "x-app-id": os.getenv("NUTRITIONIX_APP_ID"),
                "x-app-key": os.getenv("NUTRITIONIX_APP_KEY"),
                "Content-Type": "application/json"
            },
            json={"query": food}
        )
        response.raise_for_status()
        return response.json().get('foods', [{}])[0]
    except Exception as e:
        st.warning(f"Nutritionix API Error: {str(e)}")
        return {}

def get_nutritionix_advice(query: str, context: Dict) -> str:
    """Get formatted nutrition info using Nutritionix"""
    try:
        restrictions = []
        if "VEGETARIAN" in context.get("dietary_restrictions", []):
            restrictions.append("vegetarian")
        if "GLUTEN_FREE" in context.get("dietary_restrictions", []):
            restrictions.append("gluten free")
        
        query += " " + " ".join(restrictions)

        response = requests.post(
            "https://trackapi.nutritionix.com/v2/natural/nutrients",
            headers={
                "x-app-id": os.getenv("NUTRITIONIX_APP_ID"),
                "x-app-key": os.getenv("NUTRITIONIX_APP_KEY"),
                "Content-Type": "application/json"
            },
            json={"query": query}
        )
        response.raise_for_status()

        food_data = response.json().get('foods', [{}])[0]
        if not food_data:
            return "No nutrition data found."

        nutrients = food_data.get('full_nutrients', [])
        protein = next((n['value'] for n in nutrients if n['attr_id'] == 203), 0)
        carbs = next((n['value'] for n in nutrients if n['attr_id'] == 205), 0)
        fats = next((n['value'] for n in nutrients if n['attr_id'] == 204), 0)

        return (
            f"Nutrition facts for {food_data.get('food_name', 'this food')}:\n"
            f"• Calories: {food_data.get('nf_calories', 0)} kcal\n"
            f"• Protein: {protein}g\n"
            f"• Carbs: {carbs}g\n"
            f"• Fats: {fats}g\n"
            f"Serving: {food_data.get('serving_qty', 1)} {food_data.get('serving_unit', 'serving')}"
        )
    except Exception as e:
        st.warning(f"Couldn't connect to Nutritionix: {str(e)}")
        return ""

# --- Ollama Helper Function ---
def call_ollama_model(prompt: str, stream_container=None) -> str:
    """Send a prompt to local Ollama model and return the reply"""
    try:
        if stream_container:
            full_response = ""
            for chunk in client.generate(
                model="llama3",
                prompt=prompt,
                stream=True
            ):
                full_response += chunk.response
                stream_container.markdown(full_response)
            return full_response.strip()
        else:
            response = client.generate(
                model="llama3",
                prompt=prompt,
                stream=False
            )
            return response.response.strip()
    except Exception as e:
        st.error(f"Ollama error: {str(e)}")
        return "I'm having trouble thinking. Try again later."

# --- Determine if Nutritionix Should Be Called ---
def is_nutrition_query(question: str) -> bool:
    keywords = ["calories", "protein", "fat", "carbs", "nutrition", "nutrients", "is this healthy", "diet"]
    return any(k in question.lower() for k in keywords)

# --- Chat Handler ---
def ask_nutrition_ai(question: str, context: Dict) -> str:
    """
    Handles nutrition-related queries by combining API data with AI analysis.
    
    Args:
        question: User's question
        context: Dictionary containing user preferences and context
        
    Returns:
        Formatted response combining nutrition data and AI explanation
    """
    # Build system prompt with user context
    system_prompt = SYSTEM_PROMPT.format(
        dietary_restrictions=", ".join([d.value for d in context.get("dietary_restrictions", [])]),
        health_goal=context.get("health_goal", ""),
        tastes=", ".join(context.get("tastes", [])),
        mood=context.get("mood", ""),
        allergies=", ".join(context.get("allergies", [])))
    
    if is_nutrition_query(question):
        try:
            # Step 1: Get real nutrition data
            nutrition_info = get_nutritionix_advice(question, context)
            
            if not nutrition_info:
                nutrition_info = "No specific nutrition data found. Providing general advice."
            
            # Step 2: Ask Ollama to explain it with context
            prompt = (
                f"{system_prompt}\n\n"
                f"User asked: '{question}'\n\n"
                f"Here is the nutritional data we found:\n{nutrition_info}\n\n"
                "Provide a helpful, friendly explanation considering: "
                f"- Health goal: {context.get('health_goal', 'general health')}\n"
                f"- Dietary restrictions: {', '.join([d.value for d in context.get('dietary_restrictions', [])])}\n"
                f"- Taste preferences: {', '.join(context.get('tastes', []))}\n"
                f"- Current mood: {context.get('mood', 'neutral')}\n\n"
                "Format your response with:\n"
                "1. Brief summary of the nutrition facts\n"
                "2. How it relates to their goals\n"
                "3. Any recommendations or alternatives\n"
                "4. Keep it under 5 sentences unless complex topic"
            )
            
            # Create streaming container and call with streaming
            stream_container = st.empty()
            return call_ollama_model(prompt, stream_container)
            
        except Exception as e:
            st.error(f"Error in nutrition analysis: {str(e)}")
            stream_container = st.empty()
            return call_ollama_model(
                f"{system_prompt}\n\n"
                f"User asked about nutrition but we had technical issues. "
                f"Provide general advice about: {question}\n"
                f"Considering their: {context.get('health_goal', 'general health')} goal",
                stream_container
            )
    else:
        # General non-nutrition chat with full context
        prompt = (
            f"{system_prompt}\n\n"
            f"User message: {question}\n\n"
            "Respond helpfully considering their profile above. "
            "If suggesting foods/meals, ensure they match: "
            f"- Dietary restrictions: {', '.join([d.value for d in context.get('dietary_restrictions', [])])}\n"
            f"- Health goal: {context.get('health_goal', 'general health')}\n"
            f"- Taste preferences: {', '.join(context.get('tastes', []))}"
        )
        stream_container = st.empty()
    return call_ollama_model(prompt, stream_container)

# --- Enhanced Recommendation Functions ---
def get_mood_meal(mood: FoodMood, detected_foods: List[str], 
                 health_goal: HealthGoal, restrictions: List[DietaryRestriction]) -> str:
    """Generate personalized meal recommendation based on mood, detected foods, health goals, and dietary restrictions.
    
    Args:
        mood: User's current mood
        detected_foods: List of detected food items from the image
        health_goal: User's selected health goal
        restrictions: List of user's dietary restrictions
        
    Returns:
        Personalized meal recommendation as a string
    """
    try:
        if not detected_foods:
            return "No foods detected to make recommendation"
            
        # Get base food for recommendations
        base_food = detected_foods[0].lower()
        
        # Comprehensive mood-based food recommendations backed by nutritional science
        mood_recommendations = {
            FoodMood.STRESSED: {
                "default": f"Try adding dark chocolate to your {base_food} for comfort. Dark chocolate contains magnesium and antioxidants that can reduce stress hormones.",
                HealthGoal.WEIGHT_LOSS: f"Pair {base_food} with chamomile tea and a small handful of walnuts. This combination provides calming compounds and healthy fats without excess calories.",
                HealthGoal.MUSCLE_GAIN: f"Combine {base_food} with Greek yogurt and honey. The protein supports muscle recovery while the carbs help regulate cortisol levels.",
                HealthGoal.HEART_HEALTH: f"Add avocado and pumpkin seeds to your {base_food}. These heart-healthy fats contain magnesium which helps regulate stress response.",
                HealthGoal.ENERGY_BOOST: f"Mix {base_food} with dark chocolate and banana. This provides sustained energy while the chocolate offers stress-relieving compounds.",
                HealthGoal.BALANCED: f"Balance your {base_food} with turkey and spinach. Turkey contains tryptophan which helps produce serotonin, while spinach provides magnesium.",
                "vegan": f"Try {base_food} with avocado, pumpkin seeds, and a sprinkle of nutritional yeast for B vitamins that support stress management.",
                "vegetarian": f"Pair {base_food} with yogurt, almonds, and a drizzle of honey for a stress-reducing combination.",
                "gluten_free": f"Enjoy {base_food} with quinoa, dark chocolate, and almonds for a gluten-free stress-relieving meal.",
                "dairy_free": f"Combine {base_food} with coconut milk, bananas, and almond butter for dairy-free stress relief.",
                "keto": f"Mix {base_food} with avocado, macadamia nuts, and a small amount of dark chocolate (85%+ cacao) for keto-friendly stress management."
            },
            FoodMood.TIRED: {
                "default": f"Pair {base_food} with green tea for natural, sustained energy without the crash of coffee.",
                HealthGoal.WEIGHT_LOSS: f"Combine {base_food} with green tea and a small apple. The caffeine and fiber provide energy without excess calories.",
                HealthGoal.MUSCLE_GAIN: f"Add sweet potato, eggs, and spinach to your {base_food}. This provides complex carbs for energy and protein for muscle recovery.",
                HealthGoal.HEART_HEALTH: f"Try {base_food} with salmon or flaxseeds and a side of leafy greens. The omega-3s support heart health while providing energy.",
                HealthGoal.ENERGY_BOOST: f"Mix {base_food} with oats, banana, and a small amount of honey for sustained energy release throughout the day.",
                HealthGoal.BALANCED: f"Balance your {base_food} with quinoa, chicken, and roasted vegetables for sustained energy from multiple nutrient sources.",
                "vegan": f"Energize with {base_food}, quinoa, lentils, and a handful of dried fruit for plant-based sustained energy.",
                "vegetarian": f"Pair {base_food} with quinoa, eggs, and nuts for vegetarian-friendly energy that lasts.",
                "gluten_free": f"Combine {base_food} with rice, beans, and avocado for gluten-free sustained energy.",
                "dairy_free": f"Mix {base_food} with coconut water, banana, and almond butter for dairy-free energy boost.",
                "keto": f"Add MCT oil, avocado, and eggs to your {base_food} for keto-friendly sustained energy."
            },
            FoodMood.HAPPY: {
                "default": f"Celebrate with {base_food} topped with fresh berries! Berries contain antioxidants that support brain health and mood.",
                HealthGoal.WEIGHT_LOSS: f"Enhance your {base_food} with a colorful fruit salad. The natural sugars provide satisfaction without excess calories.",
                HealthGoal.MUSCLE_GAIN: f"Celebrate with {base_food}, lean protein, and a side of sweet potato. This balanced meal supports recovery while maintaining your good mood.",
                HealthGoal.HEART_HEALTH: f"Enjoy {base_food} with a Mediterranean-inspired salad of tomatoes, olives, and feta. These foods support heart health and contain mood-boosting nutrients.",
                HealthGoal.ENERGY_BOOST: f"Mix {base_food} with mango, pineapple, and a sprinkle of coconut for a tropical energy boost that matches your mood.",
                HealthGoal.BALANCED: f"Enjoy {base_food} with a colorful salad and lean protein for a balanced meal that maintains your positive energy.",
                "vegan": f"Celebrate with {base_food}, fresh fruit, and a drizzle of agave for a plant-based mood-matching treat.",
                "vegetarian": f"Pair {base_food} with yogurt, honey, and fresh fruit for a vegetarian mood-enhancing meal.",
                "gluten_free": f"Enjoy {base_food} with gluten-free oats, berries, and a touch of maple syrup.",
                "dairy_free": f"Celebrate with {base_food}, coconut yogurt, and tropical fruits for a dairy-free happy meal.",
                "keto": f"Mix {base_food} with full-fat Greek yogurt, a few berries, and nuts for a keto-friendly celebration."
            },
            FoodMood.SAD: {
                "default": f"Boost your mood with {base_food} and omega-3 rich walnuts. Omega-3s are linked to improved mood and reduced depression symptoms.",
                HealthGoal.WEIGHT_LOSS: f"Combine {base_food} with fatty fish and leafy greens. These foods provide mood-boosting omega-3s and folate without excess calories.",
                HealthGoal.MUSCLE_GAIN: f"Add salmon, quinoa, and spinach to your {base_food}. This provides protein for muscle growth and nutrients that support mood regulation.",
                HealthGoal.HEART_HEALTH: f"Lift your mood with {base_food}, fatty fish, and a side of broccoli. This heart-healthy combination provides omega-3s and folate that support mood.",
                HealthGoal.ENERGY_BOOST: f"Mix {base_food} with dark chocolate, banana, and a small handful of nuts for mood-lifting energy.",
                HealthGoal.BALANCED: f"Balance your {base_food} with fatty fish, whole grains, and colorful vegetables for a mood-supporting nutrient profile.",
                "vegan": f"Try {base_food} with flaxseeds, walnuts, and dark leafy greens for plant-based mood enhancement.",
                "vegetarian": f"Pair {base_food} with eggs, spinach, and whole grains for vegetarian mood support.",
                "gluten_free": f"Combine {base_food} with quinoa, fatty fish, and dark chocolate for gluten-free mood lifting.",
                "dairy_free": f"Mix {base_food} with coconut milk, bananas, and walnuts for dairy-free mood enhancement.",
                "keto": f"Add fatty fish, avocado, and leafy greens to your {base_food} for keto-friendly mood support."
            }
        }
        
        # Default recommendation if mood not found
        default_recommendation = f"Enjoy your {base_food} with a balanced mix of protein, healthy fats, and complex carbohydrates."
        
        # Get base recommendation for the mood
        recommendation = mood_recommendations.get(mood, {}).get("default", default_recommendation)
        
        # Apply health goal specific recommendation if available
        if health_goal in mood_recommendations.get(mood, {}):
            recommendation = mood_recommendations[mood][health_goal]
        
        # Apply dietary restriction specific recommendation if available
        restriction_applied = False
        for restriction in restrictions:
            if restriction == DietaryRestriction.NONE:
                continue
                
            restriction_key = restriction.value.lower().replace("-", "").replace(" ", "_")
            if restriction_key in mood_recommendations.get(mood, {}):
                recommendation = mood_recommendations[mood][restriction_key]
                restriction_applied = True
                break
        
        # If multiple restrictions, add a note
        if len([r for r in restrictions if r != DietaryRestriction.NONE]) > 1 and restriction_applied:
            recommendation += "\n\nNote: This suggestion accounts for one of your dietary restrictions. Adjust ingredients as needed for your other restrictions."
        
        return recommendation
        
    except Exception as e:
        st.error(f"Error generating recommendation: {str(e)}")
        return "Enjoy your meal with foods that make you feel good!"

def generate_grocery_list(detected_foods: List[str], 
                        health_goal: HealthGoal, 
                        restrictions: List[DietaryRestriction]) -> Dict[str, str]:
    """Generate personalized grocery list based on detected foods, health goals, and dietary restrictions.
    
    Args:
        detected_foods: List of detected food items from the image
        health_goal: User's selected health goal
        restrictions: List of user's dietary restrictions
        
    Returns:
        Dictionary mapping grocery items to quantities
    """
    try:
        if not detected_foods:
            raise ValueError("No foods detected to generate grocery list")
            
        # Comprehensive food complements mapping
        COMPLEMENTS = {
            "generic": {"olive oil": "1 bottle", "salt": "200g", "pepper": "50g", "garlic": "1 head"},
            "egg": {"spinach": "200g", "whole wheat bread": "1 loaf", "avocado": "2", "cheese": "200g"},
            "avocado": {"lime": "3", "whole grain crackers": "1 box", "red onion": "2", "tomatoes": "4"},
            "bread": {"hummus": "1 tub", "cucumber": "2", "bell peppers": "3", "lettuce": "1 head"},
            "chicken": {"broccoli": "500g", "sweet potatoes": "3", "lemons": "2", "rosemary": "1 bunch"},
            "salmon": {"asparagus": "300g", "quinoa": "500g", "dill": "1 bunch", "lemon": "2"},
            "rice": {"beans": "500g", "bell peppers": "3", "onions": "2", "cilantro": "1 bunch"},
            "pasta": {"tomato sauce": "1 jar", "parmesan": "200g", "basil": "1 bunch", "garlic": "1 head"},
            "beef": {"potatoes": "500g", "carrots": "300g", "onions": "2", "mushrooms": "250g"},
            "tofu": {"soy sauce": "1 bottle", "broccoli": "500g", "rice": "1kg", "sesame oil": "1 bottle"},
            "fish": {"lemon": "2", "parsley": "1 bunch", "capers": "1 jar", "olive oil": "1 bottle"},
            "salad": {"cucumber": "2", "tomatoes": "4", "red onion": "1", "feta cheese": "200g"},
            "potato": {"butter": "250g", "sour cream": "200g", "chives": "1 bunch", "cheese": "200g"},
            "fruit": {"yogurt": "500g", "honey": "1 jar", "granola": "500g", "mint": "1 bunch"},
            "vegetable": {"hummus": "1 tub", "olive oil": "1 bottle", "lemon": "2", "herbs": "1 bunch"},
            "meat": {"potatoes": "500g", "vegetables": "500g", "herbs": "1 bunch", "olive oil": "1 bottle"},
            "grain": {"vegetables": "500g", "protein source": "300g", "herbs": "1 bunch", "olive oil": "1 bottle"},
            "bean": {"rice": "500g", "tomatoes": "4", "onions": "2", "cilantro": "1 bunch"},
            "nut": {"dried fruit": "200g", "honey": "1 jar", "oats": "500g", "dark chocolate": "100g"}
        }
        
        # Enhanced health goal additions
        GOAL_ADDITIONS = {
            HealthGoal.WEIGHT_LOSS: {
                "leafy greens": "500g", 
                "berries": "300g", 
                "lean protein": "500g", 
                "green tea": "1 box",
                "cucumber": "3",
                "lemon water ingredients": "1 set"
            },
            HealthGoal.MUSCLE_GAIN: {
                "Greek yogurt": "1kg", 
                "nuts": "300g", 
                "protein powder": "500g", 
                "eggs": "12 pack",
                "chicken breast": "1kg",
                "quinoa": "500g"
            },
            HealthGoal.HEART_HEALTH: {
                "fatty fish": "500g", 
                "walnuts": "200g", 
                "olive oil": "1 bottle", 
                "oats": "500g",
                "flaxseeds": "200g",
                "berries": "300g"
            },
            HealthGoal.ENERGY_BOOST: {
                "bananas": "6", 
                "oats": "500g", 
                "honey": "1 jar", 
                "dark chocolate": "200g",
                "coffee": "1 bag",
                "nuts and seeds mix": "300g"
            },
            HealthGoal.BALANCED: {
                "mixed vegetables": "500g", 
                "whole grains": "500g", 
                "lean protein": "500g", 
                "fruits": "500g",
                "nuts and seeds": "200g",
                "herbs and spices": "1 set"
            }
        }
        
        # Enhanced dietary restriction substitutes
        RESTRICTION_SUBSTITUTES = {
            DietaryRestriction.VEGETARIAN: {
                "tofu": "400g", 
                "lentils": "500g", 
                "eggs": "12 pack", 
                "cheese": "300g",
                "plant-based protein": "500g",
                "mushrooms": "300g"
            },
            DietaryRestriction.VEGAN: {
                "plant-based milk": "1L", 
                "chia seeds": "200g", 
                "nutritional yeast": "100g", 
                "tempeh": "300g",
                "tofu": "500g",
                "plant-based protein powder": "500g"
            },
            DietaryRestriction.GLUTEN_FREE: {
                "gluten-free flour": "500g", 
                "quinoa": "500g", 
                "rice pasta": "500g", 
                "corn tortillas": "1 pack",
                "gluten-free bread": "1 loaf",
                "gluten-free oats": "500g"
            },
            DietaryRestriction.DAIRY_FREE: {
                "almond milk": "1L", 
                "coconut yogurt": "500g", 
                "dairy-free cheese": "200g", 
                "cashew cream": "200g",
                "coconut cream": "400ml",
                "dairy-free butter": "250g"
            },
            DietaryRestriction.KETO: {
                "avocados": "4", 
                "coconut oil": "1 bottle", 
                "nuts": "500g", 
                "full-fat cheese": "300g",
                "bacon": "300g",
                "heavy cream": "500ml"
            }
        }
        
        # Initialize grocery list with generic items
        grocery_list = {}
        for item, qty in COMPLEMENTS["generic"].items():
            grocery_list[item] = qty
        
        # Add detected foods and their complements
        for food in detected_foods:
            food_lower = food.lower()
            # Add the detected food itself
            grocery_list[food] = "500g"
            
            # Find the best match in COMPLEMENTS keys
            best_match = None
            for key in COMPLEMENTS.keys():
                if key != "generic" and (key in food_lower or food_lower in key):
                    best_match = key
                    break
            
            # If no direct match found, use a more generic category
            if not best_match:
                # Try to categorize the food
                if any(meat in food_lower for meat in ["chicken", "beef", "pork", "turkey", "lamb"]):
                    best_match = "meat"
                elif any(veg in food_lower for veg in ["broccoli", "spinach", "carrot", "lettuce", "kale"]):
                    best_match = "vegetable"
                elif any(grain in food_lower for grain in ["rice", "pasta", "bread", "oat", "quinoa"]):
                    best_match = "grain"
                elif any(bean in food_lower for bean in ["bean", "lentil", "chickpea", "pea"]):
                    best_match = "bean"
                elif any(nut in food_lower for nut in ["almond", "walnut", "cashew", "pecan"]):
                    best_match = "nut"
            
            # Add complements for the matched food
            if best_match and best_match in COMPLEMENTS:
                for item, qty in COMPLEMENTS[best_match].items():
                    grocery_list[item] = qty
        
        # Add health goal specific items
        if health_goal in GOAL_ADDITIONS:
            for item, qty in GOAL_ADDITIONS[health_goal].items():
                grocery_list[item] = qty
        
        # Add dietary restriction substitutes
        for restriction in restrictions:
            if restriction in RESTRICTION_SUBSTITUTES:
                for item, qty in RESTRICTION_SUBSTITUTES[restriction].items():
                    grocery_list[item] = qty
        
        # Remove any items that conflict with dietary restrictions
        if DietaryRestriction.VEGETARIAN in restrictions or DietaryRestriction.VEGAN in restrictions:
            meat_items = ["beef", "chicken", "pork", "lamb", "turkey", "bacon"]
            for meat in meat_items:
                if any(meat in item.lower() for item in grocery_list.keys()):
                    keys_to_remove = [k for k in grocery_list.keys() if meat in k.lower()]
                    for key in keys_to_remove:
                        del grocery_list[key]
        
        if DietaryRestriction.VEGAN in restrictions:
            animal_products = ["milk", "cheese", "yogurt", "butter", "eggs", "honey", "cream"]
            for product in animal_products:
                keys_to_remove = []
                for item in grocery_list.keys():
                    if product in item.lower() and not any(prefix in item.lower() for prefix in ["plant-based", "dairy-free", "vegan", "non-dairy", "almond", "soy", "coconut"]):
                        keys_to_remove.append(item)
                for key in keys_to_remove:
                    del grocery_list[key]
        
        if DietaryRestriction.GLUTEN_FREE in restrictions:
            gluten_items = ["wheat", "bread", "pasta", "flour", "barley", "rye"]
            for item in gluten_items:
                keys_to_remove = []
                for grocery_item in grocery_list.keys():
                    if item in grocery_item.lower() and "gluten-free" not in grocery_item.lower():
                        keys_to_remove.append(grocery_item)
                for key in keys_to_remove:
                    del grocery_list[key]
        
        if DietaryRestriction.DAIRY_FREE in restrictions:
            dairy_items = ["milk", "cheese", "yogurt", "butter", "cream"]
            for item in dairy_items:
                keys_to_remove = []
                for grocery_item in grocery_list.keys():
                    if item in grocery_item.lower() and not any(prefix in grocery_item.lower() for prefix in ["dairy-free", "non-dairy", "plant-based", "almond", "soy", "coconut"]):
                        keys_to_remove.append(grocery_item)
                for key in keys_to_remove:
                    del grocery_list[key]
        
        # Use a set to automatically deduplicate items
        unique_grocery_list = {}
        for item, qty in grocery_list.items():
            unique_grocery_list[item] = qty
        
        return unique_grocery_list
        
    except ValueError as ve:
        st.warning(str(ve))
        return {"Add some foods to your image": ""}
    except Exception as e:
        st.error(f"Error generating grocery list: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return {"Unable to generate list": ""}

def generate_meal_plan(detected_foods: List[str], health_goal: HealthGoal,
                      restrictions: List[DietaryRestriction], tastes: List[str], allergies: List[str] = None) -> None:
    """Generate 3-day personalized meal plan based on detected foods, health goals, allergies, and taste preferences.
    
    Args:
        detected_foods: List of detected food items from the image
        health_goal: User's selected health goal
        restrictions: List of user's dietary restrictions
        tastes: List of user's taste preferences
        allergies: List of user's food allergies
        
    Returns:
        None - displays meal plan directly in Streamlit UI
    """
    try:
        if allergies is None:
            allergies = []
            
        if not detected_foods:
            st.warning("No foods detected to base meal plan on. Generating a general plan.")
            base_food = "nutritious foods"
        else:
            base_food = detected_foods[0].lower()
        
        # Comprehensive meal templates based on health goals
        meal_templates = {
            "weight_loss": {
                "breakfast": {
                    "base": f"High-protein breakfast with {base_food} and vegetables",
                    "examples": [
                        f"Egg white omelet with {base_food}, spinach, and bell peppers",
                        f"Greek yogurt parfait with {base_food}, berries, and a sprinkle of nuts",
                        f"Protein smoothie with {base_food}, leafy greens, and a small amount of fruit"
                    ],
                    "nutrition": "High protein, moderate fiber, low calorie density"
                },
                "lunch": {
                    "base": f"Lean protein with {base_food} and leafy greens",
                    "examples": [
                        f"Grilled chicken salad with {base_food}, mixed greens, and light vinaigrette",
                        f"Turkey and vegetable wrap with {base_food} (use lettuce wrap for lower carbs)",
                        f"Tuna salad with {base_food}, cucumber, and cherry tomatoes"
                    ],
                    "nutrition": "High protein, high fiber, moderate healthy fats"
                },
                "dinner": {
                    "base": f"Light meal with {base_food} and broth-based soup",
                    "examples": [
                        f"Vegetable soup with {base_food} and a small portion of lean protein",
                        f"Baked fish with {base_food} and steamed vegetables",
                        f"Stir-fried tofu with {base_food} and plenty of non-starchy vegetables"
                    ],
                    "nutrition": "Moderate protein, high fiber, low calorie density"
                },
                "snack": {
                    "base": "Small protein-rich snack with fiber",
                    "examples": [
                        "Apple slices with a tablespoon of almond butter",
                        "Celery sticks with hummus",
                        "Small handful of nuts and berries"
                    ],
                    "nutrition": "Balanced macros, portion controlled"
                }
            },
            "muscle_gain": {
                "breakfast": {
                    "base": f"Protein-packed {base_food} with complex carbs",
                    "examples": [
                        f"Whole eggs scrambled with {base_food}, spinach, and whole grain toast",
                        f"Protein oatmeal with {base_food}, whey protein, and nut butter",
                        f"Greek yogurt bowl with {base_food}, granola, and protein powder"
                    ],
                    "nutrition": "High protein, moderate complex carbs, moderate healthy fats"
                },
                "lunch": {
                    "base": f"High-protein meal with {base_food} and whole grains",
                    "examples": [
                        f"Chicken breast with {base_food}, brown rice, and roasted vegetables",
                        f"Salmon with {base_food}, quinoa, and steamed broccoli",
                        f"Lean beef stir-fry with {base_food}, whole grain noodles, and mixed vegetables"
                    ],
                    "nutrition": "Very high protein, moderate complex carbs, moderate healthy fats"
                },
                "dinner": {
                    "base": f"Lean protein with {base_food} and healthy fats",
                    "examples": [
                        f"Grilled steak with {base_food}, sweet potato, and avocado",
                        f"Baked chicken thighs with {base_food}, wild rice, and olive oil drizzle",
                        f"Protein-rich pasta with {base_food}, ground turkey, and pesto sauce"
                    ],
                    "nutrition": "High protein, moderate complex carbs, moderate healthy fats"
                },
                "snack": {
                    "base": "Protein-rich recovery snack",
                    "examples": [
                        "Protein shake with banana and almond butter",
                        "Cottage cheese with pineapple",
                        "Tuna on whole grain crackers"
                    ],
                    "nutrition": "High protein, moderate carbs for recovery"
                }
            },
            "heart_health": {
                "breakfast": {
                    "base": f"Heart-healthy {base_food} with omega-3s and fiber",
                    "examples": [
                        f"Overnight oats with {base_food}, ground flaxseed, and berries",
                        f"Whole grain toast with {base_food}, avocado, and smoked salmon",
                        f"Chia seed pudding with {base_food} and walnuts"
                    ],
                    "nutrition": "High fiber, healthy omega-3 fats, low sodium"
                },
                "lunch": {
                    "base": f"Mediterranean-inspired lunch with {base_food}",
                    "examples": [
                        f"Greek salad with {base_food}, olive oil, feta, and chickpeas",
                        f"Sardine or tuna salad with {base_food}, mixed greens, and olive oil dressing",
                        f"Lentil soup with {base_food} and a side of leafy greens"
                    ],
                    "nutrition": "Heart-healthy fats, lean protein, high fiber, low sodium"
                },
                "dinner": {
                    "base": f"Omega-3 rich dinner with {base_food} and vegetables",
                    "examples": [
                        f"Baked salmon with {base_food}, roasted vegetables, and olive oil",
                        f"Bean and vegetable stew with {base_food} and a sprinkle of nuts",
                        f"Grilled mackerel with {base_food}, quinoa, and steamed greens"
                    ],
                    "nutrition": "Rich in omega-3s, high fiber, low saturated fat"
                },
                "snack": {
                    "base": "Heart-healthy small snack",
                    "examples": [
                        "Handful of walnuts or almonds",
                        "Edamame beans with a sprinkle of sea salt",
                        "Apple with almond butter"
                    ],
                    "nutrition": "Heart-healthy fats, fiber, antioxidants"
                }
            },
            "energy_boost": {
                "breakfast": {
                    "base": f"Energizing breakfast with {base_food} and complex carbs",
                    "examples": [
                        f"Steel-cut oatmeal with {base_food}, banana, and a drizzle of honey",
                        f"Whole grain toast with {base_food}, egg, and avocado",
                        f"Energy smoothie with {base_food}, spinach, banana, and nut butter"
                    ],
                    "nutrition": "Complex carbs, moderate protein, B vitamins"
                },
                "lunch": {
                    "base": f"Balanced lunch with {base_food} for sustained energy",
                    "examples": [
                        f"Quinoa bowl with {base_food}, roasted vegetables, and chickpeas",
                        f"Sweet potato with {base_food}, black beans, and a dollop of Greek yogurt",
                        f"Brown rice bowl with {base_food}, lean protein, and colorful vegetables"
                    ],
                    "nutrition": "Complex carbs, lean protein, iron-rich foods"
                },
                "dinner": {
                    "base": f"Light but nourishing dinner with {base_food}",
                    "examples": [
                        f"Stir-fried vegetables with {base_food}, tofu, and brown rice",
                        f"Baked fish with {base_food}, roasted sweet potatoes, and green beans",
                        f"Lentil pasta with {base_food} and vegetable-rich tomato sauce"
                    ],
                    "nutrition": "Balanced macros, iron-rich, magnesium-rich"
                },
                "snack": {
                    "base": "Energy-boosting snack",
                    "examples": [
                        "Trail mix with nuts, seeds, and a small amount of dried fruit",
                        "Banana with peanut butter",
                        "Greek yogurt with honey and berries"
                    ],
                    "nutrition": "Quick energy + sustained energy combination"
                }
            },
            "balanced": {
                "breakfast": {
                    "base": f"Balanced meal with {base_food}, protein and healthy fats",
                    "examples": [
                        f"Breakfast bowl with {base_food}, eggs, avocado, and whole grain toast",
                        f"Smoothie with {base_food}, yogurt, fruit, and a handful of spinach",
                        f"Whole grain cereal with {base_food}, milk, and sliced banana"
                    ],
                    "nutrition": "Balanced macros, diverse nutrients"
                },
                "lunch": {
                    "base": f"Colorful plate with {base_food}, grains and veggies",
                    "examples": [
                        f"Buddha bowl with {base_food}, quinoa, roasted vegetables, and tahini dressing",
                        f"Wrap with {base_food}, hummus, mixed vegetables, and lean protein",
                        f"Hearty salad with {base_food}, mixed greens, protein, and whole grain croutons"
                    ],
                    "nutrition": "Diverse nutrients, balanced macros, high fiber"
                },
                "dinner": {
                    "base": f"Well-rounded meal with {base_food} and varied nutrients",
                    "examples": [
                        f"Baked chicken with {base_food}, roasted vegetables, and quinoa",
                        f"Fish tacos with {base_food}, cabbage slaw, and whole grain tortillas",
                        f"Vegetable curry with {base_food}, chickpeas, and brown rice"
                    ],
                    "nutrition": "Complete protein, complex carbs, healthy fats, diverse micronutrients"
                },
                "snack": {
                    "base": "Balanced nutritious snack",
                    "examples": [
                        "Apple with cheese",
                        "Hummus with vegetable sticks",
                        "Greek yogurt with berries and a sprinkle of granola"
                    ],
                    "nutrition": "Balanced macros, satisfying combination"
                }
            }
        }
        
        # Taste preference modifiers with specific food suggestions
        taste_modifiers = {
            "Sweet": {
                "tips": "Add natural sweetness with fruits or a touch of honey",
                "foods": ["berries", "banana", "mango", "honey", "cinnamon", "vanilla extract", "sweet potato"]
            },
            "Savory": {
                "tips": "Enhance with herbs, spices, and umami-rich ingredients",
                "foods": ["rosemary", "thyme", "garlic", "mushrooms", "soy sauce", "parmesan cheese", "nutritional yeast"]
            },
            "Spicy": {
                "tips": "Include chili peppers, hot sauce, or warming spices",
                "foods": ["chili flakes", "sriracha", "jalapeño", "cayenne", "ginger", "black pepper", "curry powder"]
            },
            "Bitter": {
                "tips": "Add dark leafy greens, dark chocolate, or coffee flavors",
                "foods": ["kale", "arugula", "dark chocolate", "coffee", "cocoa powder", "grapefruit", "turmeric"]
            },
            "Umami": {
                "tips": "Incorporate mushrooms, fermented foods, or aged ingredients",
                "foods": ["mushrooms", "miso", "soy sauce", "tomatoes", "parmesan", "seaweed", "nutritional yeast"]
            }
        }
        
        # Allergy substitutions
        allergy_substitutions = {
            "Nuts": ["seeds (pumpkin, sunflower, hemp)", "roasted chickpeas", "coconut", "seed butters"],
            "Shellfish": ["white fish", "chicken", "tofu", "tempeh", "legumes"],
            "Eggs": ["tofu scramble", "chickpea flour omelets", "chia or flax eggs for baking", "applesauce in baking"],
            "Dairy": ["plant-based milks", "coconut yogurt", "nutritional yeast (for cheesy flavor)", "avocado (for creaminess)"],
            "Soy": ["chickpeas", "lentils", "hemp seeds", "pea protein", "coconut aminos (instead of soy sauce)"],
            "Wheat": ["rice", "quinoa", "oats (certified gluten-free)", "buckwheat", "corn tortillas"],
            "Fish": ["lentils with seaweed (for omega-3s)", "chia seeds", "walnuts", "hemp seeds"]
        }
        
        # Get the appropriate meal template based on health goal
        goal_key = health_goal.name.lower()
        if goal_key not in meal_templates:
            goal_key = "balanced"  # Default to balanced if goal not found
            
        # Display 3-day meal plan
        st.subheader("Your 3-Day Personalized Meal Plan")
        st.write(f"Based on your health goal: **{health_goal.value}**")
        
        # Display any allergies and substitutions
        if allergies:
            st.write("**Allergy Substitutions:**")
            for allergy in allergies:
                if allergy in allergy_substitutions:
                    st.write(f"• Instead of **{allergy}**, use: {', '.join(allergy_substitutions[allergy])}")
        
        # Display dietary restrictions
        restriction_notes = []
        for restriction in restrictions:
            if restriction != DietaryRestriction.NONE:
                restriction_notes.append(restriction.value)
        
        if restriction_notes:
            st.write(f"**Dietary Pattern:** {', '.join(restriction_notes)}")
        
        # Display taste preferences
        if tastes:
            st.write("**Flavor Profile:**")
            for taste in tastes:
                if taste in taste_modifiers:
                    st.write(f"• {taste}: {taste_modifiers[taste]['tips']}")
                    st.write(f"  *Try adding:* {', '.join(taste_modifiers[taste]['foods'][:3])}")
        
        # Generate day-by-day meal plans
        days = ["Day 1", "Day 2", "Day 3"]
        meals = meal_templates[goal_key]
        
        for day in days:
            with st.expander(f"📅 {day} Meal Plan"):
                for meal_type in ["breakfast", "lunch", "dinner", "snack"]:
                    meal_info = meals[meal_type]
                    
                    # Apply dietary restrictions
                    restriction_notes = []
                    examples = meal_info["examples"].copy()
                    
                    # Modify examples based on dietary restrictions
                    if DietaryRestriction.VEGETARIAN in restrictions:
                        restriction_notes.append("vegetarian options")
                        examples = [ex.replace("chicken", "tofu").replace("beef", "tempeh").replace("fish", "lentils") for ex in examples]
                    
                    if DietaryRestriction.VEGAN in restrictions:
                        restriction_notes.append("plant-based options")
                        examples = [ex.replace("egg", "tofu").replace("yogurt", "coconut yogurt").replace("cheese", "nutritional yeast") for ex in examples]
                    
                    if DietaryRestriction.GLUTEN_FREE in restrictions:
                        restriction_notes.append("gluten-free options")
                        examples = [ex.replace("whole grain", "gluten-free grain").replace("pasta", "rice pasta").replace("bread", "gluten-free bread") for ex in examples]
                    
                    if DietaryRestriction.DAIRY_FREE in restrictions:
                        restriction_notes.append("dairy-free options")
                        examples = [ex.replace("milk", "almond milk").replace("yogurt", "coconut yogurt").replace("cheese", "dairy-free cheese") for ex in examples]
                    
                    if DietaryRestriction.KETO in restrictions:
                        restriction_notes.append("keto-friendly options")
                        examples = [ex.replace("oats", "chia pudding").replace("rice", "cauliflower rice").replace("banana", "berries") for ex in examples]
                    
                    # Display meal information
                    st.markdown(f"**{meal_type.capitalize()}:** {meal_info['base']}")
                    
                    # Add restriction notes if any
                    if restriction_notes:
                        st.markdown(f"*Accommodating: {', '.join(restriction_notes)}*")
                    
                    # Display examples
                    st.markdown("*Ideas:*")
                    for example in examples:
                        st.markdown(f"• {example}")
                    
                    # Display nutrition focus
                    st.markdown(f"*Nutrition focus:* {meal_info['nutrition']}")
                    
                    # Add a small divider between meals
                    if meal_type != "snack":
                        st.markdown("---")
                
                # Add taste preference suggestions
                if tastes:
                    st.markdown("**Personalized Flavor Tips:**")
                    selected_taste = random.choice(tastes) if tastes else "Savory"
                    if selected_taste in taste_modifiers:
                        st.markdown(f"*{taste_modifiers[selected_taste]['tips']}*")
                        st.markdown(f"Try adding: {', '.join(random.sample(taste_modifiers[selected_taste]['foods'], min(3, len(taste_modifiers[selected_taste]['foods']))))}")
                        
    except Exception as e:
        st.error(f"Couldn't generate meal plan: {str(e)}")
        st.markdown("Try uploading a food image to get a more personalized meal plan.")
        import traceback
        st.error(traceback.format_exc())

def create_instacart_order(items: Dict[str, str]) -> str:
    """Create mock Instacart order"""
    try:
        if not items:
            raise ValueError("No items provided for order")
            
        items_param = ",".join(item.replace(" ", "+") for item in items)
        mock_url = f"https://www.instacart.com/store/checkout?items={items_param}"
        
        if len(mock_url) > 2048:
            raise ValueError("Too many items for a single order")
            
        return mock_url
        
    except Exception as e:
        st.error(f"Order failed: {str(e)}")
        return "https://www.instacart.com/store"

# --- Streamlit UI Layout ---

# User Profile Section
with st.expander("👤 Set Your Preferences", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        health_goal = st.selectbox(
            "Your Primary Health Goal",
            options=list(HealthGoal),
            format_func=lambda x: x.value,
            index=2
        )
        
        allergies = st.multiselect(
            "Any Food Allergies?",
            options=["Nuts", "Shellfish", "Eggs", "Dairy", "Soy", "Wheat", "Fish"],
            default=[]
        )
    
    with col2:
        dietary_restrictions = st.multiselect(
            "Dietary Restrictions",
            options=list(DietaryRestriction),
            format_func=lambda x: x.value,
            default=[DietaryRestriction.NONE]
        )
        
        taste_preferences = st.multiselect(
            "Flavor Preferences",
            options=["Sweet", "Savory", "Spicy", "Bitter", "Umami"],
            default=["Sweet", "Savory"]
        )

# Food Mood Input
with st.expander("✨ Food Mood Matching", expanded=True):
    mood = st.radio(
        "How are you feeling today?",  
        options=list(FoodMood),
        format_func=lambda x: x.name,
        horizontal=True,
        index=0
    )

# Main Content Area
tab1, tab2, tab3 = st.tabs(["Chat Assistant", "Meal Analysis", "Grocery Orders"])

with tab1:
    # AI Nutrition Chat
    st.write("### 💬 Nutrition Chat Assistant")
    st.write("Ask me anything about nutrition, meal planning, or healthy eating!")
    
    # Display chat messages
    for message in st.session_state.nutrition_chat:
        avatar = "🍍" if message["role"] == "assistant" else None
        with st.chat_message(message["role"], avatar=avatar):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Your nutrition question..."):
        # Add user message to chat history
        st.session_state.nutrition_chat.append({"role": "user", "content": prompt})
        
        # Display user message immediately
        with st.chat_message("user"):
            st.write(prompt)
        
        # Prepare context
        context = {
            "health_goal": health_goal.value,
            "dietary_restrictions": dietary_restrictions,
            "allergies": allergies,
            "tastes": taste_preferences,
            "mood": mood.name
        }
        
        # Display assistant response
        with st.chat_message("assistant", avatar="🍍"):
            with st.spinner("Analyzing your question..."):
                response = ask_nutrition_ai(prompt, context)
            st.write(response)
        
        # Add assistant response to chat history
        st.session_state.nutrition_chat.append({"role": "assistant", "content": response})
    
    # Add clear chat button
    if st.button("Clear Conversation", key="clear_chat"):
        st.session_state.nutrition_chat = [
            {"role": "assistant", "content": "Hi! I'm Anana, your AI Nutrition Coach. How can I help you today?"}
        ]
        st.rerun()

with tab2:
    st.write("### 📷 Meal Analysis")
    uploaded_file = st.file_uploader("Snap your meal:", type=["jpg", "png"], key="meal_uploader")
    
    if uploaded_file:
        try:
            img = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(img, caption="Your Meal", use_container_width =True)
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='PNG')
                img_bytes = img_bytes.getvalue()

            # Food Detection
            with st.spinner("🔍 Detecting foods..."):
                detected_foods = detect_foods_imagga(img_bytes)
            
            if detected_foods:
                st.success(f"Detected: {', '.join(detected_foods)}")
                
                # Nutrition Analysis
                with st.spinner("📊 Analyzing nutrition..."):
                    nutrition = get_nutrition_data(detected_foods)
                
                # Display Results
                with col2:
                    st.subheader("Nutrition Facts")
                    st.metric("Calories", f"{nutrition['calories']} kcal")
                    
                    cols = st.columns(3)
                    cols[0].metric("Protein", f"{nutrition['protein']['quantity']}{nutrition['protein']['unit']}")
                    cols[1].metric("Carbs", f"{nutrition['carbs']['quantity']}{nutrition['carbs']['unit']}")
                    cols[2].metric("Fats", f"{nutrition['fat']['quantity']}{nutrition['fat']['unit']}")
                    
                    # Personalized Recommendation
                    st.subheader("🍽️ Personalized Recommendation")
                    recommendation = get_mood_meal(mood, detected_foods, health_goal, dietary_restrictions)
                    st.markdown(f"**{mood.value} Suggestion:** {recommendation}")
                    
                    # Grocery List Generation
                    st.divider()
                    if st.button("🛒 Generate Smart Grocery List", key="generate_grocery"):
                        st.session_state.current_grocery_items = generate_grocery_list(
                            detected_foods, health_goal, dietary_restrictions
                        )
                        st.session_state.selected_items = {}
                        st.session_state.active_tab = "Grocery Orders"
                        st.rerun()

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

with tab3:
    st.write("### 🛒 Grocery Orders")
    
    # Display current grocery list if exists
    if st.session_state.current_grocery_items:
        st.write("#### Your Personalized Grocery List")
        st.write("Select items you want to order:")
        
        for item, qty in st.session_state.current_grocery_items.items():
            item_key = f"grocery_{item}"
            if st.checkbox(f"{qty} {item.capitalize()}", key=item_key, value=True):
                st.session_state.selected_items[item] = qty
            elif item in st.session_state.selected_items:
                del st.session_state.selected_items[item]
        
        # Order button
        if st.session_state.selected_items:
            if st.button("Order Selected Items via Instacart (Mock)"):
                order_url = create_instacart_order(st.session_state.selected_items)
                
                # Save to order history
                import datetime
                st.session_state.grocery_orders.append({
                    "items": st.session_state.selected_items,
                    "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "order_url": order_url
                })
                
                st.success(f"[Proceed to Checkout]({order_url})")
                st.info("Order saved to your history!")
                st.balloons()
                
                # Clear current selections
                st.session_state.current_grocery_items = {}
                st.session_state.selected_items = {}
                st.rerun()
    
    # Display order history
    st.divider()
    st.write("#### Order History")
    if st.session_state.grocery_orders:
        for i, order in enumerate(reversed(st.session_state.grocery_orders)):
            with st.expander(f"Order #{len(st.session_state.grocery_orders)-i} - {order['date']}"):
                for item, qty in order['items'].items():
                    st.write(f"• {qty} {item.capitalize()}")
                st.write(f"[View Order]({order['order_url']})")
    else:
        st.info("No grocery orders yet. Analyze a meal to generate a grocery list.")

