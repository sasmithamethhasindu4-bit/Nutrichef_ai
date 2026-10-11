import sqlite3

from typing import TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END


# ============================================
# 1. DATABASE
# ============================================

def create_database():

    conn = sqlite3.connect("prices.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            ingredient TEXT PRIMARY KEY,
            price REAL
        )
    """)

    data = [
        ("rice", 120),
        ("chicken", 450),
        ("egg", 50),
        ("carrot", 100),
        ("potato", 120),
        ("onion", 100),
        ("tomato", 120),
        ("beans", 150),
        ("dhal", 180),
        ("coconut", 150)
    ]

    cursor.executemany(
        "INSERT OR IGNORE INTO prices VALUES (?, ?)",
        data
    )

    conn.commit()
    conn.close()


# ============================================
# 2. LLM
# ============================================

llm = ChatOllama(
    model="llama3.2",
    temperature=0.3
)


# ============================================
# 3. SYSTEM PROMPT
# ============================================

SYSTEM_PROMPT = """
You are NutriChef AI, a personalized healthy recipe assistant.

Your job is to create a healthy recipe based on:

- User budget
- Health goal
- Food preferences
- Allergies
- Food restrictions

The user does NOT provide portion size or nutrition targets.

You must automatically determine a suitable portion size
based on the user's goal and recipe.

You must also automatically estimate:

- Calories
- Protein
- Carbohydrates
- Fat
- Fiber

Rules:

1. Never use an ingredient that the user is allergic to.
2. Respect the user's food restrictions.
3. Respect the user's food preferences.
4. Keep the recipe within the given budget.
5. Give ingredient quantities.
6. Automatically determine a suitable portion size.
7. Estimate calories per portion.
8. Estimate protein per portion.
9. Estimate carbohydrates per portion.
10. Estimate fat per portion.
11. Estimate fiber per portion.
12. Give cooking instructions.
13. If the recipe is over budget, cheaper ingredients should be considered.
14. Keep the explanation simple and easy to understand.

Important:
Nutrition values are estimates, not medical advice.
"""


# ============================================
# 4. STATE
# ============================================

class RecipeState(TypedDict):

    budget: float
    goal: str
    preferences: str
    allergies: str
    restrictions: str

    recipe: str
    cost: float

    result: str


# ============================================
# 5. BUDGET TOOL
# ============================================

def budget_tool(recipe_text):

    conn = sqlite3.connect("prices.db")

    cursor = conn.cursor()

    cursor.execute(
        "SELECT ingredient, price FROM prices"
    )

    prices = dict(cursor.fetchall())

    conn.close()

    total = 0

    recipe_lower = recipe_text.lower()

    for ingredient, price in prices.items():

        if ingredient.lower() in recipe_lower:
            total += price

    return total


# ============================================
# 6. SUBSTITUTE TOOL
# ============================================

def substitute_tool(recipe_text):

    substitutions = {
        "chicken": "egg",
        "beans": "dhal",
        "coconut": "onion",
        "carrot": "potato"
    }

    result = recipe_text

    for expensive, cheaper in substitutions.items():

        result = result.replace(
            expensive,
            cheaper
        )

    return result


# ============================================
# 7. RECIPE NODE
# ============================================

def generate_recipe(state: RecipeState):

    prompt = f"""
{SYSTEM_PROMPT}

User information:

Budget:
Rs. {state['budget']}

Goal:
{state['goal']}

Food preferences:
{state['preferences']}

Allergies:
{state['allergies']}

Restrictions:
{state['restrictions']}

Create ONE healthy recipe.

Do NOT ask the user for portion size.

Automatically decide the appropriate portion size
based on the user's goal.

Include ingredient quantities so the recipe can be
estimated properly.
"""

    response = llm.invoke(prompt)

    return {
        "recipe": response.content
    }


# ============================================
# 8. CHECK BUDGET
# ============================================

def check_budget(state: RecipeState):

    cost = budget_tool(
        state["recipe"]
    )

    return {
        "cost": cost
    }


# ============================================
# 9. BUDGET DECISION
# ============================================

def budget_decision(state: RecipeState):

    if state["cost"] <= state["budget"]:
        return "within_budget"

    return "over_budget"


# ============================================
# 10. SUBSTITUTE
# ============================================

def cheaper_recipe(state: RecipeState):

    new_recipe = substitute_tool(
        state["recipe"]
    )

    new_cost = budget_tool(
        new_recipe
    )

    return {
        "recipe": new_recipe,
        "cost": new_cost
    }


# ============================================
# 11. FINAL RECIPE + NUTRITION
# ============================================

def final_recipe(state: RecipeState):

    prompt = f"""
{SYSTEM_PROMPT}

Prepare the final recipe using the information below.

Recipe:
{state['recipe']}

Estimated cost:
Rs. {state['cost']}

User budget:
Rs. {state['budget']}

User goal:
{state['goal']}

Create the final answer in this format:

1. Recipe Name

2. Ingredients
- Ingredient
- Quantity

3. Portion
Automatically determine a suitable portion size
based on the user's goal.

4. Nutrition Per Portion
- Calories:
- Protein:
- Carbohydrates:
- Fat:
- Fiber:

5. Cooking Guide
Give simple step-by-step instructions.

6. Estimated Cost
Rs. ...

7. Why This Recipe Fits The Goal
Give a short explanation.

Do NOT ask the user for portion size,
calories, protein, carbohydrates, fat, or fiber.

Calculate/estimate these automatically.

Nutrition values are estimates.
"""

    response = llm.invoke(prompt)

    return {
        "result": response.content
    }


# ============================================
# 12. LANGGRAPH
# ============================================

graph = StateGraph(RecipeState)


graph.add_node(
    "generate_recipe",
    generate_recipe
)

graph.add_node(
    "check_budget",
    check_budget
)

graph.add_node(
    "cheaper_recipe",
    cheaper_recipe
)

graph.add_node(
    "final_recipe",
    final_recipe
)


graph.set_entry_point(
    "generate_recipe"
)


graph.add_edge(
    "generate_recipe",
    "check_budget"
)


graph.add_conditional_edges(
    "check_budget",
    budget_decision,
    {
        "within_budget": "final_recipe",
        "over_budget": "cheaper_recipe"
    }
)


graph.add_edge(
    "cheaper_recipe",
    "final_recipe"
)


graph.add_edge(
    "final_recipe",
    END
)


app = graph.compile()


# ============================================
# 13. RUN PROGRAM
# ============================================

if __name__ == "__main__":

    create_database()

    print("\n===== NutriChef AI =====\n")

    budget = float(
        input("Budget (Rs): ")
    )

    goal = input(
        "Goal (weight loss / weight gain / maintain): "
    )

    preferences = input(
        "Food preferences: "
    )

    allergies = input(
        "Allergies: "
    )

    restrictions = input(
        "Restrictions: "
    )

    # ========================================
    # NO PORTION INPUT
    # ========================================

    initial_state = {
        "budget": budget,
        "goal": goal,
        "preferences": preferences,
        "allergies": allergies,
        "restrictions": restrictions,

        "recipe": "",
        "cost": 0,
        "result": ""
    }


    output = app.invoke(
        initial_state
    )


    print("\n==============================")
    print("       FINAL RECIPE")
    print("==============================\n")

    print(
        output["result"]
    )

    print(
        f"\nEstimated Cost: Rs. {output['cost']}"
    )