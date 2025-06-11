import os
import re
import random
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine
from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph
from langchain_core.prompts import PromptTemplate
from typing import TypedDict, List
import mysql.connector
import mlflow
import tempfile
import requests
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
from mlflow.metrics import make_metric
from sentence_transformers import SentenceTransformer, util

# Load environment variables from .env
dotenv_path = ".env"
load_dotenv(dotenv_path)

# Database read
engine = create_engine("mysql+mysqlconnector://root:@localhost/telecom_data")
df = pd.read_sql("SELECT customer_id, age, plan_type, data_usage_gb, churn_risk FROM customers", engine)
print("\nAll Customer Data:\n", df)

# Initialize your models
creative_llm = OllamaLLM(model="mistral", temperature=0.9, verbose=True)
supervisor_llm = OllamaLLM(model="mistral", temperature=0.2, verbose=True)

# Prompts (same as original)
extract_prompt = PromptTemplate.from_template(
    """
You are given lines extracted from telecom-related marketing webpages:

{web_snippet_lines}

Your task:
Extract **only actual telecom marketing taglines** used in **ads or banners**. Discard:

- General blog advice
- Long descriptions
- Tips or explanations

Constraints:
- Each message must be **one complete sentence**
- **Max 20 words**
- Focus on **telecom offers, benefits, or features**
- Ignore any lines with vague content or without clear marketing intent

Output (only the selected lines):
"""
)

generate_prompt = PromptTemplate.from_template(
    """
Given the examples below, generate ONE new telecom marketing tagline.

Examples:
{examples}

Customer Info:
- Plan Type: {plan_type}
- Data Usage: {data_usage_gb} GB
- Churn Risk: {churn_risk}

Guidelines:
- Write **ONE concise sentence**, max **20 words**
- Do **NOT** include specific numbers (e.g., 9.2GB, ₹199), personal details, or plan names
- Focus on **generic persuasive appeal**, e.g., speed, savings, freedom, unlimited, convenience

Output (ONE line only):
"""
)

supervisor_plan_prompt = PromptTemplate.from_template(
    """
You are reviewing this telecom marketing message for Indian users:

Message:
{line}

Checklist:
1. Is the message under 20 words?
2. Is it clear, persuasive, and benefit-focused?
3. Is it culturally suitable for Indian telecom users?
4. Is it generally a **good, usable message** (your judgement)?

Decide the next step:
- action: **accept** (meets all criteria)
- action: **regenerate** (minor flaws)

Strict output format:
action: <accept|regenerate>
reason: <short reason why>
"""
)

validate_prompt = PromptTemplate.from_template(
    """
Review the marketing message for Indian telecom users.

Message:
{line}

Validation Criteria:
1. Under 20 words
2. Focuses on **one clear benefit** (price, data, speed, etc.)
3. Persuasive and professional tone
4. Culturally suitable for India
5. **No brand names**
6. **No quote marks**

Instructions:
- If all criteria are satisfied, return the message as is.
- Otherwise, rewrite as one **benefit-focused**, **professional sentence** (max 20 words).
- Never include quote marks.

Final output (only one valid line):
"""
)

evaluate_prompt = PromptTemplate.from_template(
    """
Evaluate the quality of the following telecom marketing message for Indian users:

Message:
{line}

Scoring Guide:
- Score from 1 (poor) to 5 (excellent)
- Consider:
  - Clarity
  - Persuasiveness
  - Relevance to Indian telecom users
  - Benefit-focus
  - Is it something you'd use in a real campaign?

Output format only:
score: <1–5>
reason: <brief explanation>
"""
)

# State definition
class MarketingState(TypedDict):
    plan_type: str
    age: int
    data_usage_gb: float
    churn_risk: str
    examples: str
    candidate: str
    decision: str
    final_message: str
    evaluation_score: int
    evaluation_reason: str

# Scraper function (same)
def scrape_jio_marketing():
    url = "https://www.jio.com"  
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"Failed to fetch jio.com page, status code: {response.status_code}")
            return []
    except Exception as e:
        print(f"Exception during requests to jio.com: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")

    promo_texts = []

    for tag in soup.find_all(['h1', 'h2', 'h3', 'p', 'span', 'li']):
        text = tag.get_text(strip=True)
        if len(text.split()) < 5 or len(text.split()) > 20:
            continue
        if re.search(r"(contact|login|signup|terms|privacy|download|more info)", text, re.I):
            continue
        if len(text) < 15:
            continue
        promo_texts.append(text)

    # Deduplicate preserving order
    seen = set()
    unique_promos = []
    for text in promo_texts:
        if text not in seen:
            seen.add(text)
            unique_promos.append(text)

    # Return top 10 promo texts
    return unique_promos[:10]

# Scrape once
scraped_lines = scrape_jio_marketing()
if not scraped_lines:
    print("No marketing examples scraped, falling back to default generic examples.")
    scraped_lines = [
        "Unlimited calls and data at the best prices.",
        "Stay connected with fast 4G speed.",
        "Affordable prepaid plans tailored for you.",
        "Enjoy seamless streaming without buffering.",
        "Recharge now and get extra benefits."
    ]

raw_text = "\n".join(scraped_lines)
prompt = extract_prompt.format(web_snippet_lines=raw_text)
example_messages = supervisor_llm.invoke(prompt).strip()

print(f"\nFinal Extracted Example Messages (used for all users):\n{example_messages}\n")

# Define the LangGraph nodes (same logic as before)

def generate_message(state: MarketingState) -> MarketingState:
    prompt = generate_prompt.format(**state)
    output = creative_llm.invoke(prompt)
    print(f"Generated candidate message:\n{output.strip()}\n")
    return {**state, "candidate": output.strip()}

def supervisor_check(state: MarketingState) -> MarketingState:
    prompt = supervisor_plan_prompt.format(line=state["candidate"])
    decision = supervisor_llm.invoke(prompt)
    print(f"Supervisor LLM decision:\n{decision}\n")
    match = re.search(r"action:\s*(accept|regenerate)", decision.lower())
    action = match.group(1) if match else "accept"
    return {**state, "decision": action}

def validate_message(state: MarketingState) -> MarketingState:
    prompt = validate_prompt.format(line=state["candidate"])
    output = supervisor_llm.invoke(prompt)
    print(f"Final validated message:\n{output.strip()}\n")
    return {**state, "final_message": output.strip()}

def evaluate_message(state: MarketingState) -> MarketingState:
    prompt = evaluate_prompt.format(line=state["final_message"])
    evaluation = supervisor_llm.invoke(prompt)
    print(f"Evaluation output:\n{evaluation}\n")
    match = re.search(r"score:\s*([1-5])", evaluation)
    score = int(match.group(1)) if match else 0
    print(f"Evaluation score: {score} → ", end="")
    if score < 4:
        print("Going to regenerate.")
    else:
        print("Accepted and finished.")
    return {**state, "evaluation_score": score, "evaluation_reason": evaluation.strip()}

# Build the workflow graph
workflow = StateGraph(state_schema=MarketingState)

workflow.add_node("generate", generate_message)
workflow.add_node("supervisor_check", supervisor_check)
workflow.add_node("validate", validate_message)
workflow.add_node("evaluate", evaluate_message)

workflow.set_entry_point("generate")
workflow.add_edge("generate", "supervisor_check")

workflow.add_conditional_edges(
    "supervisor_check",
    lambda state: state["decision"],
    {
        "accept": "validate",
        "regenerate": "generate",
    }
)

workflow.add_edge("validate", "evaluate")

workflow.add_conditional_edges(
    "evaluate",
    lambda state: (
        "generate" if state["evaluation_score"] < 4 else "finish"
    ),
    {
        "generate": "generate",
        "finish": "__end__"
    }
)

graph = workflow.compile()

# Define the run_graph function 
def run_graph(inputs: dict) -> dict:
    return graph.invoke(inputs)

# Initialize MLflow tracking
mlflow.set_experiment("telecom_marketing_message_generation")
mlflow.langchain.autolog()

messages = []

# Run for each customer with individual MLflow runs
for _, row in df.iterrows():
    inputs = {
        "plan_type": row.plan_type,
        "age": int(row.age),
        "data_usage_gb": float(row.data_usage_gb),
        "churn_risk": row.churn_risk,
        "examples": example_messages,
        "candidate": "",
        "decision": "",
        "final_message": "",
        "evaluation_score": 0,
        "evaluation_reason": "",
    }

    print(f"\n--- Processing Customer ID {row.customer_id} ---")

    # Generate prompts to log
    generate_prompt_filled = generate_prompt.format(**inputs)
    supervisor_prompt_filled = supervisor_plan_prompt.format(line=inputs["candidate"])
    validate_prompt_filled = validate_prompt.format(line=inputs["candidate"])
    evaluate_prompt_filled = evaluate_prompt.format(line=inputs["final_message"])

    with mlflow.start_run(run_name=f"customer_{row.customer_id}"):
        mlflow.log_param("customer_id", int(row.customer_id))
        mlflow.log_param("plan_type", inputs["plan_type"])
        mlflow.log_param("age", inputs["age"])
        mlflow.log_param("data_usage_gb", inputs["data_usage_gb"])
        mlflow.log_param("churn_risk", inputs["churn_risk"])
        mlflow.log_param("examples", inputs["examples"])

        with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False, encoding="utf-8") as temp_file:
            temp_file.write("Generate Prompt:\n" + generate_prompt_filled + "\n\n")
            temp_file.write("Supervisor Prompt:\n" + supervisor_prompt_filled + "\n\n")
            temp_file.write("Validate Prompt:\n" + validate_prompt_filled + "\n\n")
            temp_file.write("Evaluate Prompt:\n" + evaluate_prompt_filled + "\n")
            temp_file.flush() 

            mlflow.log_artifact(temp_file.name, artifact_path="prompts")

        os.remove(temp_file.name)

        # Run the graph once
        result = graph.invoke(inputs)

        # Log output
        mlflow.log_param("final_message", result["final_message"])
        mlflow.log_param("evaluation_reason", result["evaluation_reason"])
        mlflow.log_metric("evaluation_score", result["evaluation_score"])

        print(f"Final message: {result['final_message']} | Score: {result['evaluation_score']}")

        messages.append((result["final_message"], int(row.customer_id)))


# Update DB with generated messages
conn = mysql.connector.connect(host="localhost", user="root", password="", database="telecom_data")
cursor = conn.cursor()
cursor.executemany("UPDATE customers SET marketing_message=%s WHERE customer_id=%s", messages)
conn.commit()
cursor.close()
conn.close()

print("************** MLflow-tracked LangGraph pipeline completed. Messages saved ***************")


from mlflow.metrics import make_metric
from sentence_transformers import SentenceTransformer, util
from difflib import SequenceMatcher


# Dummy eval data
eval_data = pd.DataFrame({
    "inputs": [
        {"plan_type": "prepaid", "age": 22, "data_usage_gb": 3, "churn_risk": "low"},
        {"plan_type": "postpaid", "age": 45, "data_usage_gb": 15, "churn_risk": "high"},
    ],
    "ground_truth": [
        "Stay connected with affordable prepaid plans and fast data.",
        "Enjoy premium postpaid benefits and seamless high-speed data.",
    ],
})

# # Sentence transformer model (not used now, just kept for reference)
# st_model = SentenceTransformer("all-MiniLM-L6-v2")


def eval_langgraph_predict(inputs: pd.DataFrame) -> pd.Series:
    preds = []
    for features in inputs["inputs"]:
        state = {
            "plan_type": features["plan_type"],
            "age": features["age"],
            "data_usage_gb": features["data_usage_gb"],
            "churn_risk": features["churn_risk"],
            "examples": example_messages,
            "candidate": "",
            "decision": "",
            "final_message": "",
            "evaluation_score": 0,
            "evaluation_reason": "",
        }
        result = graph.invoke(state) 
        preds.append(result["final_message"])
    return pd.Series(preds)

# Commented out semantic similarity metric for now
# semantic_sim = make_metric(
#     name="semantic_similarity",
#     greater_is_better=True,
#     eval_fn=lambda preds, targets, metrics, **kwargs: util.cos_sim(
#         st_model.encode(preds.tolist(), convert_to_tensor=True),
#         st_model.encode(targets.tolist(), convert_to_tensor=True)
#     ).diagonal().tolist()
# )

# Custom string-based correctness
def simple_correctness(pred: str, truth: str) -> float:
    """Returns a similarity score between 0 and 1."""
    return SequenceMatcher(None, pred.lower(), truth.lower()).ratio()

# Run evaluation (only string-based correctness)
with mlflow.start_run(run_name="string_match_eval", nested=True):
    mlflow.set_tag("eval_type", "simple_correctness_only")

    # # Run evaluation
    # eval_result = mlflow.evaluate(
    #     model=eval_langgraph_predict,
    #     data=eval_data,
    #     targets="ground_truth",
    #     model_type="custom",
    #     extra_metrics=[semantic_sim],
    #     evaluators=["default"],
    # )

    # Run prediction
    predictions = eval_langgraph_predict(eval_data)

    # Evaluate string similarity
    correctness_scores = [
        simple_correctness(pred, truth)
        for pred, truth in zip(predictions, eval_data["ground_truth"])
    ]
    avg_correctness = sum(correctness_scores) / len(correctness_scores)
    mlflow.log_metric("avg_custom_correctness", avg_correctness)

    print("Custom correctness scores:", correctness_scores)
    print("Average custom correctness:", avg_correctness)
