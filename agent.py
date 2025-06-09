import os
import re
import random
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine
from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph
from langchain_core.prompts import PromptTemplate
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from typing import TypedDict
import mysql.connector
from langsmith import traceable, Client

# Load from .evnv, Setup environment
dotenv_path = ".env"
load_dotenv(dotenv_path)
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")

# Initialize LangSmith client
client = Client()
print(f"LangSmith Tracing Enabled for Project: {os.getenv('LANGSMITH_PROJECT')}")

# Database read
engine = create_engine("mysql+mysqlconnector://root:@localhost/telecom_data")
df = pd.read_sql("SELECT customer_id, age, plan_type, data_usage_gb, churn_risk FROM customers", engine)
print("\nAll Customer Data:\n", df)

# Models
creative_llm = OllamaLLM(model="mistral", temperature=0.9, verbose=True)
supervisor_llm = OllamaLLM(model="mistral", temperature=0.2, verbose=True)
search_tool = DuckDuckGoSearchRun()

# Prompts
extract_prompt = PromptTemplate.from_template(
    """
Below are some lines extracted from marketing-related web content:

{web_snippet_lines}

Your task:
- Extract only actual telecom marketing messages used in ads (max 20 words).
- Ignore generic tips, explanations, or blog-style advice.
- Return 3 to 5 short, punchy, real-world sounding telecom messages.

Output (just the lines):
"""
)

generate_prompt = PromptTemplate.from_template(
    """
Given the examples below, write ONE new persuasive short telecom marketing line for this customer.

Examples:
{examples}

Customer:
Plan: {plan_type}
Data Usage: {data_usage_gb} GB
Churn Risk: {churn_risk}

Instructions:
- concise sentence (max 20 words).
- Do not use data_usage, names, age or product explanations or any particular customer details directly.(don't use 9.2gb if the customer uses 9.2gb data) make the messages more generic.

Output:
"""
)

supervisor_plan_prompt = PromptTemplate.from_template(
    """
Message:
{line}

Checklist:
1. Is it under 20 words?
2. Is it persuasive and clear?
3. Is it suitable for Indian telecom users?
4. IMPORTANTLY, Use your judgement to decide if it is a good message.

Decide what to do:
- action: accept (if all good)
- action: regenerate (if small tweak needed)
- action: refetch (if message is bad or examples seem poor)

Format strictly:
action: <accept|regenerate|refetch>
reason: <brief reason>
"""
)

validate_prompt = PromptTemplate.from_template(
    """
Review this telecom marketing message for Indian users:

Message:
{line}

Checklist:
1. Under 20 words.
2. Highlights one clear benefit (e.g., price, data, speed, recharge).
3. Professional and persuasive tone.
4. Suitable for Indian telecom context.
5. Dont mention any telecom company names.
6. No quote marks.

Instructions:
- If all criteria are met, return the message as is.
- Otherwise, rewrite as one professional, benefit-focused sentence (max 20 words), suitable for Indian users.
- Never use quote marks.

Final output (one line only):
"""
)

evaluate_prompt = PromptTemplate.from_template(
    """
Rate the quality of the telecom marketing message below for Indian users on a scale of 1 to 5, where 5 is excellent.

Message:
{line}

Criteria:
- Clarity and persuasiveness
- Relevance to Indian telecom customers
- Brevity and focus on benefits
- IMPORTANTLY, use your judgement to decide if the LLM is working properly and hence give a score.

Output format:
score: <integer from 1 to 5>
reason: <brief explanation>

Only output the above two lines, nothing else.
"""
)

# State
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

# LangGraph Nodes
def fetch_examples(state: MarketingState) -> MarketingState:
    queries = [
        "Indian telecom promotional SMS examples from Jio",
        "Jio marketing slogans 2024 site:jio.com",
        "Examples of SMS marketing by Indian telecom companies",
        "Promotional recharge messages Jio",
        "Indian telecom advertisements"
    ]
    query = random.choice(queries)
    print(f"\nSelected Query: {query}\n")
    raw_results = search_tool.run(query)
    print(f"\nRaw DuckDuckGo results:\n{raw_results[:1000]}...\n")

    # Send full raw text to the LLM and extracts examples
    prompt = extract_prompt.format(web_snippet_lines=raw_results)
    refined_examples = supervisor_llm.invoke(prompt)

    print(f"\nFinal extracted examples:\n{refined_examples.strip()}\n")
    return {**state, "examples": refined_examples.strip()}


def generate_message(state: MarketingState) -> MarketingState:
    prompt = generate_prompt.format(**state)
    output = creative_llm.invoke(prompt)
    print(f"Generated candidate message:\n{output.strip()}\n")
    return {**state, "candidate": output.strip()}

def supervisor_check(state: MarketingState) -> MarketingState:
    prompt = supervisor_plan_prompt.format(line=state["candidate"])
    decision = supervisor_llm.invoke(prompt)
    print(f"Supervisor LLM decision:\n{decision}\n")
    match = re.search(r"action:\s*(accept|regenerate|refetch)", decision.lower())
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
    if score < 2:
        print("Going to refetch.")
    elif score < 4:
        print("Going to regenerate.")
    else:
        print("Accepted and finished.")
    return {**state, "evaluation_score": score, "evaluation_reason": evaluation.strip()}


# LangGraph Workflow
workflow = StateGraph(state_schema=MarketingState)
workflow.add_node("fetch_examples", fetch_examples)
workflow.add_node("generate", generate_message)
workflow.add_node("supervisor_check", supervisor_check)
workflow.add_node("validate", validate_message)
workflow.add_node("evaluate", evaluate_message)

workflow.set_entry_point("fetch_examples")
workflow.add_edge("fetch_examples", "generate")
workflow.add_edge("generate", "supervisor_check")

workflow.add_conditional_edges(
    "supervisor_check",
    lambda state: state["decision"],
    {
        "accept": "validate",
        "regenerate": "generate",
        "refetch": "fetch_examples"
    }
)

workflow.add_edge("validate", "evaluate")
# workflow.set_finish_point("evaluate")

# Add conditional routing after evaluation
workflow.add_conditional_edges(
    "evaluate",
    lambda state: (
        "refetch" if state["evaluation_score"] < 2
        else "regenerate" if state["evaluation_score"] < 4
        else "finish"
    ),
    {
        "refetch": "fetch_examples",
        "regenerate": "generate",
        "finish": "__end__"  
    }
)

graph = workflow.compile()

@traceable(name="generate_telecom_message")
def run_graph(inputs: dict) -> dict:
    return graph.invoke(inputs)

# Main processing loop
messages = []
for _, row in df.iterrows():
    inputs = {
        "plan_type": row.plan_type,
        "age": int(row.age),
        "data_usage_gb": float(row.data_usage_gb),
        "churn_risk": row.churn_risk,
        "examples": "",
        "candidate": "",
        "decision": "",
        "final_message": "",
        "evaluation_score": 0,
        "evaluation_reason": "",
    }
    print(f"\n--- Processing Customer ID {row.customer_id} ---")
    result = run_graph(inputs)
    print(f"Final message: {result['final_message']} | Score: {result['evaluation_score']}")
    messages.append((result["final_message"], int(row.customer_id)))

# Update DB with final marketing messages
conn = mysql.connector.connect(host="localhost", user="root", password="", database="telecom_data")
cursor = conn.cursor()
cursor.executemany("UPDATE customers SET marketing_message=%s WHERE customer_id=%s", messages)
conn.commit()
cursor.close()
conn.close()

print("LangGraph pipeline completed. Messages saved.")
