# =============================================================================
# Household Financial Health Evaluator -- A LangGraph Learning Project
# ==============================================================================
#
# This project teaches LangGraph concepts by building a financial health
# checker that evaluates a household's budget  from three angles.
#
#
# WHAT THIS DOES:
# A user enters their income, essential costs, discretionary spending, debts,
# current savings, and savings goals. The system runs 3 specialist nodes in
# PARALLEL (cash flow, debt pressure, savings readiness), then a decision
# node classifies the budget as STABLE or NEEDING CORRECTION, and routes to
# either a maintenance plan or a recovery plan.
#
# This is educational only and is NOT professional financial advice.
#
# LANGGRAPH CONCEPTS COVERED:
# 1. State Management (Pydantic) -- household finances flow through the graph
# 2. Nodes -- each function does one job (analyze cash flow, debt, savings)
# 3. Parallel Execution -- 3 specialist nodes run at the same time
# 4. Fan-in -- waiting for all 3 analyses before classifying budget health
# 5. Conditional Edges -- routing to stable vs recovery based on health
# 6. Graph Compilation -- turning the graph definition into a runnable app
#
#
# GRAPH STRUCTURE:
#
#   START
#     |
#     +---> analyze_cash_flow ----------+
#     |                                 |
#     +---> analyze_debt_pressure ------+---> classify_budget_health
#     |                                 |         |
#     +---> analyze_savings_readiness --+    (conditional)
#                                          /              \
#                                    stable?          needs_correction?
#                                       |                     |
#                              stable_budget_plan    budget_recovery_plan
#                                       |                     |
# 
#                                      END                   END
#
#
# HOW TO RUN:
#   python budget_health_graph.py
#
# DEPENDENCIES (same as requirements.txt):
#   langgraph, langchain-openai, python-dotenv, pydantic

import sys
import operator
import json
from typing import Annotated, ClassVar, Dict

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END


load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

class budgethealthState(BaseModel):
    income: float = 0.0
    essential_costs: Dict[str, float] = {}
    discretionary_spending: Dict[str, float] = {}
    debts: Dict[str, float] = {}
    current_savings: float = 0.0
    savings_goal: Dict[str, float] = {}
    cash_flow_analysis: str = ""
    debt_pressure_analysis: str = ""
    savings_readiness_analysis: str = ""
    budget_health_classification: str = ""
    health_reason: str = ""
    final_plan: str = ""
    messages: Annotated[list, operator.add] = []

    llm: ClassVar[ChatOpenAI] = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)

    @staticmethod
    def analyze_cash_flow(state: "budgethealthState") -> dict:
        essential_total = sum(state.essential_costs.values())
        discretionary_total = sum(state.discretionary_spending.values())
        surplus = state.income - (essential_total + discretionary_total)
        response = budgethealthState.llm.invoke(
            f"Analyze this household cash flow: income={state.income}, "
            f"essential costs={essential_total}, "
            f"discretionary spending={discretionary_total}, surplus={surplus}."
        )
        return {"messages": [f"Cash Flow Analysis: {response.content}"]}

    @staticmethod
    def analyze_debt_pressure(state: "budgethealthState") -> dict:
        total_debt = sum(state.debts.values())
        debt_to_income_ratio = total_debt / state.income if state.income > 0 else float('inf')
        response = budgethealthState.llm.invoke(
            f"Analyze this household debt pressure: total debt={total_debt}, "
            f"income={state.income}, debt-to-income ratio={debt_to_income_ratio:.2f}."
        )
        return {"messages": [f"Debt Pressure Analysis: {response.content}"]}

    @staticmethod
    def analyze_savings_readiness(state: "budgethealthState") -> dict:
        savings_goal_total = sum(state.savings_goal.values())
        savings_ratio = state.current_savings / savings_goal_total if savings_goal_total > 0 else float('inf')
        response = budgethealthState.llm.invoke(
            f"Analyze this household savings readiness: current savings={state.current_savings}, "
            f"savings goal={savings_goal_total}, savings ratio={savings_ratio:.2f}."
        )
        return {"messages": [f"Savings Readiness Analysis: {response.content}"]}

def classify_budget_health(state: "budgethealthState") -> dict:
    response = state.llm.invoke(
        f"Classify the household budget health based on the following analyses:\n"
        f"Cash Flow Analysis: {state.cash_flow_analysis}\n"
        f"Debt Pressure Analysis: {state.debt_pressure_analysis}\n"
        f"Savings Readiness Analysis: {state.savings_readiness_analysis}\n"
        "Classify as 'STABLE' or 'NEEDS CORRECTION' and provide a reason."
    )
    if "STABLE" in response.content:
        return {"budget_health_classification": "STABLE", "health_reason": response.content}
    else:
        return {"budget_health_classification": "NEEDS CORRECTION", "health_reason": response.content}      

def budget_recovery_plan(state: "budgethealthState") -> dict:
    """Create a recovery plan for a budget that needs correction."""
    response = state.llm.invoke(
        f"Create a practical household budget recovery plan based on this reason: "
        f"{state.health_reason}"
    )
    return {"final_plan": response.content}

def stable_budget_plan(state: "budgethealthState") -> dict:
    """Create a maintenance plan for a stable budget."""
    response = state.llm.invoke(
        f"Create a practical household budget maintenance plan based on this reason: "
        f"{state.health_reason}"
    )
    return {"final_plan": response.content}

graph = StateGraph(budgethealthState)
graph.add_node("analyze_cash_flow", budgethealthState.analyze_cash_flow)
graph.add_node("analyze_debt_pressure", budgethealthState.analyze_debt_pressure)
graph.add_node("analyze_savings_readiness", budgethealthState.analyze_savings_readiness)
graph.add_node("classify_budget_health", classify_budget_health)
graph.add_node("stable_budget_plan", stable_budget_plan)
graph.add_node("budget_recovery_plan", budget_recovery_plan)

graph.add_edge(START, "analyze_cash_flow")
graph.add_edge(START, "analyze_debt_pressure")
graph.add_edge(START, "analyze_savings_readiness")

graph.add_edge("analyze_cash_flow", "classify_budget_health")
graph.add_edge("analyze_debt_pressure", "classify_budget_health")
graph.add_edge("analyze_savings_readiness", "classify_budget_health")


graph.add_conditional_edges(
    "classify_budget_health",
    lambda state: "stable_budget_plan"
    if state.budget_health_classification == "STABLE"
    else "budget_recovery_plan",
)
graph.add_edge("stable_budget_plan", END)
graph.add_edge("budget_recovery_plan", END)    
compiled_graph = graph.compile()

def run_budget_health_analysis(state_data: dict):
    initial_state = budgethealthState(**state_data)
    final_state = budgethealthState(
        **compiled_graph.invoke(initial_state.model_dump())
    )
    print("\nFinal Budget Health Evaluation:")
    print(f"Budget Health Classification: {final_state.budget_health_classification}")
    print(f"Reason: {final_state.health_reason}")
    print(f"Final Plan: {final_state.final_plan}")
    print("\nDetailed Messages from Each Node:")
    for message in final_state.messages:
        print(message)  

    return final_state

if __name__ == "__main__":
    print("Welcome to the Household Financial Health Evaluator!")
    # Example input data for testing    
    example_data = {
        "income": 5000.0,
        "essential_costs": {"rent": 1500.0, "utilities": 300.0, "groceries": 400.0},
        "discretionary_spending": {"entertainment": 200.0, "dining_out": 150.0},
        "debts": {"credit_card": 1000.0, "student_loan": 5000.0},
        "current_savings": 2000.0,
        "savings_goal": {"emergency_fund": 5000.0, "vacation": 2000.0}
    }      
    run_budget_health_analysis(example_data)        

    while True:
        print("\nEnter your household financial data (or type 'exit' to quit):")
        try:
            income = float(input("Monthly Income: "))
            essential_costs = json.loads(input("Essential Costs (JSON format): "))
            discretionary_spending = json.loads(input("Discretionary Spending (JSON format): "))
            debts = json.loads(input("Debts (JSON format): "))
            current_savings = float(input("Current Savings: "))
            savings_goal = json.loads(input("Savings Goal (JSON format): "))

            user_data = {
                "income": income,
                "essential_costs": essential_costs,
                "discretionary_spending": discretionary_spending,
                "debts": debts,
                "current_savings": current_savings,
                "savings_goal": savings_goal
            }

            run_budget_health_analysis(user_data)

        except (ValueError, json.JSONDecodeError) as error:
            print(f"Invalid input: {error}")
            continue

        if not input:
            continue
            print("Thank you for using the Household Financial Health Evaluator. Goodbye!")
            break