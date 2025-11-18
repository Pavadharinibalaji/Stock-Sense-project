# retraining_agent.py

from crewai import Agent, Task, Crew
from monitor import evaluate_model
from retrain import retrain_model
from langchain_huggingface import HuggingFaceEndpoint
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# LLM Setup (Mistral 7B)
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    temperature=0.7,
    max_new_tokens=200
)

def run_retraining_supervisor(symbol: str):
    """
    Runs a retraining decision agent based on drift detection.
    """

    # Step 1: Check drift
    drift = evaluate_model(symbol)

    # Step 2: Supervisor agent
    supervisor = Agent(
        role="Retraining Supervisor",
        goal="Decide whether to retrain the model based on drift detection.",
        backstory=(
            "You monitor the model's performance. "
            "When accuracy drops or drift is detected, "
            "you decide if retraining is necessary."
        ),
        verbose=True,
        llm=llm
    )

    # Step 3: Task for the agent
    task = Task(
        description=(
            f"Model drift status for {symbol}: {drift}. "
            "If drift is True, recommend retraining. "
            "If False, say retraining is not needed."
        ),
        expected_output="Decision: retrain or not.",
        agent=supervisor
    )

    # Step 4: Crew
    crew = Crew(
        agents=[supervisor],
        tasks=[task],
        verbose=True,
        use_litellm=False
    )

    # Step 5: Execute
    result = crew.kickoff()

    # Step 6: If drift is real, retrain
    if drift:
        retrain_model(symbol)

    return result
