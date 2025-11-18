import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


###################################################################
# Custom Local LLM Wrapper
###################################################################
class HFLocalLLM:
    def __init__(self, pipe):
        self.pipe = pipe

    def run(self, prompt):
        out = self.pipe(prompt, max_new_tokens=150)
        return out[0]["generated_text"]


###################################################################
# Load Local Model Only Once
###################################################################
def load_local_llm():
    print("🔄 Loading distilgpt2...")

    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        temperature=0.7,
        max_new_tokens=150,
        pad_token_id=tokenizer.eos_token_id
    )

    print("✅ Local LLM ready!")
    return HFLocalLLM(pipe)


###################################################################
# Load Metrics
###################################################################
def load_latest_metrics(symbol):
    path = f"models/{symbol}_metrics.json"
    if os.path.exists(path):
        return json.load(open(path))
    return None


###################################################################
# Run "Agent" Without CrewAI LLM
###################################################################
def run_query_agent(symbol, sentiment_data=None):
    metrics = load_latest_metrics(symbol)
    if metrics is None:
        return f"No metrics found for {symbol}"

    llm = load_local_llm()  # our ONLY LLM

    # ---- prepare summary ----
    summary = f"""
    Stock: {symbol}
    RMSE: {metrics['rmse']}
    MAE: {metrics['mae']}
    MAPE: {metrics['mape']}
    Trained On: {metrics['trained_on']}
    Data Points: {metrics['data_points']}
    """

    if sentiment_data:
        summary += "\nSentiment Data:\n"
        for s in sentiment_data:
            summary += f"- {s['headline']} → {s['label']} ({s['score']})\n"

    # ---- we manually create the prompt ----
    prompt = f"""
    You are a stock market analyst.

    Analyze the following stock data:

    {summary}

    Give a simple explanation combining model performance and sentiment analysis.
    """

    print("\n🤖 Running Local Model...\n")
    return llm.run(prompt)


###################################################################
# MAIN
###################################################################
if __name__ == "__main__":
    result = run_query_agent("META")
    print("\n===== FINAL OUTPUT =====\n")
    print(result)

