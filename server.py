import logging
import pandas as pd

from mcp.server.fastmcp import FastMCP
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from tools.synthetic_data import generate_synthetic_data
from tools.outlier_detection import detect_outliers
from tools.plotter import plot_outliers
from tools.summarizer import summarize_outliers, describe_dataset
from tools.vector_store import log_to_chromadb, query_recent_sessions
from tools.planner import plan_and_recommend
from tools.planner import plan_and_recommend, call_ollama

# Enable logging
logging.basicConfig(level=logging.INFO)
print("Loaded server.py")

mcp = FastMCP("AutonomousAnalyst", stateless_http=True)


@mcp.tool(description="Search past session logs for a topic.")
def search_logs(query: str = "outliers") -> str:
    return "\n\n".join(query_recent_sessions(query))


@mcp.tool(description="Generate synthetic data and save to disk.")
def generate_data() -> str:
    df = generate_synthetic_data()
    df.to_csv("data/generated_data.csv", index=False)
    return f"{len(df)} rows of synthetic data generated."


@mcp.tool(description="Detect outliers using Isolation Forest.")
def analyze_outliers(x_col: str = "feature_1", y_col: str = "feature_2") -> str:
    df = pd.read_csv("data/generated_data.csv")
    df = detect_outliers(df, [x_col, y_col])
    df.to_csv("data/generated_data.csv", index=False)
    return f"Outlier analysis complete. Columns: {x_col}, {y_col}."


@mcp.tool(description="Plot inliers and outliers and save plot.")
def plot_results(x_col: str = "feature_1", y_col: str = "feature_2") -> str:
    df = pd.read_csv("data/generated_data.csv")
    path = plot_outliers(df, x_col, y_col)
    return f"Plot saved to {path}"


@mcp.tool(description="Summarize outlier results using LLM.")
def summarize_results() -> str:
    df = pd.read_csv("data/generated_data.csv")
    return summarize_outliers(df)


@mcp.tool(description="Summarize descriptive stats using LLM.")
def summarize_data_stats() -> str:
    df = pd.read_csv("data/generated_data.csv")
    return describe_dataset(df)


@mcp.tool(description="Store summary to vector store.")
def log_results_to_vector_store() -> str:
    df = pd.read_csv("data/generated_data.csv")
    return log_to_chromadb(df)


@mcp.tool(description="Agentic recommendation based on full dataset analysis.")
def autonomous_plan() -> str:
    df = pd.read_csv("data/generated_data.csv")
    summary1 = summarize_outliers(df)
    summary2 = describe_dataset(df)
    recommendation = call_ollama(
        f"You are an AI analyst. Given the data summaries below, decide on next steps.\n\n"
        f"--- Outlier Summary ---\n{summary1}\n\n"
        f"--- Stats Summary ---\n{summary2}\n\n"
        f"What should be done next?"
    )
    log_to_chromadb(df)

    return (
        f"📊 Outlier Summary:\n{summary1}\n\n"
        f"📈 Dataset Overview:\n{summary2}\n\n"
        f"🤖 Recommendation:\n{recommendation}"
    )

@mcp.tool(description="Automatically invoke tools based on LLM reasoning.")
async def autonomous_pipeline(user_goal: str) -> str:
    """
    Uses LLM to determine a sequence of tool invocations to meet a goal.
    """
    available_tools = [tool.name for tool in mcp.get_tools()]
    prompt = (
        f"You are an autonomous analyst. Your available tools are:\n"
        f"{', '.join(available_tools)}\n\n"
        f"The user goal is: \"{user_goal}\"\n"
        f"List the tools you would use to complete this goal, one per line, in order."
    )

    plan = generate_response(prompt).strip().splitlines()
    results = []

    async with streamablehttp_client("http://localhost:8001/mcp") as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for tool_name in plan:
                try:
                    response = await session.call_tool(tool_name.strip(), arguments={})
                    results.append(f"✅ {tool_name} → {response.content[0].text}")
                except Exception as e:
                    results.append(f"❌ Failed to call '{tool_name}': {e}")

    return "\n".join(results)


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8001)
