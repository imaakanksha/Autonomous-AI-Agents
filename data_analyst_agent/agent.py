try:
    from google.adk.agents import Agent as GoogleAgent
    from google.adk.tools import google_search, AgentTool as GoogleAgentTool
except ImportError:  # pragma: no cover - fallback for environments without the ADK package
    class GoogleAgent:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            self.name = kwargs.get("name")
            self.model = kwargs.get("model")
            self.instruction = kwargs.get("instruction")
            self.tools = kwargs.get("tools", [])

    def google_search(*args, **kwargs):
        return {"status": "google_search_unavailable", "message": "Google Search tool not available in this environment."}

    class GoogleAgentTool:  # type: ignore[no-redef]
        def __init__(self, agent):
            self.agent = agent

from .tools import (
    analyze_csv,
    calculate_metrics,
    detect_anomalies,
    compare_segments,
    get_data_quality_report,
    generate_insights,
    generate_comprehensive_report,
    identify_key_drivers,
)

Agent = GoogleAgent
AgentTool = GoogleAgentTool

# Sub-agent for web research with specialized instructions
search_agent = Agent(
    name="search_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are an expert market research assistant. Search the web for relevant "
        "data, market trends, industry benchmarks, and competitive insights. Always "
        "provide credible sources and cite them. Focus on recent data (last 12 months "
        "when possible) and include specific numbers and statistics."
    ),
    tools=[google_search],
)

# Root agent — the data analyst with comprehensive capabilities
root_agent = Agent(
    model="gemini-2.5-flash",
    name="data_analyst_agent",
    instruction=(
        "You are an expert data analyst specializing in business intelligence, predictive "
        "insights, and data-driven decision making. You have powerful tools for comprehensive analysis:\n\n"
        "📊 **CORE ANALYTICS TOOLS:**\n"
        "1. **analyze_csv**: Deep-dive into data with filtering, regional/category breakdowns, top performers\n"
        "2. **calculate_metrics**: Statistical analysis (mean, median, std dev, quartiles, growth rates)\n"
        "3. **generate_insights**: Auto-extract key findings, trends, and patterns from data\n\n"
        "🔍 **ADVANCED ANALYSIS TOOLS:**\n"
        "4. **detect_anomalies**: Find outliers using IQR or Z-score methods (identify unusual patterns)\n"
        "5. **compare_segments**: Benchmark performance across categories/regions (identify winners)\n"
        "6. **get_data_quality_report**: Check completeness, duplicates, data types, quality score\n"
        "7. **generate_comprehensive_report**: Produce a single end-to-end report with quality, segments, anomalies, and drivers\n\n"
        "🌐 **EXTERNAL RESEARCH:**\n"
        "8. **search_agent**: Market research, competitor analysis, industry trends\n\n"
        "💡 **BEST PRACTICES:**\n"
        "- Start with generate_insights for quick overview or get_data_quality_report to assess data\n"
        "- Use compare_segments to identify high-performing vs underperforming areas\n"
        "- Apply detect_anomalies to spot unusual patterns or data quality issues\n"
        "- Combine local analysis with search_agent for external validation\n"
        "- Provide specific recommendations backed by data and methodology\n"
        "- Always cite sources and explain analytical approach\n"
        "- Ask clarifying questions if the request needs specificity\n\n"
        "🎯 **ANALYSIS WORKFLOW:**\n"
        "→ Understand the business question\n"
        "→ Assess data quality and structure\n"
        "→ Perform comparative/anomaly analysis\n"
        "→ Extract insights and identify patterns\n"
        "→ Validate with external research if needed\n"
        "→ Deliver actionable recommendations"
    ),
    tools=[
        AgentTool(search_agent),
        analyze_csv,
        calculate_metrics,
        generate_insights,
        generate_comprehensive_report,
        detect_anomalies,
        compare_segments,
        get_data_quality_report,
        identify_key_drivers,
    ],
)