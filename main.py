"""
main.py
-------
Streamlit web interface for the Multi-Agent Trading Analysis System.

Loads environment variables, accepts user inputs (ticker + news context),
invokes the LangGraph trading_graph, and renders a polished dashboard with:
  • Live agent progress indicators
  • Specialist analyst report cards
  • Final recommendation badge with confidence meter
  • Raw JSON inspection panel
"""

from __future__ import annotations

import json
import os
import time

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# ── Load .env before importing any LangChain / NVIDIA modules ─────────────────
load_dotenv()

# ── Optional LangSmith tracing — uncomment to enable ─────────────────────────
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGCHAIN_API_KEY", "")
# os.environ["LANGCHAIN_PROJECT"] = "multi-agent-trading-analysis"

from graph import trading_graph  # noqa: E402  (after load_dotenv)

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Multi-Agent Trading Analysis System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Multi-Agent Trading Analysis System — Powered by LangGraph + NVIDIA NIM",
    },
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — dark premium theme
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
  /* ── Google Font ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* ── Root background ── */
  .stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1224 50%, #0a0e1a 100%);
    color: #e2e8f0;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: rgba(13, 18, 36, 0.95) !important;
    border-right: 1px solid rgba(99, 179, 237, 0.15);
  }

  /* ── Header strip ── */
  .main-header {
    background: linear-gradient(90deg, rgba(99,179,237,0.12) 0%, rgba(129,140,248,0.12) 50%, rgba(236,72,153,0.08) 100%);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 28px;
  }
  .main-header h1 {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #63b3ed, #818cf8, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
  }
  .main-header p {
    color: #94a3b8;
    font-size: 0.95rem;
    margin: 6px 0 0 0;
  }

  /* ── Metric card ── */
  .metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
    transition: transform 0.2s ease, border-color 0.2s ease;
  }
  .metric-card:hover {
    transform: translateY(-2px);
    border-color: rgba(99,179,237,0.35);
  }
  .metric-card .label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
    margin-bottom: 6px;
  }
  .metric-card .value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #e2e8f0;
  }

  /* ── Analyst report card ── */
  .report-card {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(99,179,237,0.12);
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 18px;
    transition: border-color 0.25s ease;
  }
  .report-card:hover { border-color: rgba(129,140,248,0.35); }
  .report-card .card-title {
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .report-card .card-body {
    font-size: 0.88rem;
    color: #94a3b8;
    line-height: 1.7;
    white-space: pre-wrap;
  }

  /* ── Recommendation badge ── */
  .rec-badge {
    display: inline-block;
    padding: 10px 32px;
    border-radius: 50px;
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: 0.06em;
    text-align: center;
  }
  .rec-buy  { background: rgba(16,185,129,0.2); color: #10b981; border: 2px solid #10b981; }
  .rec-sell { background: rgba(239,68,68,0.2);  color: #ef4444; border: 2px solid #ef4444; }
  .rec-hold { background: rgba(245,158,11,0.2); color: #f59e0b; border: 2px solid #f59e0b; }

  /* ── Progress step ── */
  .step-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 14px;
    border-radius: 8px;
    margin-bottom: 6px;
    font-size: 0.85rem;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(99,179,237,0.08);
  }
  .step-done  { color: #10b981; }
  .step-run   { color: #f59e0b; }
  .step-wait  { color: #475569; }

  /* ── Confidence bar ── */
  .conf-bar-bg {
    background: rgba(255,255,255,0.06);
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
    margin-top: 6px;
  }
  .conf-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 1s ease;
  }

  /* ── Streamlit input overrides ── */
  .stTextInput > div > div > input,
  .stTextArea > div > div > textarea {
    background: rgba(255,255,255,0.05) !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(99,179,237,0.2) !important;
    border-radius: 10px !important;
  }
  .stButton > button {
    background: linear-gradient(135deg, #63b3ed, #818cf8) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.6rem 1.4rem !important;
    transition: opacity 0.2s ease !important;
  }
  .stButton > button:hover { opacity: 0.88 !important; }

  /* ── Section divider ── */
  .section-divider {
    border: none;
    border-top: 1px solid rgba(99,179,237,0.1);
    margin: 26px 0;
  }

  /* ── Chip / tag ── */
  .chip {
    display: inline-block;
    background: rgba(99,179,237,0.12);
    color: #63b3ed;
    border: 1px solid rgba(99,179,237,0.25);
    border-radius: 999px;
    padding: 2px 12px;
    font-size: 0.75rem;
    font-weight: 600;
  }

  /* hide default streamlit header */
  header[data-testid="stHeader"] { display: none; }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helper rendering functions
# ─────────────────────────────────────────────────────────────────────────────

AGENT_META = {
    "fundamental_analyst": {
        "icon": "🏦",
        "title": "Fundamental Analyst",
        "color": "#63b3ed",
    },
    "technical_analyst": {
        "icon": "📈",
        "title": "Technical Analyst",
        "color": "#818cf8",
    },
    "sentiment_analyst": {
        "icon": "🧠",
        "title": "Sentiment Analyst",
        "color": "#ec4899",
    },
    "risk_manager": {
        "icon": "🛡️",
        "title": "Risk Manager",
        "color": "#f59e0b",
    },
}

REPORT_KEY_MAP = {
    "fundamental_analyst": "fundamental",
    "technical_analyst": "technical",
    "sentiment_analyst": "sentiment",
    "risk_manager": "risk",
}


def render_header():
    st.markdown(
        """
        <div class="main-header">
          <h1>📊 Multi-Agent Trading Analysis System</h1>
          <p>
            Powered by <strong>LangGraph</strong> · <strong>LangChain</strong> ·
            <strong>NVIDIA NIM (step-3.5-flash)</strong> · <strong>Streamlit</strong>
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_agent_pipeline(steps_completed: list[str], current_agent: str | None = None):
    """Render a live agent pipeline progress view in the sidebar."""
    st.sidebar.markdown("### 🔄 Agent Pipeline")
    agents_order = [
        "fundamental_analyst",
        "technical_analyst",
        "sentiment_analyst",
        "risk_manager",
    ]
    for agent in agents_order:
        meta = AGENT_META[agent]
        if agent in steps_completed:
            cls, icon_s = "step-done", "✅"
        elif agent == current_agent:
            cls, icon_s = "step-run", "⏳"
        else:
            cls, icon_s = "step-wait", "○"
        st.sidebar.markdown(
            f'<div class="step-item {cls}">{icon_s} {meta["icon"]} {meta["title"]}</div>',
            unsafe_allow_html=True,
        )


def render_recommendation(final_decision: str, confidence: int, reasoning: str, risk_summary: str):
    """Render the final supervisor decision with styling."""
    rec_class = {
        "Buy": "rec-buy",
        "Sell": "rec-sell",
        "Hold": "rec-hold",
    }.get(final_decision, "rec-hold")

    conf_color = (
        "#10b981" if confidence >= 70 else ("#f59e0b" if confidence >= 45 else "#ef4444")
    )

    st.markdown(
        f"""
        <div style="text-align:center; padding: 32px 0 20px 0;">
          <div class="rec-badge {rec_class}">{final_decision.upper()}</div>
          <div style="margin-top:16px; font-size:0.9rem; color:#64748b; font-weight:600;">
            CONFIDENCE
          </div>
          <div style="font-size:2rem; font-weight:800; color:{conf_color};">
            {confidence}%
          </div>
          <div class="conf-bar-bg" style="max-width:320px; margin: 6px auto;">
            <div class="conf-bar-fill"
                 style="width:{confidence}%; background: linear-gradient(90deg, {conf_color}80, {conf_color});">
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("#### 🧩 Synthesis Reasoning")
        st.markdown(
            f'<div class="report-card"><div class="card-body">{reasoning}</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown("#### 🛡️ Risk Profile")
        st.markdown(
            f'<div class="report-card"><div class="card-body">{risk_summary}</div></div>',
            unsafe_allow_html=True,
        )


def render_metric_row(raw_data: dict, report_type: str):
    """Render 4 key metric chips below a report card."""
    keys_to_show = {
        "fundamental": [
            "fundamental_score", "pe_ratio", "revenue_growth_pct",
            "nvt_ratio", "developer_commits_30d",
        ],
        "technical": [
            "technical_score", "rsi_14", "macd_histogram",
            "technical_bias", "golden_cross_active",
        ],
        "sentiment": [
            "sentiment_label", "compound_sentiment_score",
            "bullish_signals_detected", "bearish_signals_detected",
        ],
        "risk": [
            "risk_level", "annualized_volatility_pct",
            "sharpe_ratio", "var_95_1day_usd", "risk_score",
        ],
    }.get(report_type, [])

    chips_html = ""
    for k in keys_to_show:
        val = raw_data.get(k)
        if val is not None:
            label = k.replace("_", " ").title()
            chips_html += f'<span class="chip">{label}: {val}</span>&nbsp;'
    st.markdown(chips_html, unsafe_allow_html=True)


def render_specialist_reports(analyst_reports: dict):
    """Render all collected specialist report cards."""
    st.markdown("## 🔬 Specialist Analyst Reports")
    agent_order = ["fundamental", "technical", "sentiment", "risk"]
    agent_display = {
        "fundamental": ("🏦", "Fundamental Analyst", "#63b3ed"),
        "technical": ("📈", "Technical Analyst", "#818cf8"),
        "sentiment": ("🧠", "Sentiment Analyst", "#ec4899"),
        "risk": ("🛡️", "Risk Manager", "#f59e0b"),
    }
    for key in agent_order:
        if key not in analyst_reports:
            continue
        entry = analyst_reports[key]
        icon, title, color = agent_display[key]
        report_text = entry.get("report", "No report generated.")
        raw_data = entry.get("raw_data", {})

        st.markdown(
            f"""
            <div class="report-card">
              <div class="card-title" style="color:{color};">
                {icon} {title}
              </div>
              <div class="card-body">{report_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_metric_row(raw_data, key)
        st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — configuration panel
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar():
    st.sidebar.markdown(
        """
        <div style="text-align:center; padding-bottom:12px;">
          <div style="font-size:2rem;">📊</div>
          <div style="font-weight:700; font-size:1.1rem; color:#e2e8f0;">Trading Analysis</div>
          <div style="font-size:0.75rem; color:#64748b;">Multi-Agent System</div>
        </div>
        <hr style="border-color:rgba(99,179,237,0.1); margin-bottom:16px;">
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("### ⚙️ System Info")
    st.sidebar.markdown(
        """
        <div class="metric-card" style="margin-bottom:8px;">
          <div class="label">LLM Model</div>
          <div class="value" style="font-size:0.85rem;">step-3.5-flash</div>
        </div>
        <div class="metric-card" style="margin-bottom:8px;">
          <div class="label">Provider</div>
          <div class="value" style="font-size:0.85rem;">NVIDIA NIM</div>
        </div>
        <div class="metric-card" style="margin-bottom:8px;">
          <div class="label">Orchestrator</div>
          <div class="value" style="font-size:0.85rem;">LangGraph</div>
        </div>
        <div class="metric-card">
          <div class="label">Agents</div>
          <div class="value">5</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("<hr style='border-color:rgba(99,179,237,0.1);'>", unsafe_allow_html=True)
    st.sidebar.markdown("### 🏗️ Architecture")
    st.sidebar.markdown(
        """
        ```
        START
          └─▶ Supervisor
                ├─▶ Fundamental Analyst
                ├─▶ Technical Analyst
                ├─▶ Sentiment Analyst
                └─▶ Risk Manager
                      └─▶ Supervisor
                            └─▶ END
        ```
        """
    )

    st.sidebar.markdown("<hr style='border-color:rgba(99,179,237,0.1);'>", unsafe_allow_html=True)
    st.sidebar.markdown(
        """
        <div style="font-size:0.75rem; color:#475569; text-align:center;">
          Built with ❤️ using LangChain + LangGraph<br>
          NVIDIA NIM · Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main application
# ─────────────────────────────────────────────────────────────────────────────

def main():
    render_header()
    render_sidebar()

    # ── Input form ────────────────────────────────────────────────────────────
    with st.container():
        col_left, col_right = st.columns([1, 2], gap="large")

        with col_left:
            st.markdown("### 🎯 Asset Configuration")
            ticker = st.text_input(
                "Ticker Symbol",
                value="AAPL",
                placeholder="e.g. AAPL, RELIANCE, BTC, SOL",
                help="Enter any stock ticker (e.g. AAPL, MSFT, RELIANCE) or crypto (e.g. BTC, ETH, SOL).",
                key="ticker_input",
            ).strip().upper()

            st.markdown("**Supported Asset Classes**")
            st.markdown(
                '<span class="chip">📈 Equities</span>&nbsp;<span class="chip">🪙 Crypto</span>',
                unsafe_allow_html=True,
            )
            st.markdown(
                "<br><small style='color:#64748b;'>Examples: AAPL · MSFT · RELIANCE · TSLA · BTC · ETH · SOL · BNB</small>",
                unsafe_allow_html=True,
            )

        with col_right:
            st.markdown("### 📰 Market Context & News")
            context = st.text_area(
                "News / Market Context (for Sentiment Analysis)",
                height=130,
                placeholder=(
                    "Paste recent news headlines, analyst commentary, or market notes here.\n\n"
                    "Example: 'AAPL beats Q3 earnings estimates, iPhone sales surge 18% YoY. "
                    "Fed signals rate cut in September, bullish for tech growth stocks. "
                    "Analysts upgrade AAPL to Strong Buy with $230 price target.'"
                ),
                key="context_input",
                help="This text is fed directly to the Sentiment Analyst agent.",
            )

        st.markdown("<br>", unsafe_allow_html=True)
        run_col, _, info_col = st.columns([1, 2, 3])
        with run_col:
            run_button = st.button(
                "🚀 Run Analysis",
                key="run_analysis_btn",
                use_container_width=True,
            )
        with info_col:
            if not os.environ.get("NVIDIA_API_KEY"):
                st.warning("⚠️ NVIDIA_API_KEY not found in environment. Set it in your `.env` file.", icon="⚠️")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Analysis execution ────────────────────────────────────────────────────
    if run_button:
        if not ticker:
            st.error("Please enter a ticker symbol before running the analysis.")
            return

        if not os.environ.get("NVIDIA_API_KEY"):
            st.error(
                "NVIDIA_API_KEY is not set. Please add it to your `.env` file and restart the app.",
                icon="🔑",
            )
            return

        # Build initial state
        initial_state = {
            "messages": [HumanMessage(content=f"Analyse {ticker}. Context: {context}")],
            "ticker": ticker,
            "context": context or "No additional market context provided.",
            "analyst_reports": {},
            "final_decision": None,
            "confidence": None,
            "reasoning": None,
            "risk_summary": None,
            "next": "",
            "steps_completed": [],
            "error": None,
        }

        # ── Live progress UI ──────────────────────────────────────────────────
        st.markdown(f"## 🔍 Analysing: `{ticker}`")

        agent_order = [
            "fundamental_analyst",
            "technical_analyst",
            "sentiment_analyst",
            "risk_manager",
        ]
        step_labels = {
            "fundamental_analyst": "🏦 Fundamental Analysis",
            "technical_analyst": "📈 Technical Analysis",
            "sentiment_analyst": "🧠 Sentiment Analysis",
            "risk_manager": "🛡️ Risk Assessment",
            "supervisor": "🤖 Supervisor Synthesis",
        }

        progress_container = st.empty()
        status_text = st.empty()

        def render_progress(done: list[str], current: str):
            steps_html = ""
            all_steps = agent_order + ["supervisor"]
            for step in all_steps:
                if step in done:
                    icon_s, cls = "✅", "step-done"
                elif step == current:
                    icon_s, cls = "⏳", "step-run"
                else:
                    icon_s, cls = "○", "step-wait"
                label = step_labels.get(step, step)
                steps_html += f'<div class="step-item {cls}">{icon_s} {label}</div>'
            progress_container.markdown(steps_html, unsafe_allow_html=True)

        # Stream through the graph
        try:
            final_state = None
            steps_done: list[str] = []

            with st.spinner(f"Multi-agent analysis running for **{ticker}**..."):
                for event in trading_graph.stream(initial_state, {"recursion_limit": 30}):
                    for node_name, node_state in event.items():
                        steps = node_state.get("steps_completed", [])
                        completed_agents = [
                            s for s in agent_order if s in steps
                        ]
                        current_running = node_name if node_name != "supervisor" else (
                            "supervisor" if len(completed_agents) == 4 else None
                        )
                        render_progress(completed_agents, current_running or "")
                        status_text.markdown(
                            f'<small style="color:#64748b;">Running: <strong>{step_labels.get(node_name, node_name)}</strong></small>',
                            unsafe_allow_html=True,
                        )
                        final_state = node_state

            status_text.empty()
            progress_container.empty()

            if final_state is None:
                st.error("Analysis returned no output. Check your API key and try again.")
                return

            # ── Final Recommendation ─────────────────────────────────────────
            final_decision = final_state.get("final_decision") or "Hold"
            confidence = final_state.get("confidence") or 50
            reasoning = final_state.get("reasoning") or "No reasoning provided."
            risk_summary = final_state.get("risk_summary") or "No risk summary provided."
            analyst_reports = final_state.get("analyst_reports", {})

            st.success(f"✅ Analysis complete for **{ticker}**!")
            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

            # ── Render final recommendation ──────────────────────────────────
            st.markdown("## 🏁 Final Investment Recommendation")
            render_recommendation(final_decision, confidence, reasoning, risk_summary)

            # Update sidebar pipeline
            render_agent_pipeline(agent_order, None)

            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

            # ── Render specialist reports ────────────────────────────────────
            render_specialist_reports(analyst_reports)

            # ── Raw JSON panel ───────────────────────────────────────────────
            with st.expander("🔧 Raw JSON Output — Full State", expanded=False):
                # Serialize messages separately (they're not JSON-serialisable natively)
                safe_state = {
                    **final_state,
                    "messages": [
                        {"type": type(m).__name__, "content": m.content}
                        for m in final_state.get("messages", [])
                    ],
                }
                st.json(safe_state)

        except Exception as exc:
            st.error(f"❌ An error occurred during analysis: {exc}")
            with st.expander("🔍 Error Details"):
                import traceback
                st.code(traceback.format_exc(), language="python")

    else:
        # ── Landing state ─────────────────────────────────────────────────────
        st.markdown(
            """
            <div style="text-align:center; padding: 60px 20px;">
              <div style="font-size:4rem; margin-bottom:16px;">🤖</div>
              <div style="font-size:1.4rem; font-weight:700; color:#e2e8f0; margin-bottom:10px;">
                Ready to Analyse Any Asset
              </div>
              <div style="font-size:0.95rem; color:#64748b; max-width:520px; margin:0 auto; line-height:1.7;">
                Enter a ticker symbol above and optionally paste some market context / news.
                Hit <strong>Run Analysis</strong> to activate the 5-agent LangGraph pipeline.
              </div>
            </div>

            <div style="display:flex; gap:18px; justify-content:center; flex-wrap:wrap; margin-top:32px;">
              <div class="metric-card" style="width:160px;">
                <div class="label">🏦 Fundamental</div>
                <div class="value" style="font-size:0.9rem; color:#63b3ed;">Valuation & Health</div>
              </div>
              <div class="metric-card" style="width:160px;">
                <div class="label">📈 Technical</div>
                <div class="value" style="font-size:0.9rem; color:#818cf8;">Price & Momentum</div>
              </div>
              <div class="metric-card" style="width:160px;">
                <div class="label">🧠 Sentiment</div>
                <div class="value" style="font-size:0.9rem; color:#ec4899;">Market Psychology</div>
              </div>
              <div class="metric-card" style="width:160px;">
                <div class="label">🛡️ Risk</div>
                <div class="value" style="font-size:0.9rem; color:#f59e0b;">VaR & Volatility</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
