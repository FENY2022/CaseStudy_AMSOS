"""
ICT-AMSOS: AI-Powered ICT Asset Management and Service Optimization System
Redesigned UI — Executive Dashboard Interface
"""

import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

st.set_page_config(
    page_title='ICT-AMSOS | ICT Asset Management and Service Optimization System',
    page_icon='🏛️',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ── Import modules ──
from backend import (
    load_data, load_models,
    build_employee_equipment_map, compute_office_equipment_counts,
    policy_full_analysis, compute_priority_scores, optimize_budget,
    compute_enhanced_priority_scores, generate_ai_summary_text,
    BASE_DIR, DATA_DIR, MODEL_DIR, OUTPUT_DIR,
)
from ui_components import inject_custom_css, render_header, render_table, to_csv, to_excel
from model_manager import render_model_selector, get_selected_model, get_model_names


# ── Load data with caching ──
@st.cache_resource
def get_models():
    return load_models()

@st.cache_data(ttl=3600)
def get_data():
    return load_data()

with st.spinner('Loading system data...'):
    try:
        inv, repair, div, repl_df, emp_df, div_short, proc = get_data()
        # Also attempt to load models
        try:
            clf, reg, encoder, feature_cols = get_models()
        except Exception:
            clf = reg = encoder = feature_cols = None
        # Fetch ollama models for selection
        try:
            get_model_names()
        except Exception:
            pass
    except Exception as e:
        st.error(f'Failed to load data: {e}')
        st.stop()


# ── Inject custom CSS ──
inject_custom_css()

# ── Initialize session state ──
if 'proc_step' not in st.session_state:
    st.session_state.proc_step = 1
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# =====================================================================
# TOP HEADER
# =====================================================================
render_header()

# =====================================================================
# SIDEBAR NAVIGATION
# =====================================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">AI</div>
        <div>
            <div class="sidebar-brand-text">ICT-AMSOS</div>
            <div class="sidebar-brand-sub">Decision Support</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    nav_items = [
        '🏠 Dashboard',
        '🧠 Asset Condition',
        '📦 ICT Inventory',
        '👥 Employees',
        '🏢 Offices',
        '💰 Procurement',
        '📊 Analytics',
        '⚙ Settings',
    ]

    selected = st.radio('Navigation', nav_items, label_visibility='collapsed', key='nav')

    st.markdown('<hr style="margin: 1rem 0;">', unsafe_allow_html=True)

    # Model selector
    st.sidebar.markdown('<div style="font-size:0.75rem;color:#6B7280;margin-bottom:0.25rem;">AI Analysis Model</div>', unsafe_allow_html=True)
    render_model_selector()

    st.markdown('<hr style="margin: 1rem 0;">', unsafe_allow_html=True)

    # Summary stats in sidebar
    st.markdown(f"""
    <div style="font-size: 0.75rem; color: #6B7280; padding: 0 0.5rem;">
        <div>Assets: <strong>{len(inv):,}</strong></div>
        <div>Employees: <strong>{inv['employeeName'].nunique():,}</strong></div>
        <div>Offices: <strong>{inv['officeDivision'].nunique()}</strong></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-footer">ICT-AMSOS v2.0</div>', unsafe_allow_html=True)


# =====================================================================
# PAGE ROUTING
# =====================================================================

selected_model = get_selected_model()

if selected == '🏠 Dashboard':
    from pages.dashboard import render as render_dashboard
    render_dashboard(inv, repair, repl_df, emp_df, div_short, selected_model)

elif selected == '🧠 Asset Condition':
    from pages.asset_condition import render as render_asset_condition
    render_asset_condition(inv, repair, selected_model)

elif selected == '📦 ICT Inventory':
    from pages.inventory import render as render_inventory
    render_inventory(inv, selected_model)

elif selected == '👥 Employees':
    from pages.employees import render as render_employees
    render_employees(inv, emp_df, selected_model)

elif selected == '🏢 Offices':
    from pages.offices import render as render_offices
    render_offices(inv, div_short, selected_model)

elif selected == '💰 Procurement':
    from pages.procurement import render as render_procurement
    render_procurement(inv, selected_model)

elif selected == '📊 Analytics':
    from pages.analytics import render as render_analytics
    render_analytics(inv, repl_df, selected_model)

elif selected == '⚙ Settings':
    from pages.settings import render as render_settings
    render_settings(inv, selected_model)
