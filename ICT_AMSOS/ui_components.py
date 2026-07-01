"""
UI Components — CSS, cards, tables, header, sidebar, exports.
"""

import streamlit as st
import pandas as pd
import base64
import importlib.util
from io import BytesIO


def inject_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

        .stApp {
            background-color: #F8FAFC;
        }

        /* Remove default padding/margins */
        .main > .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
            padding-left: 1.5rem;
            padding-right: 1.5rem;
            max-width: 1400px;
        }

        /* ── TOP HEADER ── */
        .top-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.75rem 1.5rem;
            background: #FFFFFF;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
            margin-bottom: 1.5rem;
            border: 1px solid #E5E7EB;
        }
        .top-header-left {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        .top-header-logo {
            width: 36px;
            height: 36px;
            background: #166534;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 18px;
            font-weight: 700;
        }
        .top-header-title {
            font-size: 1rem;
            font-weight: 600;
            color: #111827;
        }
        .top-header-subtitle {
            font-size: 0.75rem;
            color: #6B7280;
        }
        .top-header-right {
            display: flex;
            align-items: center;
            gap: 1.25rem;
        }
        .top-header-item {
            display: flex;
            align-items: center;
            gap: 0.4rem;
            font-size: 0.8rem;
            color: #4B5563;
        }
        .top-header-item .label {
            color: #9CA3AF;
        }

        /* ── SIDEBAR ── */
        section[data-testid="stSidebar"] {
            background: #FFFFFF;
            border-right: 1px solid #E5E7EB;
            width: 240px;
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.25rem;
            padding-left: 0.75rem;
            padding-right: 0.75rem;
        }
        .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            padding: 0.5rem 0.75rem;
            margin-bottom: 1.5rem;
        }
        .sidebar-brand-icon {
            width: 32px;
            height: 32px;
            background: #166534;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 16px;
            font-weight: 700;
        }
        .sidebar-brand-text {
            font-weight: 600;
            font-size: 0.9rem;
            color: #111827;
        }
        .sidebar-brand-sub {
            font-size: 0.65rem;
            color: #9CA3AF;
        }

        /* Nav items */
        div[data-testid="stRadio"] > label {
            display: none !important;
        }
        div[data-testid="stRadio"] > div {
            gap: 2px;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label {
            padding: 0.5rem 0.75rem;
            border-radius: 8px;
            font-size: 0.85rem;
            font-weight: 500;
            color: #4B5563;
            transition: all 0.15s ease;
            cursor: pointer;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
            background: #F1F5F9;
            color: #111827;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label[data-selected="true"] {
            background: #F0FDF4;
            color: #166534;
            font-weight: 600;
        }
        .sidebar-footer {
            position: fixed;
            bottom: 1rem;
            left: 0.75rem;
            right: 0.75rem;
            font-size: 0.7rem;
            color: #9CA3AF;
            text-align: center;
            padding: 0.5rem;
        }

        /* ── KPI CARDS ── */
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        .kpi-card {
            background: #FFFFFF;
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
            border: 1px solid #E5E7EB;
            transition: box-shadow 0.2s;
        }
        .kpi-card:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }
        .kpi-icon {
            font-size: 1.25rem;
            margin-bottom: 0.5rem;
        }
        .kpi-value {
            font-size: 1.75rem;
            font-weight: 700;
            color: #111827;
            line-height: 1.2;
            margin-bottom: 0.25rem;
        }
        .kpi-label {
            font-size: 0.78rem;
            color: #6B7280;
            font-weight: 500;
        }
        .kpi-trend {
            font-size: 0.7rem;
            margin-top: 0.35rem;
            color: #166534;
        }
        .kpi-trend.down { color: #DC2626; }
        .kpi-trend.neutral { color: #F59E0B; }

        /* ── SUBHEADER ── */
        .section-header {
            font-size: 1.1rem;
            font-weight: 600;
            color: #111827;
            margin: 1.5rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #166534;
        }

        /* ── BUTTONS ── */
        div.stButton > button {
            border-radius: 8px;
            font-weight: 500;
            font-size: 0.85rem;
            padding: 0.4rem 1rem;
            border: 1px solid #E5E7EB;
            background: #FFFFFF;
            color: #374151;
            transition: all 0.15s;
        }
        div.stButton > button:hover {
            border-color: #166534;
            color: #166534;
            background: #F0FDF4;
        }
        div.stButton > button[kind="primary"] {
            background: #166534;
            color: white;
            border: none;
        }
        div.stButton > button[kind="primary"]:hover {
            background: #14532D;
        }

        /* ── METRIC OVERRIDE ── */
        div[data-testid="stMetric"] {
            background: #FFFFFF;
            border-radius: 12px;
            padding: 1rem 1.25rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
            border: 1px solid #E5E7EB;
        }
        div[data-testid="stMetric"] > div {
            gap: 0.15rem;
        }
        div[data-testid="stMetric"] label {
            font-size: 0.78rem !important;
            color: #6B7280 !important;
            font-weight: 500 !important;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            font-size: 1.75rem !important;
            font-weight: 700 !important;
            color: #111827 !important;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
            font-size: 0.7rem !important;
        }

        /* ── TABLES ── */
        div[data-testid="stDataFrame"] {
            border-radius: 8px;
            border: 1px solid #E5E7EB;
            overflow: hidden;
        }
        div[data-testid="stDataFrame"] thead tr th {
            background: #F8FAFC;
            font-size: 0.75rem;
            font-weight: 600;
            color: #4B5563;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: 0.6rem 0.75rem;
            border-bottom: 1px solid #E5E7EB;
        }
        div[data-testid="stDataFrame"] tbody tr td {
            font-size: 0.8rem;
            padding: 0.5rem 0.75rem;
            color: #374151;
            border-bottom: 1px solid #F3F4F6;
        }
        div[data-testid="stDataFrame"] tbody tr:hover td {
            background: #FAFAFA;
        }

        /* ── CHAT MESSAGES ── */
        div[data-testid="stChatMessage"] {
            background: #FFFFFF;
            border-radius: 12px;
            padding: 1rem;
            border: 1px solid #E5E7EB;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            margin-bottom: 0.5rem;
        }

        /* ── EXPANDER ── */
        div[data-testid="stExpander"] {
            border: 1px solid #E5E7EB;
            border-radius: 8px;
            background: #FFFFFF;
        }
        div[data-testid="stExpander"] summary {
            font-weight: 500;
            font-size: 0.85rem;
            color: #374151;
        }

        /* ── DIVIDER ── */
        hr {
            margin: 1.5rem 0;
            border-color: #E5E7EB;
        }

        /* ── CONTAINER / BORDER ── */
        div[data-testid="stVerticalBlockBorderWrapper"] > div {
            border-radius: 8px;
        }

        /* ── INPUTS ── */
        div[data-testid="stNumberInput"] input, div[data-testid="stTextInput"] input, div[data-testid="stSelectInput"] div[data-baseweb="select"] {
            border-radius: 8px;
            border: 1px solid #E5E7EB;
        }
        div[data-testid="stNumberInput"] input:focus, div[data-testid="stTextInput"] input:focus {
            border-color: #166534;
            box-shadow: 0 0 0 2px rgba(22, 101, 52, 0.1);
        }

        /* ── MULTISELECT ── */
        div[data-testid="stMultiSelect"] div[data-baseweb="select"] {
            border-radius: 8px;
            border: 1px solid #E5E7EB;
        }

        /* ── TABS ── */
        div[data-testid="stTabs"] button {
            border-radius: 8px 8px 0 0;
            font-size: 0.8rem;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            border-bottom: 2px solid #166534;
            color: #166534;
        }

        /* ── PROGRESS ── */
        div[data-testid="stProgress"] > div {
            background: #E5E7EB;
        }

        /* ── WORKFLOW ── */
        .workflow-steps {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 2rem;
            padding: 1rem;
            background: #FFFFFF;
            border-radius: 12px;
            border: 1px solid #E5E7EB;
        }
        .workflow-step {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.35rem 0.75rem;
            border-radius: 20px;
            font-size: 0.78rem;
            font-weight: 500;
            color: #9CA3AF;
            background: #F3F4F6;
        }
        .workflow-step.active {
            background: #166534;
            color: white;
        }
        .workflow-step.completed {
            background: #F0FDF4;
            color: #166534;
        }
        .workflow-arrow {
            color: #D1D5DB;
            font-size: 1rem;
        }

        /* ── PRIORITY BADGES ── */
        .badge-high {
            background: #FEF2F2;
            color: #DC2626;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.7rem;
            font-weight: 600;
        }
        .badge-medium {
            background: #FFFBEB;
            color: #D97706;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.7rem;
            font-weight: 600;
        }
        .badge-low {
            background: #F0FDF4;
            color: #16A34A;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.7rem;
            font-weight: 600;
        }

        /* ── AI SUMMARY ── */
        .ai-summary-box {
            background: #F8FAFC;
            border-left: 4px solid #166534;
            border-radius: 8px;
            padding: 1.25rem 1.5rem;
            margin: 1rem 0;
            font-size: 0.85rem;
            line-height: 1.6;
            color: #374151;
        }

        /* ── STATUS BADGES ── */
        .status-approved {
            background: #F0FDF4;
            color: #16A34A;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 0.7rem;
            font-weight: 600;
        }
        .status-waiting {
            background: #FEF3C7;
            color: #D97706;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 0.7rem;
            font-weight: 600;
        }

        /* ── DARK MODE (placeholder — applied via class) ── */
        .dark-mode .stApp { background: #1F2937; }

        @media (max-width: 768px) {
            .top-header { flex-direction: column; gap: 0.75rem; }
            .top-header-right { flex-wrap: wrap; justify-content: center; }
            .kpi-grid { grid-template-columns: repeat(2, 1fr); }
        }
        @media (max-width: 480px) {
            .kpi-grid { grid-template-columns: 1fr; }
        }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    """Render the top header bar with logo, title, user, date, dark mode toggle."""
    from datetime import date
    today = date.today().strftime('%B %d, %Y')
    dark_mode = st.session_state.get('dark_mode', False)
    col_logo, col_spacer, col_right = st.columns([1, 4, 3])
    with col_logo:
        st.markdown("""
        <div class="top-header-left">
            <div class="top-header-logo">AI</div>
            <div>
                <div class="top-header-title">ICT-AMSOS</div>
                <div class="top-header-subtitle">Asset Management and Service Optimization System</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_right:
        user_col, date_col, toggle_col = st.columns([1, 1.2, 0.8])
        with user_col:
            st.markdown(f"""
            <div class="top-header-item">
                <span>👤</span>
                <span><span class="label">User</span><br>Administrator</span>
            </div>
            """, unsafe_allow_html=True)
        with date_col:
            st.markdown(f"""
            <div class="top-header-item">
                <span>📅</span>
                <span><span class="label">Date</span><br>{today}</span>
            </div>
            """, unsafe_allow_html=True)
        with toggle_col:
            dark_mode = st.toggle('🌙', value=dark_mode, key='dark_mode_toggle', help='Dark Mode')
            if dark_mode != st.session_state.get('dark_mode', False):
                st.session_state.dark_mode = dark_mode
                st.rerun()


def render_kpi_card(icon, value, label, trend=None, trend_down=False, trend_neutral=False):
    cls = 'kpi-trend'
    if trend_down:
        cls += ' down'
    elif trend_neutral:
        cls += ' neutral'
    trend_html = f'<div class="{cls}">{trend}</div>' if trend else ''
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        {trend_html}
    </div>
    """, unsafe_allow_html=True)


def render_section_header(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def to_csv(df):
    return df.to_csv(index=False).encode('utf-8')


def to_excel(df):
    if importlib.util.find_spec('openpyxl') is None:
        st.error('Excel export needs the openpyxl package. Run: pip install openpyxl')
        return b''
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()


def render_export_buttons(df, filename_prefix='data', key_suffix=''):
    """Render CSV, Excel, PDF export buttons for a dataframe."""
    csv_data = to_csv(df)
    csv_name = f'{filename_prefix}.csv'
    st.download_button(
        label='📥 CSV',
        data=csv_data,
        file_name=csv_name,
        mime='text/csv',
        key=f'csv_{key_suffix}',
        use_container_width=True,
    )


def render_table(df, key='table', height=400, column_config=None, hide_index=True):
    """Render a styled table with search, export, and sticky header."""
    if df is None or df.empty:
        st.info('No data available.')
        return

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search_term = st.text_input('🔍 Search', '', key=f'search_{key}', placeholder='Search records...')
    with col2:
        export_choice = st.selectbox('Export', ['None', 'CSV', 'Excel'], key=f'export_{key}')
    with col3:
        page_size = st.selectbox('Show', [10, 25, 50, 100], key=f'page_{key}', index=0)

    if search_term:
        mask = df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
        filtered = df[mask]
    else:
        filtered = df

    total_rows = len(filtered)
    if page_size and page_size < total_rows:
        page_num = st.number_input('Page', min_value=1, max_value=max(1, (total_rows + page_size - 1) // page_size), value=1, key=f'pagenum_{key}')
        start = (page_num - 1) * page_size
        end = min(start + page_size, total_rows)
        display_df = filtered.iloc[start:end]
        st.caption(f'Showing {start+1}–{end} of {total_rows} records')
    else:
        display_df = filtered
        st.caption(f'{total_rows} records')

    st.dataframe(display_df, use_container_width=True, height=height, column_config=column_config, hide_index=hide_index)

    if export_choice == 'CSV':
        st.download_button('📥 Download CSV', to_csv(filtered), f'{key}.csv', 'text/csv', use_container_width=True)
    elif export_choice == 'Excel':
        st.download_button('📥 Download Excel', to_excel(filtered), f'{key}.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', use_container_width=True)
