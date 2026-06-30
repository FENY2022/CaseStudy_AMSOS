"""Dashboard page — KPI cards only."""

import streamlit as st
import pandas as pd
import numpy as np
from ui_components import render_section_header, render_kpi_card


def render(inv, repair, repl_df, emp_df, div_short):
    render_section_header('🏠 Executive Dashboard')

    inv['yearAcquired'] = pd.to_numeric(inv['yearAcquired'], errors='coerce')
    CURRENT_YEAR = 2026

    total_assets = len(inv)
    total_computers = len(inv[inv['equipmentType'].str.contains('Desktop|Laptop', case=False, na=False)])
    total_employees = inv['employeeName'].nunique()
    total_offices = inv['officeDivision'].nunique()

    beyond_5yr = len(inv[inv['yearAcquired'] <= CURRENT_YEAR - 6])
    beyond_life = len(inv[inv['shelfLife'] == 'Beyond 5 year'])
    total_repairs = len(repair)
    emps_without_pc = len(emp_df) if emp_df is not None else 0

    total_budget_est = 0
    if repl_df is not None:
        critical = len(repl_df[repl_df['predicted_priority'] == 'Critical'])
        high = len(repl_df[repl_df['predicted_priority'] == 'High'])
        total_budget_est = (critical + high) * 50000

    st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card('💻', f'{total_assets:,}', 'Total ICT Assets', f'{total_computers:,} computers')
    with c2:
        render_kpi_card('👥', f'{total_employees:,}', 'Total Employees', f'{total_offices} offices')
    with c3:
        render_kpi_card('🏢', f'{total_offices}', 'Total Offices', f'{emps_without_pc} need computers')
    with c4:
        render_kpi_card('💰', f'₱{total_budget_est:,}', 'Est. Procurement Budget', 'Critical + High assets')

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        render_kpi_card('⚠️', f'{beyond_5yr:,}', 'Assets Over 5 Years', f'{beyond_5yr/total_assets*100:.0f}% of total' if total_assets > 0 else '')
    with c6:
        render_kpi_card('🔄', f'{beyond_life:,}', 'Beyond Useful Life', 'Needs replacement')
    with c7:
        render_kpi_card('🔧', f'{total_repairs:,}', 'Repair Requests', 'Pending maintenance')
    with c8:
        render_kpi_card('🖥️', f'{emps_without_pc:,}', 'Employees Without Computer', 'Needs procurement')

    st.markdown('</div>', unsafe_allow_html=True)
