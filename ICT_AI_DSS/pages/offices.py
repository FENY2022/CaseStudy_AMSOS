"""Offices page."""

import streamlit as st
import pandas as pd
from ui_components import render_section_header, to_csv, to_excel


def render(inv, div_short=None):
    render_section_header('🏢 Offices')

    office_stats = inv.groupby('officeDivision').agg(
        Total_Employees=('employeeName', 'nunique'),
        Total_Assets=('id', 'count'),
        Computers=('equipmentType', lambda x: x.str.contains('Desktop|Laptop', case=False, na=False).sum()),
    ).reset_index()
    office_stats.columns = ['Office', 'Total Employees', 'Total Assets', 'Computers']
    office_stats['Coverage'] = (office_stats['Computers'] / office_stats['Total Employees'] * 100).round(1)
    office_stats['Coverage'] = office_stats['Coverage'].apply(lambda x: f'{x}%')
    office_stats = office_stats.sort_values('Total Employees', ascending=False).reset_index(drop=True)

    st.markdown(f'<div style="margin-bottom: 1rem; font-size: 0.85rem; color: #6B7280;">{len(office_stats)} offices</div>', unsafe_allow_html=True)
    st.dataframe(office_stats, use_container_width=True, height=400, hide_index=True)

    if div_short is not None and not div_short.empty:
        render_section_header('📉 Division Computer Shortage')
        st.dataframe(div_short, use_container_width=True, height=350, hide_index=True)

    export_c1, export_c2 = st.columns(2)
    with export_c1:
        st.download_button('📥 Download CSV', to_csv(office_stats), 'office_stats.csv', 'text/csv', use_container_width=True)
    with export_c2:
        st.download_button('📥 Download Excel', to_excel(office_stats), 'office_stats.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', use_container_width=True)
