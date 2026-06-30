"""Employees page."""

import streamlit as st
import pandas as pd
from ui_components import render_section_header, render_kpi_card, to_csv, to_excel


def render(inv, emp_df):
    render_section_header('👥 Employees')

    total_unique = inv['employeeName'].nunique()
    total_with_pc = inv[inv['equipmentType'].str.contains('Desktop|Laptop', case=False, na=False)]['employeeName'].nunique()
    total_without_pc = total_unique - total_with_pc

    st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric('Total Unique Employees', f'{total_unique:,}')
    with c2:
        st.metric('With Computer', f'{total_with_pc:,}')
    with c3:
        st.metric('Without Computer', f'{total_without_pc:,}')
    with c4:
        st.metric('Coverage Rate', f'{total_with_pc/total_unique*100:.1f}%' if total_unique > 0 else '0%')
    st.markdown('</div>', unsafe_allow_html=True)

    if emp_df is not None and len(emp_df) > 0:
        render_section_header('📋 Employees Requiring Computer')

        display = emp_df[['employee_name', 'office_division', 'status_of_employment', 'nature_of_work', 'priority_score', 'recommendation']].copy() if 'priority_score' in emp_df.columns else emp_df
        display.columns = ['Employee Name', 'Office', 'Status', 'Nature of Work', 'Priority Score', 'Recommendation'] if 'priority_score' in emp_df.columns else emp_df.columns

        page_size = st.selectbox('Show', [10, 25, 50, 100], key='emp_page', index=1)
        total = len(display)
        if page_size < total:
            page = st.number_input('Page', 1, max(1, (total + page_size - 1) // page_size), 1, key='emp_pagenum')
            start = (page - 1) * page_size
            end = min(start + page_size, total)
            st.dataframe(display.iloc[start:end], use_container_width=True, height=400, hide_index=True)
            st.caption(f'Showing {start+1}–{end} of {total}')
        else:
            st.dataframe(display, use_container_width=True, height=400, hide_index=True)

        export_col1, export_col2 = st.columns(2)
        with export_col1:
            st.download_button('📥 Download CSV', to_csv(display), 'employees_needing_computer.csv', 'text/csv', use_container_width=True)
        with export_col2:
            st.download_button('📥 Download Excel', to_excel(display), 'employees_needing_computer.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', use_container_width=True)
    else:
        st.info('All employees appear to have assigned computers.')
