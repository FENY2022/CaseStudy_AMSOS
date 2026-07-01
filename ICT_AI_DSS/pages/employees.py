"""Employees page."""

import streamlit as st
import pandas as pd
from ui_components import render_section_header, render_kpi_card, to_csv, to_excel
from model_manager import generate_employee_recommendation, get_selected_model


def render(inv, emp_df, selected_model=None):
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

        # LLM Employee Recommendation
        model = selected_model or get_selected_model()
        if model:
            st.divider()
            render_section_header('🤖 AI Employee Recommendation')
            st.markdown(f'Select an employee to generate a personalized AI recommendation using **{model}**.')
            emp_names = [''] + list(emp_df['employee_name'].unique())
            selected_emp = st.selectbox('Choose Employee', emp_names, key='emp_llm_select')
            if selected_emp:
                emp_row = emp_df[emp_df['employee_name'] == selected_emp].iloc[0]
                emp_data = {
                    'employee_name': emp_row.get('employee_name', ''),
                    'office_division': emp_row.get('office_division', ''),
                    'status_of_employment': emp_row.get('status_of_employment', ''),
                    'nature_of_work': emp_row.get('nature_of_work', ''),
                    'priority_score': emp_row.get('priority_score', 0),
                    'recommendation': emp_row.get('recommendation', ''),
                }

                if st.button('Generate Personalized Recommendation', key='emp_llm_btn', type='primary', use_container_width=True):
                    with st.spinner(f'Generating recommendation with {model}...'):
                        llm_result = generate_employee_recommendation(model, emp_data)
                    if llm_result:
                        sections = {'## Analysis': '', '## Recommendation': '', '## Why': ''}
                        current_section = None
                        for line in llm_result.split('\n'):
                            stripped = line.strip()
                            if stripped.startswith('## '):
                                current_section = stripped
                                if current_section not in sections:
                                    current_section = None
                            elif current_section and current_section in sections:
                                sections[current_section] += line + '\n'
                        if any(sections.values()):
                            tabs = st.tabs(['Analysis', 'Recommendation', 'Why'])
                            section_keys = ['## Analysis', '## Recommendation', '## Why']
                            for tab, sk in zip(tabs, section_keys):
                                with tab:
                                    st.markdown(sections.get(sk, '*No content*'))
                        else:
                            st.markdown(f'<div class="ai-summary-box">{llm_result}</div>', unsafe_allow_html=True)
                        st.caption(f'Recommendation by **{model}**')
                    else:
                        st.warning(f'Could not reach Ollama model **{model}**. Ensure `ollama serve` is running.')
    else:
        st.info('All employees appear to have assigned computers.')
