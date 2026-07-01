"""Analytics page with charts and Ollama-powered deep reasoning."""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from model_manager import generate_analytics_deep_reasoning, get_selected_model
from ui_components import render_section_header


CURRENT_YEAR = 2026


def _money(value):
    return f'PHP {int(value):,}'


def _analytics_context(inv, repl_df):
    inv_copy = inv.copy()
    inv_copy['yearAcquired'] = pd.to_numeric(inv_copy.get('yearAcquired'), errors='coerce')

    total_assets = len(inv_copy)
    total_employees = inv_copy['employeeName'].nunique()
    total_offices = inv_copy['officeDivision'].nunique()
    computer_mask = inv_copy['equipmentType'].str.contains('Desktop|Laptop', case=False, na=False)
    computers = int(computer_mask.sum())
    employees_with_computer = inv_copy[computer_mask]['employeeName'].nunique()
    employees_without_computer = max(0, total_employees - employees_with_computer)
    coverage_rate = (employees_with_computer / total_employees * 100) if total_employees else 0
    assets_over_5_years = int((inv_copy['yearAcquired'] <= CURRENT_YEAR - 6).sum())
    shelf_life = inv_copy.get('shelfLife', pd.Series(dtype=str))
    beyond_shelf_life = int((shelf_life == 'Beyond 5 year').sum())

    top_equipment = [
        {'equipment': str(equipment), 'count': int(count)}
        for equipment, count in inv_copy['equipmentType'].value_counts().head(8).items()
    ]

    office_rows = []
    for office in sorted(inv_copy['officeDivision'].dropna().unique()):
        office_df = inv_copy[inv_copy['officeDivision'] == office]
        office_employees = office_df['employeeName'].nunique()
        office_with_computer = office_df[
            office_df['equipmentType'].str.contains('Desktop|Laptop', case=False, na=False)
        ]['employeeName'].nunique()
        office_rows.append({
            'office': str(office),
            'employees': int(office_employees),
            'coverage': (office_with_computer / office_employees * 100) if office_employees else 0,
        })

    priority_mix = {}
    estimated_budget = 0
    if repl_df is not None and 'predicted_priority' in repl_df.columns:
        priority_counts = repl_df['predicted_priority'].value_counts()
        priority_mix = {
            priority: int(priority_counts.get(priority, 0))
            for priority in ['Critical', 'High', 'Medium', 'Low']
        }
        estimated_budget = (priority_mix.get('Critical', 0) + priority_mix.get('High', 0)) * 50000

    budget_items = [
        ('Desktop', len(inv_copy[inv_copy['equipmentType'].str.contains('Desktop Computer', case=False, na=False)]) * 50000),
        ('Laptop', len(inv_copy[inv_copy['equipmentType'].str.contains('Laptop', case=False, na=False)]) * 55000),
        ('Printer', len(inv_copy[inv_copy['equipmentType'].str.contains('Printer', case=False, na=False)]) * 15000),
        ('Scanner', len(inv_copy[inv_copy['equipmentType'].str.contains('Scanner', case=False, na=False)]) * 12000),
        ('UPS', len(inv_copy[inv_copy['equipmentType'].str.contains('UPS', case=False, na=False)]) * 5000),
    ]

    return {
        'total_assets': total_assets,
        'total_employees': total_employees,
        'total_offices': total_offices,
        'computers': computers,
        'coverage_rate': coverage_rate,
        'employees_without_computer': employees_without_computer,
        'assets_over_5_years': assets_over_5_years,
        'beyond_shelf_life': beyond_shelf_life,
        'estimated_budget': estimated_budget,
        'top_equipment': top_equipment,
        'low_coverage_offices': sorted(office_rows, key=lambda row: row['coverage'])[:5],
        'priority_mix': priority_mix,
        'budget_breakdown': [
            {'item': item, 'budget': int(budget)}
            for item, budget in budget_items
            if budget > 0
        ],
    }


def _render_llm_sections(llm_result):
    sections = {
        '## Executive Readout': '',
        '## Risk Drivers': '',
        '## Priority Offices': '',
        '## Recommended Actions': '',
        '## Reasoning': '',
    }
    current_section = None
    for line in llm_result.split('\n'):
        stripped = line.strip()
        if stripped.startswith('## '):
            current_section = stripped
            if current_section not in sections:
                current_section = None
        elif current_section and current_section in sections:
            sections[current_section] += line + '\n'

    if any(value.strip() for value in sections.values()):
        tabs = st.tabs(['Readout', 'Risks', 'Offices', 'Actions', 'Reasoning'])
        for tab, key in zip(tabs, sections.keys()):
            with tab:
                content = sections.get(key, '').strip()
                st.markdown(content if content else '*No content*')
    else:
        st.markdown(f'<div class="ai-summary-box">{llm_result}</div>', unsafe_allow_html=True)


def render(inv, repl_df, selected_model=None):
    render_section_header('Analytics')

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        'Inventory Distribution',
        'ICT Coverage by Office',
        'Procurement Budget',
        'Asset Condition',
        'AI Deep Reasoning',
    ])

    with tab1:
        st.markdown('**Inventory Distribution by Equipment Type**')
        equip_counts = inv['equipmentType'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['#166534', '#1A7A3E', '#219653', '#27AE60', '#2ECC71',
                  '#6FCF97', '#A9DFBF', '#D5F5E3', '#E8F8F5', '#F0FFF4']
        bars = ax.barh(equip_counts.index[::-1], equip_counts.values[::-1],
                       color=colors[:len(equip_counts)], edgecolor='white', height=0.7)
        ax.set_xlabel('Count')
        for bar, val in zip(bars, equip_counts.values[::-1]):
            ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                    str(val), va='center', fontweight=600, fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('#F8FAFC')
        fig.patch.set_facecolor('#F8FAFC')
        st.pyplot(fig, use_container_width=True)

    with tab2:
        st.markdown('**ICT Coverage Rate by Office**')
        office_data = []
        for office in inv['officeDivision'].unique():
            total = inv[inv['officeDivision'] == office]['employeeName'].nunique()
            with_pc = inv[
                (inv['officeDivision'] == office) &
                (inv['equipmentType'].str.contains('Desktop|Laptop', case=False, na=False))
            ]['employeeName'].nunique()
            office_data.append({'Office': office, 'Coverage': (with_pc / total * 100) if total > 0 else 0})
        cov_df = pd.DataFrame(office_data).sort_values('Coverage')

        fig, ax = plt.subplots(figsize=(10, 5))
        colors_cov = ['#DC2626' if v < 50 else '#F59E0B' if v < 80 else '#166534' for v in cov_df['Coverage']]
        bars = ax.barh(cov_df['Office'], cov_df['Coverage'], color=colors_cov, edgecolor='white', height=0.7)
        ax.axvline(x=50, color='#DC2626', linestyle='--', alpha=0.3, label='50% Threshold')
        ax.axvline(x=80, color='#166534', linestyle='--', alpha=0.3, label='80% Target')
        ax.set_xlabel('Coverage Rate (%)')
        for bar, val in zip(bars, cov_df['Coverage']):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f'{val:.0f}%', va='center', fontweight=600, fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('#F8FAFC')
        fig.patch.set_facecolor('#F8FAFC')
        ax.legend(fontsize=8, loc='lower right')
        st.pyplot(fig, use_container_width=True)

    with tab3:
        st.markdown('**Estimated Procurement Budget Breakdown**')
        budget_items = ['Desktop', 'Laptop', 'Printer', 'Scanner', 'UPS', 'Other']
        budget_values = [
            len(inv[inv['equipmentType'].str.contains('Desktop Computer', case=False, na=False)]) * 50000,
            len(inv[inv['equipmentType'].str.contains('Laptop', case=False, na=False)]) * 55000,
            len(inv[inv['equipmentType'].str.contains('Printer', case=False, na=False)]) * 15000,
            len(inv[inv['equipmentType'].str.contains('Scanner', case=False, na=False)]) * 12000,
            len(inv[inv['equipmentType'].str.contains('UPS', case=False, na=False)]) * 5000,
            0,
        ]
        budget_df = pd.DataFrame({'Item': budget_items, 'Budget': budget_values}).sort_values('Budget', ascending=True)
        budget_df = budget_df[budget_df['Budget'] > 0]

        fig, ax = plt.subplots(figsize=(10, 5))
        colors_b = ['#166534', '#1A7A3E', '#219653', '#27AE60', '#2ECC71'][:len(budget_df)]
        bars = ax.barh(budget_df['Item'], budget_df['Budget'], color=colors_b, edgecolor='white', height=0.6)
        ax.set_xlabel('Budget (PHP)')
        for bar, val in zip(bars, budget_df['Budget']):
            ax.text(bar.get_width() + max(budget_df['Budget']) * 0.005, bar.get_y() + bar.get_height() / 2,
                    f'PHP {val:,}', va='center', fontweight=600, fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('#F8FAFC')
        fig.patch.set_facecolor('#F8FAFC')
        st.pyplot(fig, use_container_width=True)

    with tab4:
        st.markdown('**Asset Condition Summary**')
        if repl_df is not None and 'predicted_priority' in repl_df.columns:
            priority_counts = repl_df['predicted_priority'].value_counts()
            priority_order = ['Critical', 'High', 'Medium', 'Low']
            priority_counts = priority_counts.reindex(priority_order, fill_value=0)

            fig, ax = plt.subplots(figsize=(10, 5))
            colors_p = {'Critical': '#DC2626', 'High': '#F59E0B', 'Medium': '#2563EB', 'Low': '#166534'}
            bar_colors = [colors_p.get(p, '#9CA3AF') for p in priority_counts.index]
            bars = ax.bar(priority_counts.index, priority_counts.values, color=bar_colors, edgecolor='white', width=0.6)
            ax.set_ylabel('Count')
            for bar, val in zip(bars, priority_counts.values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                        str(val), ha='center', fontweight=600, fontsize=11)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_facecolor('#F8FAFC')
            fig.patch.set_facecolor('#F8FAFC')
            st.pyplot(fig, use_container_width=True)

            detail_cols = ['propertyNumber', 'equipmentType', 'asset_health_score', 'predicted_priority']
            available_cols = [col for col in detail_cols if col in repl_df.columns]
            with st.expander('Condition Distribution Details'):
                st.dataframe(repl_df[available_cols].head(100), use_container_width=True, height=300, hide_index=True)
        else:
            st.info('Run the ML model pipeline to view asset condition data.')

    with tab5:
        st.markdown('**AI Deep Reasoning for Analytics**')
        context = _analytics_context(inv, repl_df)

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric('Computer Coverage', f"{context['coverage_rate']:.1f}%")
        with k2:
            st.metric('Employees Without Computer', f"{context['employees_without_computer']:,}")
        with k3:
            st.metric('Assets Over 5 Years', f"{context['assets_over_5_years']:,}")
        with k4:
            st.metric('Budget Signal', _money(context['estimated_budget']))

        st.markdown("""
        <div class="ai-summary-box">
            This combines the Analytics charts into one AI reasoning prompt so the selected Ollama model can explain
            what the data means, which risks matter most, and what actions should be prioritized.
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown('**Lowest Coverage Offices**')
            office_df = pd.DataFrame(context['low_coverage_offices'])
            if office_df.empty:
                st.info('No office coverage data available.')
            else:
                office_df['coverage'] = office_df['coverage'].map(lambda value: f'{value:.1f}%')
                office_df.columns = ['Office', 'Employees', 'Coverage']
                st.dataframe(office_df, use_container_width=True, height=220, hide_index=True)
        with col2:
            st.markdown('**Asset Priority Mix**')
            priority_df = pd.DataFrame([
                {'Priority': priority, 'Assets': count}
                for priority, count in context['priority_mix'].items()
            ])
            if priority_df.empty:
                st.info('No asset priority data available.')
            else:
                st.dataframe(priority_df, use_container_width=True, height=220, hide_index=True)

        model = selected_model or get_selected_model()
        if model:
            if st.button('Generate Deep Reasoning with Ollama', key='analytics_llm_btn', type='primary', use_container_width=True):
                with st.spinner(f'Analyzing Analytics data with {model}...'):
                    llm_result = generate_analytics_deep_reasoning(model, context)
                if llm_result:
                    _render_llm_sections(llm_result)
                    st.caption(f'Analysis by **{model}**')
                else:
                    st.warning(f'Could not reach Ollama model **{model}**. Ensure `ollama serve` is running.')
            else:
                st.caption(f'Click to generate Analytics-wide reasoning using **{model}**.')
        else:
            st.warning('No Ollama model is selected. Start Ollama and choose a model in the sidebar.')
