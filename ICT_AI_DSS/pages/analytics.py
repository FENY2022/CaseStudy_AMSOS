"""Analytics page — limited to 4 charts."""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ui_components import render_section_header


def render(inv, repl_df, selected_model=None):
    render_section_header('📊 Analytics')

    tab1, tab2, tab3, tab4 = st.tabs([
        '📦 Inventory Distribution', '💻 ICT Coverage by Office',
        '💰 Procurement Budget', '🔧 Asset Condition'
    ])

    # ── CHART 1: Inventory Distribution ──
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
            ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                    str(val), va='center', fontweight=600, fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('#F8FAFC')
        fig.patch.set_facecolor('#F8FAFC')
        st.pyplot(fig, use_container_width=True)

    # ── CHART 2: ICT Coverage by Office ──
    with tab2:
        st.markdown('**ICT Coverage Rate by Office**')
        office_data = []
        for office in inv['officeDivision'].unique():
            total = inv[inv['officeDivision'] == office]['employeeName'].nunique()
            with_pc = inv[(inv['officeDivision'] == office) &
                          (inv['equipmentType'].str.contains('Desktop|Laptop', case=False, na=False))]['employeeName'].nunique()
            office_data.append({'Office': office, 'Coverage': (with_pc / total * 100) if total > 0 else 0})
        cov_df = pd.DataFrame(office_data).sort_values('Coverage')

        fig, ax = plt.subplots(figsize=(10, 5))
        colors_cov = ['#DC2626' if v < 50 else '#F59E0B' if v < 80 else '#166534' for v in cov_df['Coverage']]
        bars = ax.barh(cov_df['Office'], cov_df['Coverage'], color=colors_cov, edgecolor='white', height=0.7)
        ax.axvline(x=50, color='#DC2626', linestyle='--', alpha=0.3, label='50% Threshold')
        ax.axvline(x=80, color='#166534', linestyle='--', alpha=0.3, label='80% Target')
        ax.set_xlabel('Coverage Rate (%)')
        for bar, val in zip(bars, cov_df['Coverage']):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{val:.0f}%', va='center', fontweight=600, fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('#F8FAFC')
        fig.patch.set_facecolor('#F8FAFC')
        ax.legend(fontsize=8, loc='lower right')
        st.pyplot(fig, use_container_width=True)

    # ── CHART 3: Procurement Budget Breakdown ──
    with tab3:
        st.markdown('**Estimated Procurement Budget Breakdown**')
        budget_items = ['Desktop', 'Laptop', 'Printer', 'Scanner', 'UPS', 'Other']
        budget_values = [
            len(inv[inv['equipmentType'].str.contains('Desktop Computer', case=False, na=False)]) * 50000,
            len(inv[inv['equipmentType'].str.contains('Laptop', case=False, na=False)]) * 55000,
            len(inv[inv['equipmentType'].str.contains('Printer', case=False, na=False)]) * 15000,
            len(inv[inv['equipmentType'].str.contains('Scanner', case=False, na=False)]) * 12000,
            len(inv[inv['equipmentType'].str.contains('UPS', case=False, na=False)]) * 5000,
            0
        ]
        budget_df = pd.DataFrame({'Item': budget_items, 'Budget': budget_values}).sort_values('Budget', ascending=True)
        budget_df = budget_df[budget_df['Budget'] > 0]

        fig, ax = plt.subplots(figsize=(10, 5))
        colors_b = ['#166534', '#1A7A3E', '#219653', '#27AE60', '#2ECC71'][:len(budget_df)]
        bars = ax.barh(budget_df['Item'], budget_df['Budget'], color=colors_b, edgecolor='white', height=0.6)
        ax.set_xlabel('Budget (₱)')
        for bar, val in zip(bars, budget_df['Budget']):
            ax.text(bar.get_width() + max(budget_df['Budget'])*0.005, bar.get_y() + bar.get_height()/2,
                    f'₱{val:,}', va='center', fontweight=600, fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('#F8FAFC')
        fig.patch.set_facecolor('#F8FAFC')
        st.pyplot(fig, use_container_width=True)

    # ── CHART 4: Asset Condition Summary ──
    with tab4:
        st.markdown('**Asset Condition Summary**')
        if repl_df is not None:
            priority_counts = repl_df['predicted_priority'].value_counts()
            priority_order = ['Critical', 'High', 'Medium', 'Low']
            priority_counts = priority_counts.reindex(priority_order, fill_value=0)

            fig, ax = plt.subplots(figsize=(10, 5))
            colors_p = {'Critical': '#DC2626', 'High': '#F59E0B', 'Medium': '#2563EB', 'Low': '#166534'}
            bar_colors = [colors_p.get(p, '#9CA3AF') for p in priority_counts.index]
            bars = ax.bar(priority_counts.index, priority_counts.values, color=bar_colors, edgecolor='white', width=0.6)
            ax.set_ylabel('Count')
            for bar, val in zip(bars, priority_counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        str(val), ha='center', fontweight=600, fontsize=11)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_facecolor('#F8FAFC')
            fig.patch.set_facecolor('#F8FAFC')
            st.pyplot(fig, use_container_width=True)

            with st.expander('📋 Condition Distribution Details'):
                st.dataframe(repl_df[['propertyNumber', 'equipmentType', 'asset_health_score', 'predicted_priority']]
                            .head(100), use_container_width=True, height=300, hide_index=True)
        else:
            st.info('Run the ML model pipeline to view asset condition data.')
