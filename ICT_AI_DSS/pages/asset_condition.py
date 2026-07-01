"""Asset Condition page — Explainable AI Asset Health Assessment."""

import streamlit as st
import pandas as pd
from ui_components import render_section_header
from model_manager import generate_deep_analysis, get_selected_model


def render(inv, repair, selected_model=None):
    from backend import compute_asset_health_analysis

    render_section_header('🧠 AI Asset Health Assessment')

    with st.spinner('AI analyzing repair history and computing health scores...'):
        analysis_df, kpi = compute_asset_health_analysis(inv, repair)

    st.markdown("""
    <div class="ai-summary-box">
        <strong>🤖 AI Assessment Complete</strong> — The AI evaluated <strong>{total}</strong> assets
        using repair frequency, Help Desk remarks, shelf life, and equipment age.
        Each asset received an explainable health score and risk classification.
    </div>
    """.format(total=kpi['total']), unsafe_allow_html=True)

    st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">🟢</div>
            <div class="kpi-value">{kpi['healthy']}</div>
            <div class="kpi-label">Healthy Assets</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">🟡</div>
            <div class="kpi-value">{kpi['moderate']}</div>
            <div class="kpi-label">Moderate</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">🟠</div>
            <div class="kpi-value">{kpi['high_risk']}</div>
            <div class="kpi-label">High Risk</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">🔴</div>
            <div class="kpi-value">{kpi['critical']}</div>
            <div class="kpi-label">Critical</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    render_section_header('📋 Asset Health Assessment Results')

    col1, col2, col3 = st.columns(3)
    with col1:
        search_term = st.text_input('🔍 Search', '', key='ac_search', placeholder='Search property no, employee, equipment...')
    with col2:
        risk_filter = st.multiselect('Risk Level', ['🟢 Healthy', '🟡 Moderate', '🟠 High Risk', '🔴 Critical'], default=[], key='ac_risk')
    with col3:
        equip_filter = st.multiselect('Equipment Type', sorted(analysis_df['equipment_type'].dropna().unique()), default=[], key='ac_equip')

    filtered = analysis_df.copy()
    if risk_filter:
        filtered = filtered[filtered['risk_label'].isin(risk_filter)]
    if equip_filter:
        filtered = filtered[filtered['equipment_type'].isin(equip_filter)]
    if search_term:
        mask = filtered.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
        filtered = filtered[mask]

    st.caption(f'{len(filtered)} of {kpi["total"]} assets')

    for idx, (_, row) in enumerate(filtered.iterrows()):
        issues_str = ', '.join(row['issues_found']) if row['issues_found'] else 'None'
        remarks_list = row.get('remarks_history', [])
        valid_remarks = [r for r in remarks_list if not pd.isna(r) and str(r).strip() not in ('', 'No remarks recorded')]

        with st.expander(
            f"**{row['propertyNumber']}** — {row['equipment_type']} — "
            f"{row['risk_label']} (Score: {row['health_score']:.0f}/100)"
        ):
            info_cols = st.columns([1, 1, 1, 1, 1])
            with info_cols[0]:
                st.markdown(f"**Property No**<br>{row['propertyNumber']}", unsafe_allow_html=True)
            with info_cols[1]:
                st.markdown(f"**Employee**<br>{row['employee']}", unsafe_allow_html=True)
            with info_cols[2]:
                st.markdown(f"**Office**<br>{row['office']}", unsafe_allow_html=True)
            with info_cols[3]:
                st.markdown(f"**Equipment**<br>{row['equipment_type']}", unsafe_allow_html=True)
            with info_cols[4]:
                st.markdown(f"**Actual User**<br>{row['actual_user']}", unsafe_allow_html=True)

            st.markdown("---")

            detail_cols = st.columns([1, 1, 1, 1])
            with detail_cols[0]:
                st.markdown(f"**Total Repairs:** {row['repairs_count']}")
            with detail_cols[1]:
                st.markdown(f"**Current Status:** {row['current_status']}")
            with detail_cols[2]:
                st.markdown(f"**AI Health Score:** {row['health_score']:.0f}/100")
            with detail_cols[3]:
                st.markdown(f"**Risk Level:** {row['risk_label']}")

            if row['issues_found']:
                st.markdown(f"**🔄 Recurring Issues:** {issues_str}")

            with st.container():
                st.markdown("**🧠 AI Explanation**")
                st.info(row['explanation'])

            with st.container():
                st.markdown("**📝 AI Summary**")
                st.success(row['ai_summary'])

            with st.container():
                st.markdown("**💡 Recommendation**")
                st.warning(row['recommendation'])

            # LLM Deep Analysis
            model = selected_model or get_selected_model()
            if model:
                deep_key = f'deep_{row["propertyNumber"]}_{idx}'
                with st.container():
                    st.markdown("**🤖 Deep Analysis (LLM)**")
                    if st.button('Generate Deep Analysis', key=deep_key, use_container_width=True):
                        with st.spinner(f'Analyzing with {model}...'):
                            deep_result = generate_deep_analysis(model, row)
                        if deep_result:
                            # Parse sections for structured display
                            sections = {'## Asset Overview': '', '## Deep Analysis': '', '## Recommendation': '', '## Why': ''}
                            current_section = None
                            for line in deep_result.split('\n'):
                                stripped = line.strip()
                                if stripped.startswith('## '):
                                    current_section = stripped
                                    if current_section not in sections:
                                        current_section = None
                                elif current_section and current_section in sections:
                                    sections[current_section] += line + '\n'

                            if any(sections.values()):
                                tabs = st.tabs(['Overview', 'Deep Analysis', 'Recommendation', 'Why'])
                                section_keys = ['## Asset Overview', '## Deep Analysis', '## Recommendation', '## Why']
                                for tab, sk in zip(tabs, section_keys):
                                    with tab:
                                        st.markdown(sections.get(sk, '*No content*'))
                            else:
                                st.markdown(f'<div class="ai-summary-box">{deep_result}</div>', unsafe_allow_html=True)
                            st.caption(f'Analysis by **{model}**')
                        else:
                            st.warning(f'Could not reach Ollama model **{model}**. Ensure `ollama serve` is running.')
                    else:
                        st.caption(f'Click to analyze this asset using **{model}**')

            if valid_remarks:
                with st.container():
                    st.markdown("**📜 Repair History**")
                    for i, remark in enumerate(valid_remarks):
                        date_str = ''
                        if i < len(row.get('date_recorded_list', [])):
                            d = row['date_recorded_list'][i]
                            date_str = str(d) if not pd.isna(d) else ''
                        staff_str = ''
                        if i < len(row.get('action_staff_list', [])):
                            s = row['action_staff_list'][i]
                            staff_str = f" — *{s}*" if not pd.isna(s) else ''
                        label = f"#{i+1}"
                        if date_str:
                            label += f" ({date_str})"
                        st.markdown(f"<details><summary><strong>{label}</strong>{staff_str}</summary><blockquote>{remark}</blockquote></details>", unsafe_allow_html=True)

    if filtered.empty:
        st.info('No assets match the current filters.')
