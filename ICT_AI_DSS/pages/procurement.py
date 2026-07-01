"""Procurement page with separate per-item AI recommendation analyses."""

import streamlit as st
import pandas as pd

from backend import (
    EQUIPMENT_COSTS,
    build_employee_equipment_map,
    compute_office_equipment_counts,
)
from ui_components import render_section_header, render_kpi_card, to_csv, to_excel
from model_manager import generate_procurement_recommendation, get_selected_model


COMPUTER_ITEMS = ('Desktop Computer', 'Laptop Computer')


def _has_desktop(equipment_set):
    return 'Desktop Computer' in equipment_set


def _has_laptop(equipment_set):
    return any(item in equipment_set for item in ('Laptop Computer', 'Laptop Computers'))


def _current_equipment(equipment_set):
    return ', '.join(sorted(equipment_set)) if isinstance(equipment_set, set) and equipment_set else 'None'


def _money(value):
    return f'₱{int(value):,}'


def _computer_candidates(emp_map, item_type):
    rows = []
    for _, row in emp_map.iterrows():
        equipment_set = row['equipment_set']
        if _has_desktop(equipment_set) or _has_laptop(equipment_set):
            continue

        current = _current_equipment(equipment_set)
        if not equipment_set:
            priority_tier = 1
            reason = 'No Desktop or Laptop assigned; no ICT equipment recorded.'
        else:
            priority_tier = 2
            reason = f'No Desktop or Laptop assigned; current equipment is limited to {current}.'

        rows.append({
            'Employee': row['employeeName'],
            'Office': row['officeDivision'],
            'Current Equipment': current,
            'Priority Tier': priority_tier,
            'Reason': reason,
            'Recommended Item': item_type,
            'Cost': EQUIPMENT_COSTS.get(item_type, 0),
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    result = result.sort_values(['Office', 'Priority Tier', 'Employee']).reset_index(drop=True)
    result['Rank'] = result.groupby('Office').cumcount() + 1
    return result[['Rank', 'Employee', 'Office', 'Current Equipment', 'Reason', 'Recommended Item', 'Cost', 'Priority Tier']]


def _computer_office_ranking(emp_df, item_type):
    need_col = 'Employees Needing Desktop' if item_type == 'Desktop Computer' else 'Employees Needing Laptop'
    if emp_df.empty:
        return pd.DataFrame(columns=['Rank', 'Office', need_col, 'Budget', 'Priority'])

    office_df = emp_df.groupby('Office').size().reset_index(name=need_col)
    office_df['Budget'] = office_df[need_col] * EQUIPMENT_COSTS.get(item_type, 0)
    office_df = office_df.sort_values([need_col, 'Budget'], ascending=[False, False]).reset_index(drop=True)
    office_df['Rank'] = range(1, len(office_df) + 1)
    max_need = office_df[need_col].max()
    office_df['Priority'] = office_df[need_col].apply(
        lambda n: 'HIGH' if n >= max_need * 0.5 else ('MEDIUM' if n >= max_need * 0.2 else 'LOW')
    )
    return office_df[['Rank', 'Office', need_col, 'Budget', 'Priority']]


def _printer_office_ranking(emp_map, office_counts):
    employee_counts = emp_map['officeDivision'].value_counts().reset_index()
    employee_counts.columns = ['Office', 'Employees']
    if employee_counts.empty:
        return pd.DataFrame()

    largest_count = int(employee_counts['Employees'].max())
    largest_office = employee_counts.loc[employee_counts['Employees'].idxmax(), 'Office']
    rows = []
    for _, row in employee_counts.iterrows():
        office = row['Office']
        employees = int(row['Employees'])
        existing = int(office_counts.get(office, {}).get('Printer', 0))
        target = int((employees / largest_count) * 7 + 0.5) if largest_count else 0
        target = max(0, min(7, target))
        needed = max(0, target - existing)
        reason = 'Largest office with highest workload.' if office == largest_office else (
            f'Office has {employees:,} employees, proportional to {largest_office} with {largest_count:,} employees.'
        )
        rows.append({
            'Office': office,
            'Employees': employees,
            'Existing Printers': existing,
            'Target Printers': target,
            'Additional Printers Needed': needed,
            'Estimated Cost': needed * EQUIPMENT_COSTS.get('Printer', 0),
            'Reason': reason,
        })

    result = pd.DataFrame(rows)
    result = result.sort_values(['Additional Printers Needed', 'Employees'], ascending=[False, False]).reset_index(drop=True)
    result['Rank'] = range(1, len(result) + 1)
    return result[['Rank', 'Office', 'Employees', 'Existing Printers', 'Target Printers', 'Additional Printers Needed', 'Estimated Cost', 'Reason']]


def _render_money_dataframe(df, money_cols, height=320):
    display = df.copy()
    for col in money_cols:
        if col in display.columns:
            display[col] = display[col].apply(_money)
    st.dataframe(display, use_container_width=True, height=height, hide_index=True)


def _render_item_summary_row(info):
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; padding:0.5rem 1rem; background:#FFFFFF; border:1px solid #E5E7EB; border-radius:8px; margin-bottom:0.35rem;">
        <span style="font-weight:500;">{info['item']}</span>
        <span>{info['units']:,} × {_money(info['unit_cost'])} = <strong style="color:#166534;">{_money(info['total'])}</strong></span>
    </div>
    """, unsafe_allow_html=True)


def _computer_summary(emp_map, item_type):
    candidates = _computer_candidates(emp_map, item_type)
    unit_cost = EQUIPMENT_COSTS.get(item_type, 0)
    units = len(candidates)
    return {'item': item_type, 'units': units, 'unit_cost': unit_cost, 'total': units * unit_cost}


def _printer_summary(emp_map, office_counts):
    printer_df = _printer_office_ranking(emp_map, office_counts)
    units = int(printer_df['Additional Printers Needed'].sum()) if not printer_df.empty else 0
    unit_cost = EQUIPMENT_COSTS.get('Printer', 0)
    return {'item': 'Printer', 'units': units, 'unit_cost': unit_cost, 'total': units * unit_cost}


def _render_computer_analysis(item_type, emp_map):
    is_desktop = item_type == 'Desktop Computer'
    title = '💻 Desktop Computer Procurement' if is_desktop else '💼 Laptop Procurement'
    recommended_label = 'Recommended Desktop Purchases' if is_desktop else 'Recommended Laptop Purchases'

    candidates = _computer_candidates(emp_map, item_type)
    office_rank = _computer_office_ranking(candidates, item_type)

    total_employees = len(emp_map)
    assigned_desktop = int(emp_map['equipment_set'].apply(_has_desktop).sum())
    assigned_laptop = int(emp_map['equipment_set'].apply(_has_laptop).sum())
    without_computer = int(emp_map['equipment_set'].apply(lambda s: not _has_desktop(s) and not _has_laptop(s)).sum())
    recommended = len(candidates)
    unit_cost = EQUIPMENT_COSTS.get(item_type, 0)
    budget = recommended * unit_cost

    render_section_header(title)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric('Total Employees', f'{total_employees:,}')
        st.metric('Employees Without Desktop or Laptop', f'{without_computer:,}')
    with c2:
        st.metric('Employees Already Assigned Desktop', f'{assigned_desktop:,}')
        st.metric(recommended_label, f'{recommended:,}')
    with c3:
        st.metric('Employees Already Assigned Laptop', f'{assigned_laptop:,}')
        st.metric('Estimated Budget', _money(budget))

    explanation = (
        f"The AI detected **{without_computer:,}** employees who do not have either a Desktop or Laptop assigned. "
        f"For **{item_type}**, only those employees are eligible. Employees who already own a Desktop or a Laptop are excluded. "
        f"Mathematical computation: **{recommended:,} × {_money(unit_cost)} = {_money(budget)}**."
    )
    st.markdown(f'<div class="ai-summary-box">{explanation}</div>', unsafe_allow_html=True)

    st.markdown('#### Office Priority Ranking')
    if office_rank.empty:
        st.info(f'No offices need {item_type}.')
    else:
        _render_money_dataframe(office_rank, ['Budget'], height=280)

    st.markdown('#### Employee Priority List')
    if candidates.empty:
        st.info(f'No eligible employees for {item_type}.')
    else:
        display = candidates.drop(columns=['Priority Tier']).copy()
        _render_money_dataframe(display, ['Cost'], height=380)
        export_c1, export_c2 = st.columns(2)
        export_name = item_type.lower().replace(' ', '_')
        with export_c1:
            st.download_button('📥 Download CSV', to_csv(display), f'{export_name}_priority_list.csv', 'text/csv', use_container_width=True, key=f'{export_name}_csv')
        with export_c2:
            st.download_button('📥 Download Excel', to_excel(display), f'{export_name}_priority_list.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', use_container_width=True, key=f'{export_name}_xlsx')

    return {'item': item_type, 'units': recommended, 'unit_cost': unit_cost, 'total': budget}


def _render_printer_analysis(emp_map, office_counts):
    printer_df = _printer_office_ranking(emp_map, office_counts)
    total_units = int(printer_df['Additional Printers Needed'].sum()) if not printer_df.empty else 0
    unit_cost = EQUIPMENT_COSTS.get('Printer', 0)
    total_budget = total_units * unit_cost
    largest = printer_df.sort_values('Employees', ascending=False).iloc[0] if not printer_df.empty else None

    render_section_header('🖨 Printer Procurement')
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric('Reference Office', largest['Office'] if largest is not None else 'None')
    with c2:
        st.metric('Largest Office Employees', f"{int(largest['Employees']):,}" if largest is not None else '0')
    with c3:
        st.metric('Recommended Printers', f'{total_units:,}')
    with c4:
        st.metric('Estimated Budget', _money(total_budget))

    if largest is not None:
        explanation = (
            f"The AI allocates printers based on office size rather than employee assignment. "
            f"**{largest['Office']}** has the largest workforce with **{int(largest['Employees']):,}** unique employees, "
            f"so it receives the maximum target allocation of **7** shared printers. "
            f"Formula: **ROUND(Office Employees / {int(largest['Employees']):,} × 7) - Existing Printers**. "
            f"Mathematical computation: **{total_units:,} × {_money(unit_cost)} = {_money(total_budget)}**."
        )
        st.markdown(f'<div class="ai-summary-box">{explanation}</div>', unsafe_allow_html=True)

    st.markdown('#### Office Priority Ranking')
    if printer_df.empty:
        st.info('No printer recommendation available.')
    else:
        _render_money_dataframe(printer_df, ['Estimated Cost'], height=360)

    return {'item': 'Printer', 'units': total_units, 'unit_cost': unit_cost, 'total': total_budget}


def _render_step_indicator(step):
    steps = ['Select Procurement Items', 'Enter Budget', 'Generate AI Recommendation']
    icons = ['📦', '💵', '🤖']
    step_html = '<div class="workflow-steps">'
    for i, (label, icon) in enumerate(zip(steps, icons), 1):
        cls = 'active' if i == step else ('completed' if i < step else '')
        step_html += f'<div class="workflow-step {cls}">{icon} {label}</div>'
        if i < len(steps):
            step_html += '<span class="workflow-arrow">→</span>'
    step_html += '</div>'
    st.markdown(step_html, unsafe_allow_html=True)


def render(inv, selected_model=None):
    render_section_header('💰 Procurement')
    step = st.session_state.get('proc_step', 1)
    _render_step_indicator(step)

    if 'proc_items' not in st.session_state:
        st.session_state.proc_items = ['Desktop Computer', 'Laptop Computer']
    if 'proc_budget' not in st.session_state:
        st.session_state.proc_budget = 1675000

    if step == 1:
        st.markdown('### Step 1: Select Procurement Items')
        st.markdown('<div style="color: #6B7280; font-size: 0.85rem; margin-bottom: 1rem;">Choose the ICT equipment to procure:</div>', unsafe_allow_html=True)

        default_items = st.session_state.proc_items
        col1, col2, col3 = st.columns(3)
        selected = []
        item_labels = {
            'Desktop Computer': ('💻', 'Desktop Computer'),
            'Laptop Computer': ('💼', 'Laptop Computer'),
            'Printer': ('🖨', 'Printer'),
            'CCTV': ('📹', 'CCTV'),
            'Router': ('📡', 'Router'),
            'Switch': ('🔀', 'Switch'),
            'Scanner': ('📄', 'Scanner'),
            'Monitor': ('🖥', 'Monitor'),
            'UPS': ('🔋', 'UPS'),
            'Projector': ('📽', 'Projector'),
        }

        for i, (key, (icon, name)) in enumerate(item_labels.items()):
            col = [col1, col2, col3][i % 3]
            with col:
                if st.checkbox(f'{icon} {name}', value=key in default_items, key=f'proc_check_{key}'):
                    selected.append(key)

        st.session_state.proc_items = selected if selected else ['Desktop Computer']

        if st.button('Continue → Enter Budget', type='primary', use_container_width=True):
            st.session_state.proc_step = 2
            st.rerun()

    elif step == 2:
        st.markdown('### Step 2: Enter Available Budget')
        selected_summary = ', '.join(st.session_state.proc_items)
        st.markdown(f'<div style="font-size:0.85rem; margin-bottom:1rem;"><strong>Selected items:</strong> {selected_summary}</div>', unsafe_allow_html=True)

        budget = st.number_input(
            'Available Budget (₱)',
            min_value=0,
            max_value=100_000_000,
            value=st.session_state.proc_budget,
            step=10_000,
            format='%d',
            key='proc_budget_input',
        )
        st.session_state.proc_budget = budget
        st.markdown(f'<div style="font-size: 1.5rem; font-weight: 700; color: #166534; margin: 0.5rem 0 1rem 0;">{_money(budget)}</div>', unsafe_allow_html=True)

        nav_c1, _, nav_c3 = st.columns([1, 1, 1])
        with nav_c1:
            if st.button('← Back to Items', use_container_width=True):
                st.session_state.proc_step = 1
                st.rerun()
        with nav_c3:
            if st.button('Generate AI Recommendation →', type='primary', use_container_width=True):
                st.session_state.proc_step = 3
                st.rerun()

    elif step == 3:
        selected_items = st.session_state.proc_items
        available_budget = st.session_state.proc_budget
        st.markdown(f'<div style="font-size:0.85rem; color:#6B7280; margin-bottom:1rem;">Items: {", ".join(selected_items)} | Budget: {_money(available_budget)}</div>', unsafe_allow_html=True)

        with st.spinner('AI analyzing procurement requirements...'):
            emp_map = build_employee_equipment_map(inv)
            office_counts = compute_office_equipment_counts(inv)
            total_unique_emps = len(emp_map)
            total_inv_records = len(inv)
            duplicate_records = total_inv_records - total_unique_emps
            assigned_desktop = int(emp_map['equipment_set'].apply(_has_desktop).sum())
            assigned_laptop = int(emp_map['equipment_set'].apply(_has_laptop).sum())
            emps_without_computer = int(emp_map['equipment_set'].apply(lambda s: not _has_desktop(s) and not _has_laptop(s)).sum())

            preview_summary = []
            for item in selected_items:
                if item in COMPUTER_ITEMS:
                    preview_summary.append(_computer_summary(emp_map, item))
                elif item == 'Printer':
                    preview_summary.append(_printer_summary(emp_map, office_counts))

            total_budget = sum(info['total'] for info in preview_summary)

        st.markdown('### 📊 AI Recommendation Overview')
        kc1, kc2, kc3, kc4 = st.columns(4)
        with kc1:
            render_kpi_card('💰', _money(total_budget), 'Required Budget', f'Available: {_money(available_budget)}')
        with kc2:
            render_kpi_card('👥', f'{emps_without_computer:,}', 'Employees Without Desktop/Laptop', 'Unique employees')
        with kc3:
            render_kpi_card('📦', f'{len(preview_summary)}', 'Separate Item Analyses', 'One section per item')
        with kc4:
            budget_status = 'HIGH' if total_budget > available_budget else 'WITHIN BUDGET'
            render_kpi_card('⚠️', budget_status, 'Budget Status', 'Compared with available budget')

        st.markdown("""
        <div class="ai-summary-box">
        The AI generated a separate procurement analysis for each selected item. Desktop and Laptop use one computer-ownership rule:
        employees who already have either a Desktop or Laptop are excluded. Printer is treated separately as a shared office resource,
        allocated by office employee count.
        </div>
        """, unsafe_allow_html=True)

        summary_rows = []
        for idx, item in enumerate(selected_items, 1):
            if idx > 1:
                st.divider()
            if item in COMPUTER_ITEMS:
                summary_rows.append(_render_computer_analysis(item, emp_map))
            elif item == 'Printer':
                summary_rows.append(_render_printer_analysis(emp_map, office_counts))
            else:
                render_section_header(f'{item} Procurement')
                st.info(f'Separate AI analysis for {item} is not configured yet. Desktop, Laptop, and Printer analyses are available.')

        summary_rows = [row for row in summary_rows if row and row.get('units', 0) > 0]
        if summary_rows:
            st.divider()
            render_section_header('📈 Item Budget Breakdown')
            for info in summary_rows:
                _render_item_summary_row(info)

            grand_total = sum(info['total'] for info in summary_rows)
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; padding:0.75rem 1rem; background:#F0FDF4; border:2px solid #166534; border-radius:8px; margin-top:0.5rem;">
                <span style="font-weight:700; color:#166534;">Grand Total</span>
                <span style="font-weight:700; color:#166534; font-size:1.1rem;">{_money(grand_total)}</span>
            </div>
            """, unsafe_allow_html=True)

        # LLM Procurement Recommendation
        model = selected_model or get_selected_model()
        rec_key = 'proc_llm_rec'
        if model:
            st.divider()
            render_section_header('🤖 AI Procurement Recommendation')
            with st.container():
                st.markdown(f'Generating recommendation using **{model}**...')
                if st.button('Generate LLM Recommendation', key=rec_key, type='primary', use_container_width=True):
                    context = {
                        'items': selected_items,
                        'total_employees': total_unique_emps,
                        'without_computer': emps_without_computer,
                        'offices_needing': len(office_counts),
                        'required_budget': grand_total,
                        'available_budget': available_budget,
                        'desktop_recommended': next((s['units'] for s in summary_rows if s['item'] == 'Desktop Computer'), 0),
                        'laptop_recommended': next((s['units'] for s in summary_rows if s['item'] == 'Laptop Computer'), 0),
                        'printer_recommended': next((s['units'] for s in summary_rows if s['item'] == 'Printer'), 0),
                    }
                    budget_info = {'available': available_budget, 'required': grand_total}
                    with st.spinner(f'Generating recommendation with {model}...'):
                        llm_result = generate_procurement_recommendation(model, context, budget_info)
                    if llm_result:
                        sections = {'## Procurement Summary': '', '## Analysis': '', '## Recommendation': '', '## Why': ''}
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
                            tabs = st.tabs(['Summary', 'Analysis', 'Recommendation', 'Why'])
                            section_keys = ['## Procurement Summary', '## Analysis', '## Recommendation', '## Why']
                            for tab, sk in zip(tabs, section_keys):
                                with tab:
                                    content = sections.get(sk, '*No content*')
                                    st.markdown(content if content.strip() else '*No content*')
                        else:
                            st.markdown(f'<div class="ai-summary-box">{llm_result}</div>', unsafe_allow_html=True)
                        st.caption(f'Recommendation by **{model}**')
                    else:
                        st.warning(f'Could not reach Ollama model **{model}**. Ensure `ollama serve` is running.')
                else:
                    st.caption('Click to generate an AI-powered procurement recommendation with detailed reasoning.')

        with st.expander('❓ How AI Computed These Recommendations'):
            st.markdown(f"""
**Employee Consolidation**
- Total inventory records: {total_inv_records:,}
- Unique employees: {total_unique_emps:,}
- Duplicate employee records removed: {duplicate_records:,}

**Computer Ownership Rule**
- Employees already assigned Desktop: {assigned_desktop:,}
- Employees already assigned Laptop: {assigned_laptop:,}
- Employees without Desktop or Laptop: {emps_without_computer:,}
- Desktop/Laptop recommendation list = only employees without Desktop and without Laptop.

**Printer Formula**
- Target Printers = ROUND(Office Employee Count / Largest Office Employee Count × 7)
- Recommended Printers = Target Printers - Existing Printers
- Minimum recommendation = 0
""")
            for info in summary_rows:
                st.markdown(f"- **{info['item']}**: {info['units']:,} × {_money(info['unit_cost'])} = {_money(info['total'])}")

        with st.expander('📋 Procurement Policy'):
            st.markdown("""
| Equipment | Allocation Rule |
|-----------|----------------|
| **Desktop** | Recommend only to employees without Desktop and without Laptop |
| **Laptop** | Recommend only to employees without Desktop and without Laptop |
| **Printer** | Shared office resource; proportional to unique employee count, max target 7 |
""")

        with st.expander('🔐 Priority Methodology'):
            st.markdown("""
| Priority | Criteria |
|----------|----------|
| **1** | No Desktop, no Laptop, and no ICT equipment recorded |
| **2** | No Desktop or Laptop; only peripheral ICT equipment recorded |
| **3** | Old computer, used for replacement analysis and excluded from new Desktop/Laptop assignment |
| **4** | Poor computer condition, used for replacement analysis and excluded from new Desktop/Laptop assignment |
""")

        nav_c1, _, nav_c3 = st.columns([1, 1, 1])
        with nav_c1:
            if st.button('← Adjust Budget', use_container_width=True):
                st.session_state.proc_step = 2
                st.rerun()
        with nav_c3:
            if st.button('🔄 New Analysis', type='primary', use_container_width=True):
                st.session_state.proc_step = 1
                st.rerun()
