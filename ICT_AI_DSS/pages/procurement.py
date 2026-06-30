"""Procurement page — step-by-step workflow."""

import streamlit as st
import pandas as pd
from backend import (
    CURRENT_YEAR, EQUIPMENT_COSTS, ALL_PROCUREMENT_ITEMS, MAX_PRINTER_TARGET,
    build_employee_equipment_map, compute_office_equipment_counts,
    policy_full_analysis, compute_priority_scores, optimize_budget,
    classify_employee_priority_tiers, get_item_employee_list,
    compute_item_office_priority, compute_printer_office_priority,
    generate_per_item_ai_explanation,
)
from ui_components import render_section_header, render_kpi_card, render_export_buttons, to_csv, to_excel


def render(inv, repl_df, ollama_models):
    render_section_header('💰 Procurement')

    # ── Step indicator ──
    step = st.session_state.get('proc_step', 1)
    steps = ['Select Procurement Items', 'Enter Budget', 'Generate AI Recommendation']
    icons = ['📦', '💵', '🤖']

    step_html = '<div class="workflow-steps">'
    for i, (s, ic) in enumerate(zip(steps, icons), 1):
        cls = 'active' if i == step else ('completed' if i < step else '')
        step_html += f'<div class="workflow-step {cls}">{ic} {s}</div>'
        if i < len(steps):
            step_html += '<span class="workflow-arrow">→</span>'
    step_html += '</div>'
    st.markdown(step_html, unsafe_allow_html=True)

    # ── Session state init ──
    if 'proc_items' not in st.session_state:
        st.session_state.proc_items = ['Desktop Computer', 'Laptop Computer']
    if 'proc_budget' not in st.session_state:
        st.session_state.proc_budget = 1675000

    # ================================================================
    # STEP 1 — Select Procurement Items
    # ================================================================
    if step == 1:
        st.markdown('### Step 1: Select Procurement Items')
        st.markdown('<div style="color: #6B7280; font-size: 0.85rem; margin-bottom: 1rem;">Choose the ICT equipment to procure:</div>', unsafe_allow_html=True)

        default_items = st.session_state.proc_items
        col1, col2, col3 = st.columns(3)
        selected = []
        item_labels = {
            'Desktop Computer': ('💻', 'Desktop Computer'),
            'Laptop Computer': ('💻', 'Laptop Computer'),
            'Printer': ('🖨️', 'Printer'),
            'CCTV': ('📹', 'CCTV'),
            'Router': ('📡', 'Router'),
            'Switch': ('🔀', 'Switch'),
            'Scanner': ('📄', 'Scanner'),
            'Monitor': ('🖥️', 'Monitor'),
            'UPS': ('🔋', 'UPS'),
            'Projector': ('📽️', 'Projector'),
        }

        for i, (key, (icon, name)) in enumerate(item_labels.items()):
            col = [col1, col2, col3][i % 3]
            with col:
                checked = st.checkbox(f'{icon} {name}', value=key in default_items, key=f'proc_check_{key}')
                if checked:
                    selected.append(key)

        st.session_state.proc_items = selected if selected else ['Desktop Computer']

        if st.button('Continue → Enter Budget', type='primary', use_container_width=True):
            st.session_state.proc_step = 2
            st.rerun()

    # ================================================================
    # STEP 2 — Enter Available Budget
    # ================================================================
    elif step == 2:
        st.markdown('### Step 2: Enter Available Budget')
        st.markdown('<div style="color: #6B7280; font-size: 0.85rem; margin-bottom: 1rem;">Set the total available procurement budget:</div>', unsafe_allow_html=True)

        selected_summary = ', '.join(st.session_state.proc_items)
        st.markdown(f'<div style="font-size:0.85rem; margin-bottom:1rem;"><strong>Selected items:</strong> {selected_summary}</div>', unsafe_allow_html=True)

        budget = st.number_input(
            'Available Budget (₱)',
            min_value=0, max_value=100_000_000,
            value=st.session_state.proc_budget, step=10_000, format='%d',
            key='proc_budget_input',
        )
        st.session_state.proc_budget = budget

        st.markdown(f'<div style="font-size: 1.5rem; font-weight: 700; color: #166534; margin: 0.5rem 0 1rem 0;">₱{budget:,}</div>', unsafe_allow_html=True)

        nav_c1, nav_c2, nav_c3 = st.columns([1, 1, 1])
        with nav_c1:
            if st.button('← Back to Items', use_container_width=True):
                st.session_state.proc_step = 1
                st.rerun()
        with nav_c3:
            if st.button('Generate AI Recommendation →', type='primary', use_container_width=True):
                st.session_state.proc_step = 3
                st.rerun()

    # ================================================================
    # STEP 3 — AI Recommendation Results
    # ================================================================
    elif step == 3:
        selected_items = st.session_state.proc_items
        available_budget = st.session_state.proc_budget

        st.markdown(f'<div style="font-size:0.85rem; color:#6B7280; margin-bottom:1rem;">Items: {", ".join(selected_items)} | Budget: ₱{available_budget:,}</div>', unsafe_allow_html=True)

        # ── Run analysis ──
        with st.spinner('AI analyzing procurement requirements...'):
            emp_map = build_employee_equipment_map(inv)
            office_counts = compute_office_equipment_counts(inv)
            emp_recs, shared_df, ups_df, offices_df, proc_list, summary, total_units, total_budget, printer_df = \
                policy_full_analysis(emp_map, office_counts, selected_items)

            total_unique_emps = len(emp_map)
            emps_with_computer = emp_map['equipment_set'].apply(
                lambda s: any(v in s for v in ['Desktop Computer', 'Laptop Computer', 'Laptop Computers'])
            ).sum()
            emps_without_computer = total_unique_emps - emps_with_computer
            offices_needing = len(offices_df[offices_df['Employees without Computer'] > 0]) if offices_df is not None and not offices_df.empty else 0
            num_offices = len(offices_df) if offices_df is not None else 0

            desktop_count = summary.get('Desktop Computer', {}).get('units', 0)
            laptop_count = summary.get('Laptop Computer', {}).get('units', 0)
            printer_count = summary.get('Printer', {}).get('units', 0)

        # ════════════════════════════════════════════
        # 4 KPI CARDS
        # ════════════════════════════════════════════
        st.markdown('### 📊 AI Recommendation Overview')
        st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
        kc1, kc2, kc3, kc4 = st.columns(4)
        with kc1:
            render_kpi_card('💰', f'₱{total_budget:,}', 'Required Budget', f'Available: ₱{available_budget:,}')
        with kc2:
            render_kpi_card('👥', f'{emps_without_computer}', 'Employees Requiring ICT', 'Unique individuals')
        with kc3:
            render_kpi_card('🏢', f'{offices_needing}', 'Offices Requiring Procurement', f'Of {num_offices} total')
        with kc4:
            priority_val = 'HIGH' if emps_without_computer > 0 else 'LOW'
            render_kpi_card('⚠️', priority_val, 'Procurement Priority', 'Based on employee need')
        st.markdown('</div>', unsafe_allow_html=True)

        # ════════════════════════════════════════════
        # PER-ITEM AI ANALYSIS SECTIONS
        # ════════════════════════════════════════════
        priority_tiers_df = classify_employee_priority_tiers(emp_map, repl_df)

        for item in selected_items:
            if item == 'Desktop Computer':
                st.markdown('### 💻 Desktop Computer Procurement')
                w_desk = int(emp_map['equipment_set'].apply(lambda s: 'Desktop Computer' in s).sum())
                w_lap = int(emp_map['equipment_set'].apply(lambda s: any(v in s for v in ['Laptop Computer', 'Laptop Computers'])).sum())
                wo_comp = total_unique_emps - (w_desk + w_lap)
                rec_count = summary.get('Desktop Computer', {}).get('units', 0)
                rec_budget = summary.get('Desktop Computer', {}).get('total', 0)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric('Total Employees', f'{total_unique_emps:,}')
                c2.metric('Already Assigned Desktop', f'{w_desk:,}')
                c3.metric('Already Assigned Laptop', f'{w_lap:,}')
                c4.metric('Without Desktop or Laptop', f'{wo_comp:,}')

                st.markdown(f'<div style="display:flex; gap:2rem; margin:0.5rem 0 1rem 0; padding:0.75rem 1rem; background:#FFF; border:1px solid #E5E7EB; border-radius:8px;"><span><strong>Recommended Desktop Purchases:</strong> {rec_count}</span><span><strong>Estimated Budget:</strong> ₱{rec_budget:,}</span></div>', unsafe_allow_html=True)

                office_df = compute_item_office_priority(emp_map, 'Desktop Computer', selected_items)
                if not office_df.empty:
                    st.markdown('**Office Priority Ranking**')
                    disp = office_df[['Rank', 'Office', 'Employees Needing Desktop', 'Budget', 'Priority']].copy()
                    disp['Budget'] = disp['Budget'].apply(lambda x: f'₱{int(x):,}')
                    def _hl_off1(row):
                        if row['Priority'] == 'HIGH':
                            return ['background:#FEF2F2;color:#DC2626;font-weight:600']*len(row)
                        elif row['Priority'] == 'MEDIUM':
                            return ['background:#FFFBEB;color:#D97706']*len(row)
                        return ['']*len(row)
                    st.dataframe(disp.style.apply(_hl_off1, axis=1), use_container_width=True, height=200, hide_index=True)

                emp_list = get_item_employee_list(emp_recs, 'Desktop Computer', priority_tiers_df)
                if not emp_list.empty:
                    st.markdown('**Employee Priority List**')
                    disp_emp = emp_list[['office_rank', 'employeeName', 'office', 'current_equipment', 'priority_label', 'item', 'cost']].copy()
                    disp_emp.columns = ['Rank', 'Employee', 'Office', 'Current Equipment', 'Reason', 'Recommended Item', 'Cost']
                    disp_emp['Cost'] = disp_emp['Cost'].apply(lambda x: f'₱{int(x):,}')
                    st.dataframe(disp_emp, use_container_width=True, height=300, hide_index=True)

                expl = generate_per_item_ai_explanation('Desktop Computer', emp_map, office_counts, desktop_count=rec_count)
                st.markdown(f'<div class="ai-summary-box">{expl}</div>', unsafe_allow_html=True)
                st.markdown('<hr style="margin:1.5rem 0;">', unsafe_allow_html=True)

            elif item == 'Laptop Computer':
                st.markdown('### 💼 Laptop Procurement')
                w_desk = int(emp_map['equipment_set'].apply(lambda s: 'Desktop Computer' in s).sum())
                w_lap = int(emp_map['equipment_set'].apply(lambda s: any(v in s for v in ['Laptop Computer', 'Laptop Computers'])).sum())
                wo_comp = total_unique_emps - (w_desk + w_lap)
                rec_count = summary.get('Laptop Computer', {}).get('units', 0)
                rec_budget = summary.get('Laptop Computer', {}).get('total', 0)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric('Total Employees', f'{total_unique_emps:,}')
                c2.metric('Already Assigned Desktop', f'{w_desk:,}')
                c3.metric('Already Assigned Laptop', f'{w_lap:,}')
                c4.metric('Without Desktop or Laptop', f'{wo_comp:,}')

                st.markdown(f'<div style="display:flex; gap:2rem; margin:0.5rem 0 1rem 0; padding:0.75rem 1rem; background:#FFF; border:1px solid #E5E7EB; border-radius:8px;"><span><strong>Recommended Laptop Purchases:</strong> {rec_count}</span><span><strong>Estimated Budget:</strong> ₱{rec_budget:,}</span></div>', unsafe_allow_html=True)

                office_df = compute_item_office_priority(emp_map, 'Laptop Computer', selected_items)
                if not office_df.empty:
                    st.markdown('**Office Priority Ranking**')
                    disp = office_df[['Rank', 'Office', 'Employees Needing Laptop', 'Budget', 'Priority']].copy()
                    disp['Budget'] = disp['Budget'].apply(lambda x: f'₱{int(x):,}')
                    def _hl_off2(row):
                        if row['Priority'] == 'HIGH':
                            return ['background:#FEF2F2;color:#DC2626;font-weight:600']*len(row)
                        elif row['Priority'] == 'MEDIUM':
                            return ['background:#FFFBEB;color:#D97706']*len(row)
                        return ['']*len(row)
                    st.dataframe(disp.style.apply(_hl_off2, axis=1), use_container_width=True, height=200, hide_index=True)

                emp_list = get_item_employee_list(emp_recs, 'Laptop Computer', priority_tiers_df)
                if not emp_list.empty:
                    st.markdown('**Employee Priority List**')
                    disp_emp = emp_list[['office_rank', 'employeeName', 'office', 'current_equipment', 'priority_label', 'item', 'cost']].copy()
                    disp_emp.columns = ['Rank', 'Employee', 'Office', 'Current Equipment', 'Reason', 'Recommended Item', 'Cost']
                    disp_emp['Cost'] = disp_emp['Cost'].apply(lambda x: f'₱{int(x):,}')
                    st.dataframe(disp_emp, use_container_width=True, height=300, hide_index=True)

                expl = generate_per_item_ai_explanation('Laptop Computer', emp_map, office_counts, laptop_count=rec_count)
                st.markdown(f'<div class="ai-summary-box">{expl}</div>', unsafe_allow_html=True)
                st.markdown('<hr style="margin:1.5rem 0;">', unsafe_allow_html=True)

            elif item == 'Printer':
                st.markdown('### 🖨 Printer Procurement')
                printer_office_df = compute_printer_office_priority(emp_map, office_counts)
                if not printer_office_df.empty:
                    disp = printer_office_df[['Rank', 'Office', 'Employees', 'Existing Printers', 'Target Printers', 'Additional Printers Needed', 'Estimated Cost', 'Reason']].copy()
                    disp['Estimated Cost'] = disp['Estimated Cost'].apply(lambda x: f'₱{int(x):,}')
                    st.dataframe(disp, use_container_width=True, height=350, hide_index=True)

                expl = generate_per_item_ai_explanation('Printer', emp_map, office_counts)
                st.markdown(f'<div class="ai-summary-box">{expl}</div>', unsafe_allow_html=True)
                st.markdown('<hr style="margin:1.5rem 0;">', unsafe_allow_html=True)

        # ════════════════════════════════════════════
        # OFFICE PRIORITY TABLE (Overview)
        # ════════════════════════════════════════════
        if offices_df is not None and not offices_df.empty:
            render_section_header('🏢 Office Priority')
            rank_df = offices_df[['Rank', 'Office', 'Employees without Computer', 'Estimated Budget', 'Priority']].copy()
            rank_df.columns = ['Rank', 'Office', 'Employees Requiring ICT', 'Budget', 'Priority']
            rank_df['Budget'] = rank_df['Budget'].apply(lambda x: f'₱{int(x):,}')

            def highlight_priority_row(row):
                if row['Priority'] == 'HIGH':
                    return ['background-color: #FEF2F2; color: #DC2626; font-weight: 600'] * len(row)
                elif row['Priority'] == 'MEDIUM':
                    return ['background-color: #FFFBEB; color: #D97706'] * len(row)
                return [''] * len(row)

            st.dataframe(
                rank_df.style.apply(highlight_priority_row, axis=1),
                use_container_width=True, height=350, hide_index=True
            )

        # ════════════════════════════════════════════
        # EMPLOYEE PROCUREMENT LIST
        # ════════════════════════════════════════════
        if emp_recs is not None and not emp_recs.empty:
            render_section_header('👤 Employee Procurement List')

            priority_df = compute_priority_scores(emp_map)
            merged = priority_df.merge(emp_recs[['employeeName', 'office', 'item', 'cost', 'reason']], on=['employeeName', 'office'], how='inner')

            display = merged[['employeeName', 'office', 'equipment_set', 'item', 'cost', 'priority_reason', 'priority_score']].copy()
            display.columns = ['Employee Name', 'Office', 'Current Equipment', 'Recommended', 'Estimated Cost', 'Reason', 'Priority']
            display['Current Equipment'] = display['Current Equipment'].apply(lambda s: ', '.join(sorted(s)) if isinstance(s, set) and s else 'None')
            display['Estimated Cost'] = display['Estimated Cost'].apply(lambda x: f'₱{int(x):,}')

            # Add Status column
            priority_df_opt = compute_priority_scores(emp_map)
            opt_result, remaining, total_requirement = optimize_budget(priority_df_opt, emp_recs, ups_df, available_budget)
            if not opt_result.empty:
                opt_map = opt_result.set_index('Employee Name')['Status'].to_dict()
                display['Status'] = display['Employee Name'].map(opt_map).fillna('WAITING FOR NEXT BUDGET')
            else:
                display['Status'] = 'APPROVED'

            display = display.sort_values('Priority', ascending=False).reset_index(drop=True)

            def style_emp_row(row):
                score = row['Priority']
                status = row['Status']
                styles = ['' for _ in range(len(row))]
                if score >= 80 or status == 'WAITING FOR NEXT BUDGET':
                    styles = ['background-color: #FEF2F2; color: #DC2626'] * len(row)
                elif score >= 50:
                    styles = ['background-color: #FFFBEB; color: #D97706'] * len(row)
                if status == 'APPROVED':
                    styles[len(styles)-1] = 'background-color: #F0FDF4; color: #16A34A; font-weight: 600'
                return styles

            st.dataframe(
                display.style.apply(style_emp_row, axis=1),
                use_container_width=True, height=400, hide_index=True
            )

            export_c1, export_c2 = st.columns(2)
            with export_c1:
                st.download_button('📥 Download CSV', to_csv(display), 'employee_procurement_list.csv', 'text/csv', use_container_width=True)
            with export_c2:
                st.download_button('📥 Download Excel', to_excel(display), 'employee_procurement_list.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', use_container_width=True)

        # ════════════════════════════════════════════
        # BUDGET BREAKDOWN
        # ════════════════════════════════════════════
        items_with_units = [(item, info) for item, info in summary.items() if info.get('units', 0) > 0]
        if items_with_units:
            render_section_header('📈 Budget Breakdown')

            for item, info in items_with_units:
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; padding:0.5rem 1rem; background:#FFFFFF; border:1px solid #E5E7EB; border-radius:8px; margin-bottom:0.35rem;">
                    <span style="font-weight:500;">{item}</span>
                    <span>{info['units']:,} × ₱{info['unit_cost']:,} = <strong style="color:#166534;">₱{info['total']:,}</strong></span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; padding:0.75rem 1rem; background:#F0FDF4; border:2px solid #166534; border-radius:8px; margin-top:0.5rem;">
                <span style="font-weight:700; color:#166534;">Grand Total</span>
                <span style="font-weight:700; color:#166534; font-size:1.1rem;">₱{total_budget:,}</span>
            </div>
            """, unsafe_allow_html=True)

        # ════════════════════════════════════════════
        # DETAILS — EXPANDERS
        # ════════════════════════════════════════════
        with st.expander('❓ How AI Computed This Recommendation'):
            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.markdown('**Decision Flow**')
                total_inv_records = len(inv)
                duplicate_records = total_inv_records - total_unique_emps
                st.markdown(f"""```
Total Records: {total_inv_records:,}
       ↓
Unique Employees: {total_unique_emps:,}
       ↓
With Desktop/Laptop: {emps_with_computer:,}
       ↓
Without Computer: {emps_without_computer:,}
       ↓
Office Priority Analysis
       ↓
Budget Allocation
       ↓
Final Recommendation
```""")
            with col_b:
                st.markdown('**Key Calculations**')
                st.markdown(f"""
- **Unique Employees** = {total_inv_records:,} − {duplicate_records:,} = {total_unique_emps:,}
- **Employees Without PC** = {total_unique_emps:,} − {emps_with_computer:,} = {emps_without_computer:,}
- **Budget Required** = Σ(Units × Unit Cost) = ₱{total_budget:,}
""")
                for item, info in items_with_units:
                    st.markdown(f"- **{item}**: {info['units']} × ₱{info['unit_cost']:,} = ₱{info['total']:,}")

        with st.expander('📋 Procurement Policy'):
            st.markdown("""
| Equipment | Allocation Rule |
|-----------|----------------|
| **Desktop** | 1 per employee without computer |
| **Laptop** | 1 per employee without computer |
| **Printer** | Shared — proportional to employee count (max 7) |
| **CCTV** | Shared — max 4 per office |
| **Projector** | Shared — max 1 per division |
| **Switch** | Shared — max 1 per division |
| **Router** | Shared — max 1 per division |
| **UPS** | 1 per Desktop recipient without UPS |
""")

        with st.expander('🔬 Detailed Computation'):
            st.markdown(f"""
**Duplicate Records Removed:**
- Total inventory records: {total_inv_records if 'total_inv_records' in dir() else len(inv):,}
- Unique employees (by name): {total_unique_emps:,}
- Duplicates removed: {(len(inv) - total_unique_emps):,}

**Office Calculations:**
""")
            if offices_df is not None and not offices_df.empty:
                st.dataframe(offices_df, use_container_width=True, height=200, hide_index=True)

            st.markdown('**Printer Calculations:**')
            if printer_df is not None and not printer_df.empty:
                st.markdown("- Proportional allocation based on employee count")
                st.markdown("- Largest division receives max target of 7 printers")
                st.dataframe(printer_df[['office', 'units', 'cost_per_unit']], use_container_width=True, hide_index=True)

        with st.expander('🔐 Priority Scoring Methodology'):
            st.markdown("""
| Score | Criteria |
|-------|----------|
| **100** | Employee has neither Desktop nor Laptop |
| **90** | Employee has no ICT equipment |
| **80** | Employee only owns peripherals |
| **70** | Office ICT coverage is below agency average |
| **60** | Computer is older than 5 years |
| **50** | Computer condition is Poor |
| **0** | Already has a Desktop or Laptop |
""")

        with st.expander('📜 Duplicate Records Removed'):
            dup_count = len(inv) - total_unique_emps
            st.markdown(f"""
The system automatically consolidated **{dup_count:,}** duplicate employee records.
Each employee is counted only once regardless of how many ICT assets they are assigned.
This ensures fair and accurate procurement allocation.
""")

        # ════════════════════════════════════════════
        # AI DEEP ANALYSIS (Ollama)
        # ════════════════════════════════════════════
        if ollama_models:
            with st.expander('🤖 Deep AI Budget Analysis', expanded=False):
                st.markdown('Generate an AI-powered strategic analysis using Ollama.')
                if st.button('📊 Generate Deep AI Analysis', use_container_width=True, key='deep_ai_btn'):
                    import ollama
                    budget_ai_model = ollama_models[0]
                    if 'qwen3:latest' in ollama_models:
                        budget_ai_model = 'qwen3:latest'

                    priority_df_opt = compute_priority_scores(emp_map)
                    opt_result, _, total_requirement = optimize_budget(priority_df_opt, emp_recs, ups_df, available_budget)
                    employees_funded = len(opt_result[opt_result['Status'] == 'APPROVED']) if not opt_result.empty else 0
                    employees_waiting = len(opt_result[opt_result['Status'] == 'WAITING FOR NEXT BUDGET']) if not opt_result.empty else 0
                    remaining = max(0, available_budget - total_requirement)

                    context_lines = [
                        f"BUDGET DATA:",
                        f"- Available Budget: ₱{available_budget:,}",
                        f"- Total Requirement: ₱{total_requirement:,}",
                        f"- Remaining: ₱{remaining:,}",
                        f"- Employees Funded: {employees_funded}",
                        f"- Employees Waiting: {employees_waiting}",
                    ]
                    for item in selected_items:
                        info = summary.get(item, {'units': 0, 'unit_cost': 0})
                        if info['units'] > 0:
                            context_lines.append(f"- {item}: {info['units']} units @ ₱{info['unit_cost']:,} = ₱{info['units'] * info['unit_cost']:,}")

                    prompt = f"""You are a Senior ICT Procurement Analyst. Analyze the budget data:

{' '.join(context_lines)}

Provide a structured analysis covering:
1. Budget Adequacy
2. Spending Breakdown
3. Priority Assessment
4. Gap Analysis
5. Risk Assessment
6. Phasing Recommendation
7. Strategic Recommendations"""

                    with st.chat_message('assistant'):
                        with st.spinner('AI analyzing budget...'):
                            msg = st.empty()
                            full = ''
                            try:
                                stream = ollama.chat(model=budget_ai_model, messages=[{'role': 'user', 'content': prompt}], stream=True)
                                for chunk in stream:
                                    if 'message' in chunk and 'content' in chunk['message']:
                                        full += chunk['message']['content']
                                        msg.markdown(full + '▌')
                                msg.markdown(full)
                            except Exception as e:
                                st.error(f'Ollama error: {e}')

        # ── Navigation ──
        nav_c1, nav_c2, nav_c3 = st.columns([1, 1, 1])
        with nav_c1:
            if st.button('← Adjust Budget', use_container_width=True):
                st.session_state.proc_step = 2
                st.rerun()
        with nav_c3:
            if st.button('🔄 New Analysis', type='primary', use_container_width=True):
                st.session_state.proc_step = 1
                st.rerun()
