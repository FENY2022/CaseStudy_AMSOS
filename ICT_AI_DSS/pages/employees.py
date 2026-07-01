"""Employees page with office-based equipment prioritization."""

import streamlit as st
import pandas as pd

from model_manager import generate_employee_recommendation, get_selected_model
from ui_components import render_section_header, to_csv, to_excel


COMPUTER_KEYWORDS = ('Desktop', 'Laptop')
DEFAULT_EQUIPMENT = ['Desktop Computer', 'Laptop Computers']
INVALID_PERSON_VALUES = ('0', 'COS', 'PMD')


def _clean_text(value):
    if pd.isna(value):
        return ''
    text = str(value).strip()
    return '' if text.upper() in ('', 'N/A', 'NULL', 'NONE', 'NAN') else text


def _employee_name(row):
    actual_user = _person_name_or_blank(row.get('actualUser', ''))
    employee_name = _person_name_or_blank(row.get('employeeName', ''))
    return actual_user or employee_name


def _person_name_or_blank(value):
    text = _clean_text(value)
    if not text:
        return ''
    while len(text) > 1 and text.startswith('0') and text[1].isalpha():
        text = text[1:].strip()
    if text.upper() in INVALID_PERSON_VALUES or text.isdigit():
        return ''
    if len(text) <= 3 and text.isupper():
        return ''
    return text


def _normalize_office(value):
    office = _clean_text(value)
    if office.lower() == 'legal':
        return 'LEGAL'
    return office or 'Unassigned'


def _has_equipment(equipment_set, equipment_name):
    target = equipment_name.lower()
    if 'desktop' in target:
        return any('desktop' in item.lower() for item in equipment_set)
    if 'laptop' in target:
        return any('laptop' in item.lower() for item in equipment_set)
    return any(item.lower() == target for item in equipment_set)


def _has_computer(equipment_set):
    return any(keyword.lower() in item.lower() for item in equipment_set for keyword in COMPUTER_KEYWORDS)


def _current_equipment(equipment_set):
    return ', '.join(sorted(equipment_set)) if equipment_set else 'No equipment recorded'


def _equipment_options(inv):
    unique_items = sorted(_clean_text(item) for item in inv['equipmentType'].dropna().unique())
    unique_items = [item for item in unique_items if item]
    preferred = [item for item in DEFAULT_EQUIPMENT if item in unique_items]
    if 'Laptop Computer' not in unique_items and 'Laptop Computers' in unique_items:
        preferred = ['Desktop Computer', 'Laptop Computers']
    remaining = [item for item in unique_items if item not in preferred]
    return preferred + remaining


def _recommended_equipment(row, selected_equipment):
    equipment_set = row['equipment_set']
    missing = [item for item in selected_equipment if not _has_equipment(equipment_set, item)]
    selected_computers = [
        item for item in selected_equipment
        if any(keyword.lower() in item.lower() for keyword in COMPUTER_KEYWORDS)
    ]

    if not row['has_computer'] and selected_computers:
        nature = row['nature_of_work'].lower()
        if len(selected_computers) > 1:
            desktop = next((item for item in selected_computers if 'desktop' in item.lower()), selected_computers[0])
            laptop = next((item for item in selected_computers if 'laptop' in item.lower()), selected_computers[0])
            return desktop if 'technical' in nature else laptop
        return selected_computers[0]

    return ', '.join(missing) if missing else 'No missing selected equipment'


def _priority_reason(row, selected_equipment):
    if not row['has_computer']:
        return 'Employee has no Desktop or Laptop assigned.'

    missing = [item for item in selected_equipment if not _has_equipment(row['equipment_set'], item)]
    if missing:
        return f"Missing selected equipment: {', '.join(missing)}."
    return 'Employee already has the selected equipment.'


def _priority_level(without_computer, coverage_rate):
    if without_computer >= 10 or coverage_rate < 50:
        return 'HIGH'
    if without_computer >= 4 or coverage_rate < 80:
        return 'MEDIUM'
    return 'LOW'


def _build_employee_equipment_map(inv):
    records = []
    source = inv.copy()
    source['_priority_employee'] = source.apply(_employee_name, axis=1)
    source['_priority_office'] = source['officeDivision'].apply(_normalize_office)
    source = source[source['_priority_employee'] != '']

    for employee, group in source.groupby('_priority_employee', dropna=False):
        office_counts = group['_priority_office'].value_counts()
        office = office_counts.index[0] if not office_counts.empty else 'Unassigned'
        equipment_set = set(_clean_text(item) for item in group['equipmentType'].dropna())
        equipment_set = {item for item in equipment_set if item}
        actual_status = group['actualUserStatusOfEmployment'].dropna().map(_clean_text)
        employee_status = group['statusOfEmployment'].dropna().map(_clean_text)
        nature = group['natureOfWork'].dropna().map(_clean_text)
        records.append({
            'employee_name': employee,
            'office': office,
            'status': next((item for item in actual_status if item), next((item for item in employee_status if item), 'Unknown')),
            'nature_of_work': next((item for item in nature if item), 'Unknown'),
            'equipment_set': equipment_set,
            'has_computer': _has_computer(equipment_set),
        })

    emp_map = pd.DataFrame(records)
    if emp_map.empty:
        return emp_map
    return emp_map.sort_values(['office', 'employee_name']).reset_index(drop=True)


def _office_priority(emp_map):
    if emp_map.empty:
        return pd.DataFrame()

    rows = []
    for office, group in emp_map.groupby('office'):
        total = len(group)
        with_computer = int(group['has_computer'].sum())
        without_computer = total - with_computer
        coverage_rate = (with_computer / total * 100) if total else 0
        rows.append({
            'Office': office,
            'Total Employees': total,
            'With Computer': with_computer,
            'Without Computer': without_computer,
            'Coverage Rate': coverage_rate,
            'Priority Level': _priority_level(without_computer, coverage_rate),
        })

    office_df = pd.DataFrame(rows)
    office_df = office_df.sort_values(
        ['Without Computer', 'Coverage Rate', 'Total Employees'],
        ascending=[False, True, False],
    ).reset_index(drop=True)
    office_df['Office Priority'] = range(1, len(office_df) + 1)
    return office_df[[
        'Office Priority', 'Office', 'Total Employees', 'With Computer',
        'Without Computer', 'Coverage Rate', 'Priority Level'
    ]]


def _employment_score(status):
    lowered = status.lower()
    if 'permanent' in lowered:
        return 20
    if 'casual' in lowered or 'contract' in lowered or 'job order' in lowered or 'cos' in lowered:
        return 12
    return 8


def _work_score(nature):
    lowered = nature.lower()
    if 'technical' in lowered:
        return 20
    if 'administrative' in lowered or 'clerical' in lowered:
        return 15
    return 10


def _build_priority_list(emp_map, office_df, selected_equipment, selected_offices, show_only_without_computer):
    if emp_map.empty or office_df.empty or not selected_equipment:
        return pd.DataFrame()

    office_lookup = office_df.set_index('Office').to_dict('index')
    rows = []
    for _, row in emp_map.iterrows():
        if selected_offices and row['office'] not in selected_offices:
            continue

        missing_selected = [item for item in selected_equipment if not _has_equipment(row['equipment_set'], item)]
        if show_only_without_computer and row['has_computer']:
            continue
        if not show_only_without_computer and not missing_selected and row['has_computer']:
            continue

        office_info = office_lookup.get(row['office'], {})
        coverage_rate = float(office_info.get('Coverage Rate', 100))
        office_priority = int(office_info.get('Office Priority', 999))
        no_computer_score = 100 if not row['has_computer'] else 0
        office_need_score = max(0, int(100 - coverage_rate))
        priority_score = (
            no_computer_score +
            _employment_score(row['status']) +
            _work_score(row['nature_of_work']) +
            office_need_score
        )

        rows.append({
            'Office Priority': office_priority,
            'Office': row['office'],
            'Employee Name': row['employee_name'],
            'Current Equipment': _current_equipment(row['equipment_set']),
            'Missing Selected Equipment': ', '.join(missing_selected) if missing_selected else 'None',
            'Recommended Equipment': _recommended_equipment(row, selected_equipment),
            'Status': row['status'],
            'Nature of Work': row['nature_of_work'],
            'Priority Score': priority_score,
            'Reason': _priority_reason(row, selected_equipment),
        })

    priority_df = pd.DataFrame(rows)
    if priority_df.empty:
        return priority_df
    priority_df = priority_df.sort_values(
        ['Office Priority', 'Priority Score', 'Employee Name'],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    priority_df.insert(0, 'Rank', range(1, len(priority_df) + 1))
    return priority_df


def _format_office_table(office_df):
    display = office_df.copy()
    if not display.empty:
        display['Coverage Rate'] = display['Coverage Rate'].map(lambda value: f'{value:.1f}%')
    return display


def render(inv, emp_df, selected_model=None):
    render_section_header('Employees')

    emp_map = _build_employee_equipment_map(inv)
    office_df = _office_priority(emp_map)

    total_unique = len(emp_map)
    total_with_pc = int(emp_map['has_computer'].sum()) if not emp_map.empty else 0
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
        coverage = total_with_pc / total_unique * 100 if total_unique > 0 else 0
        st.metric('Coverage Rate', f'{coverage:.1f}%')
    st.markdown('</div>', unsafe_allow_html=True)

    render_section_header('Employee Prioritization by Office')

    equipment_options = _equipment_options(inv)
    default_equipment = [item for item in DEFAULT_EQUIPMENT if item in equipment_options]
    if not default_equipment and equipment_options:
        default_equipment = equipment_options[:1]

    f1, f2, f3 = st.columns([1.4, 1.4, 1])
    with f1:
        selected_equipment = st.multiselect(
            'Equipment to prioritize',
            equipment_options,
            default=default_equipment,
            key='emp_priority_equipment',
        )
    with f2:
        office_options = office_df['Office'].tolist() if not office_df.empty else []
        selected_offices = st.multiselect(
            'Office filter',
            office_options,
            default=[],
            key='emp_priority_offices',
            placeholder='All offices',
        )
    with f3:
        show_only_without_computer = st.checkbox(
            'Only without computer',
            value=True,
            key='emp_priority_without_computer',
        )

    priority_df = _build_priority_list(
        emp_map,
        office_df,
        selected_equipment,
        selected_offices,
        show_only_without_computer,
    )

    p1, p2, p3 = st.columns(3)
    with p1:
        offices_needing = int((office_df['Without Computer'] > 0).sum()) if not office_df.empty else 0
        st.metric('Offices Needing Computers', f'{offices_needing:,}')
    with p2:
        st.metric('Priority Employees Listed', f'{len(priority_df):,}')
    with p3:
        st.metric('Selected Equipment', f'{len(selected_equipment):,}')

    st.markdown('**Office Priority**')
    if office_df.empty:
        st.info('No employee records found in the inventory CSV.')
    else:
        st.dataframe(_format_office_table(office_df), use_container_width=True, height=260, hide_index=True)

    st.markdown('**Prioritized Employee List**')
    if priority_df.empty:
        st.info('No employees match the selected office and equipment filters.')
    else:
        page_size = st.selectbox('Show priority rows', [10, 25, 50, 100], key='emp_priority_page_size', index=1)
        total = len(priority_df)
        if page_size < total:
            page = st.number_input(
                'Priority page',
                1,
                max(1, (total + page_size - 1) // page_size),
                1,
                key='emp_priority_page',
            )
            start = (page - 1) * page_size
            end = min(start + page_size, total)
            st.dataframe(priority_df.iloc[start:end], use_container_width=True, height=430, hide_index=True)
            st.caption(f'Showing {start + 1}-{end} of {total} priority records')
        else:
            st.dataframe(priority_df, use_container_width=True, height=430, hide_index=True)

        export_col1, export_col2 = st.columns(2)
        with export_col1:
            st.download_button(
                'Download Priority CSV',
                to_csv(priority_df),
                'employee_equipment_priority.csv',
                'text/csv',
                use_container_width=True,
                key='emp_priority_csv',
            )
        with export_col2:
            st.download_button(
                'Download Priority Excel',
                to_excel(priority_df),
                'employee_equipment_priority.xlsx',
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                use_container_width=True,
                key='emp_priority_xlsx',
            )

    model = selected_model or get_selected_model()
    if model and not priority_df.empty:
        st.divider()
        render_section_header('AI Employee Recommendation')
        st.markdown(f'Select a prioritized employee to generate a personalized AI recommendation using **{model}**.')

        ai_options = priority_df.apply(
            lambda row: f"{row['Employee Name']} | {row['Office']} | Rank {row['Rank']}",
            axis=1,
        ).tolist()
        selected_emp = st.selectbox('Choose Employee', [''] + ai_options, key='emp_llm_select')
        if selected_emp:
            selected_index = ai_options.index(selected_emp)
            emp_row = priority_df.iloc[selected_index]
            emp_data = {
                'employee_name': emp_row.get('Employee Name', ''),
                'office_division': emp_row.get('Office', ''),
                'status_of_employment': emp_row.get('Status', ''),
                'nature_of_work': emp_row.get('Nature of Work', ''),
                'priority_score': emp_row.get('Priority Score', 0),
                'recommendation': emp_row.get('Recommended Equipment', ''),
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
