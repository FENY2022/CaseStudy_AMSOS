"""
ICT-AMSOS Backend Logic — All original functions preserved + enhanced additions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

CURRENT_YEAR = 2026
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'models'
OUTPUT_DIR = BASE_DIR / 'outputs'
DESKTOP_COST = 50000
LAPTOP_COST = 55000

EQUIPMENT_COSTS = {
    'Desktop Computer': 50000,
    'Laptop Computer': 55000,
    'Printer': 15000,
    'Scanner': 12000,
    'Monitor': 8000,
    'UPS': 5000,
    'Projector': 25000,
    'Router': 3000,
    'Switch': 4000,
    'CCTV': 10000,
}

ALL_PROCUREMENT_ITEMS = [
    'Desktop Computer', 'Laptop Computer', 'Printer', 'Scanner',
    'Monitor', 'UPS', 'Projector', 'Router', 'Switch', 'CCTV'
]

SHARED_POLICIES = [
    ('CCTV', 4, ['CCTV / IP CAMERA', 'CCTV']),
    ('Projector', 1, ['LCD PROJECTORS', 'Projector']),
    ('Switch', 1, ['Network Switches', 'Switch']),
    ('Router', 1, ['Routers', 'Router']),
]

MAX_PRINTER_TARGET = 7


def compute_employee_priority(inv_df):
    unique_emps = inv_df.drop_duplicates(subset='employeeName')
    comps = inv_df[inv_df['equipmentType'].str.contains('Desktop|Laptop', case=False, na=False)]
    emps_with_computer = comps['employeeName'].dropna().unique()
    needs_computer = unique_emps[~unique_emps['employeeName'].isin(emps_with_computer)].copy()
    employees_no_pc = []
    for _, row in needs_computer.iterrows():
        employees_no_pc.append({
            'employee_name': row['employeeName'],
            'sex': row.get('sex', 'Unknown'),
            'office_division': row.get('officeDivision', 'Unknown'),
            'status_of_employment': row.get('statusOfEmployment', 'Unknown'),
            'nature_of_work': row.get('natureOfWork', 'Unknown'),
        })
    return pd.DataFrame(employees_no_pc)


def compute_employee_priority_score(df):
    if df.empty:
        return df
    def score(row):
        s = 0
        status = str(row['status_of_employment']).lower()
        nature = str(row['nature_of_work']).lower()
        if 'permanent' in status:
            s += 30
        elif 'casual' in status or 'cotractual' in status:
            s += 15
        elif 'contract' in status or 'job' in status or 'cos' in status:
            s += 10
        if 'technical' in nature:
            s += 30
        elif 'administrative' in nature or 'clerical' in nature:
            s += 20
        else:
            s += 10
        return s
    df['priority_score'] = df.apply(score, axis=1)
    def recommendation(row):
        score = row['priority_score']
        nature = str(row['nature_of_work']).lower()
        if score >= 50:
            return 'Assign Desktop Computer'
        elif score >= 30:
            return 'Assign Desktop or Laptop'
        else:
            return 'No Procurement - Shared workstation sufficient'
    df['recommendation'] = df.apply(recommendation, axis=1)
    df = df.sort_values('priority_score', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    return df


def compute_procurement(emp_priority):
    rec_counts = emp_priority['recommendation'].value_counts()
    emp_need_desktop = int(rec_counts.get('Assign Desktop Computer', 0))
    emp_need_either = int(rec_counts.get('Assign Desktop or Laptop', 0))
    emp_no_proc = int(rec_counts.get('No Procurement - Shared workstation sufficient', 0))
    need_computer = emp_priority[emp_priority['recommendation'].str.contains('Desktop|Laptop', na=False)]
    tech_mask = need_computer['nature_of_work'].str.contains('Technical', case=False, na=False)
    rec_desktop = int(need_computer[tech_mask].shape[0])
    rec_laptop = int(need_computer[~tech_mask].shape[0])
    desktop_budget = rec_desktop * DESKTOP_COST
    laptop_budget = rec_laptop * LAPTOP_COST
    total_budget = desktop_budget + laptop_budget
    proc = pd.DataFrame([{
        'category': 'Desktop Computer',
        'recommended_purchase': rec_desktop,
        'unit_cost': DESKTOP_COST,
        'estimated_budget': desktop_budget,
        'notes': f'Employees needing computer with Technical work: {rec_desktop}'
    }, {
        'category': 'Laptop Computer',
        'recommended_purchase': rec_laptop,
        'unit_cost': LAPTOP_COST,
        'estimated_budget': laptop_budget,
        'notes': f'Employees needing computer with Administrative/other work: {rec_laptop}'
    }, {
        'category': 'TOTAL',
        'recommended_purchase': rec_desktop + rec_laptop,
        'unit_cost': 0,
        'estimated_budget': total_budget,
        'notes': 'Total procurement budget for employees without computer'
    }])
    return proc


def compute_division_shortage(inv_df, div_df):
    unique_emps = inv_df[['employeeName', 'officeDivision']].drop_duplicates(subset='employeeName')
    comps = inv_df[inv_df['equipmentType'].str.contains('Desktop|Laptop', case=False, na=False)]
    comps_unique = comps[['employeeName', 'officeDivision']].drop_duplicates(subset='employeeName')
    comps_per_div = comps_unique.groupby('officeDivision').size().reset_index(name='assigned_computers')
    div_df['Count'] = pd.to_numeric(div_df['Count'], errors='coerce').fillna(0).astype(int)
    shortage = div_df.merge(comps_per_div, left_on='Division', right_on='officeDivision', how='left')
    shortage['assigned_computers'] = shortage['assigned_computers'].fillna(0).astype(int)
    shortage['shortage'] = (shortage['Count'] - shortage['assigned_computers']).clip(lower=0)
    shortage['recommended_procurement'] = shortage['shortage']
    shortage = shortage.sort_values('recommended_procurement', ascending=False).reset_index(drop=True)
    result = shortage[['Division', 'Count', 'assigned_computers', 'shortage']]
    result['replacement_count'] = 0
    return result


def normalize_office(name):
    if pd.isna(name):
        return name
    name = str(name).strip()
    if name.lower() == 'legal':
        return 'LEGAL'
    return name


def build_employee_equipment_map(inv_df):
    unique_emps = inv_df[['employeeName', 'officeDivision', 'sex', 'statusOfEmployment', 'natureOfWork']].drop_duplicates(subset='employeeName').reset_index(drop=True)
    unique_emps['officeDivision'] = unique_emps['officeDivision'].apply(normalize_office)
    emp_equip = inv_df.groupby('employeeName')['equipmentType'].apply(lambda x: set(x.dropna())).reset_index()
    emp_equip.columns = ['employeeName', 'equipment_set']
    result = unique_emps.merge(emp_equip, on='employeeName', how='left')
    result['equipment_set'] = result['equipment_set'].apply(lambda s: s if isinstance(s, set) else set())
    return result


def compute_office_equipment_counts(inv_df):
    df = inv_df.copy()
    df['officeDivision'] = df['officeDivision'].apply(normalize_office)
    offices = df['officeDivision'].unique()
    result = {}
    for office in offices:
        odf = df[df['officeDivision'] == office]
        counts = {}
        for item, _, variants in SHARED_POLICIES:
            pat = '|'.join(variants)
            counts[item] = int(odf[odf['equipmentType'].str.contains(pat, case=False, na=False)].shape[0])
        counts['Printer'] = int(odf[odf['equipmentType'].str.contains('Printer', case=False, na=False)].shape[0])
        unique_in_office = df[df['officeDivision'] == office]['employeeName'].unique()
        counts['Desktop Computer'] = int(df[(df['officeDivision'] == office) & (df['equipmentType'].str.contains('Desktop Computer', case=False, na=False))]['employeeName'].nunique())
        counts['Laptop Computer'] = int(df[(df['officeDivision'] == office) & (df['equipmentType'].str.contains('Laptop', case=False, na=False))]['employeeName'].nunique())
        counts['UPS'] = int(odf[odf['equipmentType'].str.contains('UPS', case=False, na=False)].shape[0])
        result[office] = counts
    return result


def compute_employee_computer_procurement(emp_map, selected_items):
    want_desktop = 'Desktop Computer' in selected_items
    want_laptop = 'Laptop Computer' in selected_items
    no_computer = emp_map[emp_map['equipment_set'].apply(lambda s: not any(v in s for v in ['Desktop Computer', 'Laptop Computer', 'Laptop Computers']))].copy()
    results = []
    for _, row in no_computer.iterrows():
        nature = str(row.get('natureOfWork', '')).lower()
        if want_desktop and (not want_laptop or 'technical' in nature):
            results.append({
                'employeeName': row['employeeName'],
                'office': row['officeDivision'],
                'item': 'Desktop Computer',
                'cost': EQUIPMENT_COSTS['Desktop Computer'],
                'reason': 'Employee has no assigned Desktop or Laptop computer.',
            })
        elif want_laptop:
            results.append({
                'employeeName': row['employeeName'],
                'office': row['officeDivision'],
                'item': 'Laptop Computer',
                'cost': EQUIPMENT_COSTS['Laptop Computer'],
                'reason': 'Employee has no assigned Desktop or Laptop computer.',
            })
    return pd.DataFrame(results)


def compute_shared_procurement(office_counts, selected_items):
    rows = []
    for item, max_allowed, variants in SHARED_POLICIES:
        if item not in selected_items:
            continue
        for office, counts in office_counts.items():
            current = counts.get(item, 0)
            recommend = int(max(0, max_allowed - current))
            if recommend > 0:
                if item == 'Printer':
                    explanation = (
                        f"The procurement policy allows a maximum of {max_allowed} shared printers per division. "
                        f"Since {office} currently has {current} printer(s), "
                        f"the AI recommends procuring {recommend} additional printer(s)."
                    )
                elif item == 'CCTV':
                    explanation = (
                        f"The procurement policy recommends at least {max_allowed} CCTV units per office for security coverage. "
                        f"Since {office} currently has {current} CCTV unit(s), "
                        f"the AI recommends procuring {recommend} additional unit(s)."
                    )
                else:
                    explanation = (
                        f"The procurement policy allows a maximum of {max_allowed} {item}(s) per division. "
                        f"Since {office} currently has {current}, "
                        f"the AI recommends procuring {recommend} unit(s)."
                    )
                rows.append({
                    'office': office,
                    'item': item,
                    'units': recommend,
                    'cost_per_unit': EQUIPMENT_COSTS.get(item, 0),
                    'reason': explanation,
                })
    return pd.DataFrame(rows)


def compute_proportional_printer_allocation(emp_map, office_counts):
    emp_per_division = emp_map['officeDivision'].value_counts().reset_index()
    emp_per_division.columns = ['office', 'unique_employees']
    largest_count = emp_per_division['unique_employees'].max()
    if largest_count == 0:
        return pd.DataFrame()
    largest_office = emp_per_division.loc[emp_per_division['unique_employees'].idxmax(), 'office']
    max_target = MAX_PRINTER_TARGET
    rows = []
    for _, row in emp_per_division.iterrows():
        office = row['office']
        emp_count = row['unique_employees']
        target = round((emp_count / largest_count) * max_target)
        target = max(1, target)
        existing = office_counts.get(office, {}).get('Printer', 0)
        recommend = max(0, target - existing)
        if office == largest_office:
            explanation = (
                f"{office} has the largest workforce with {emp_count} unique employees "
                f"and is assigned the maximum target of {max_target} shared printers."
            )
        else:
            pct = (emp_count / largest_count) * 100
            explanation = (
                f"{office} has {emp_count} unique employees "
                f"({pct:.0f}% of {largest_office}'s workforce), "
                f"resulting in a target allocation of {target} printers."
            )
        if existing > 0:
            explanation += (
                f" Since {office} already has {existing} printer(s), "
                f"the AI recommends procuring {recommend} additional printer(s)."
            )
        else:
            explanation += (
                f" Since {office} has no existing printers, "
                f"the AI recommends procuring all {recommend} printer(s)."
            )
        rows.append({
            'office': office,
            'item': 'Printer',
            'units': recommend,
            'cost_per_unit': EQUIPMENT_COSTS.get('Printer', 15000),
            'reason': explanation,
        })
    return pd.DataFrame(rows)


def compute_ups_procurement(emp_map, desktop_recs, selected_items):
    if 'UPS' not in selected_items or desktop_recs.empty:
        return pd.DataFrame()
    results = []
    for _, rec in desktop_recs.iterrows():
        emp_data = emp_map[emp_map['employeeName'] == rec['employeeName']]
        if not emp_data.empty and 'UPS' not in emp_data.iloc[0]['equipment_set']:
            results.append({
                'employeeName': rec['employeeName'],
                'office': rec['office'],
                'item': 'UPS',
                'cost': EQUIPMENT_COSTS['UPS'],
                'reason': f"Employee {rec['employeeName']} is recommended a Desktop Computer but currently has no UPS.",
            })
    return pd.DataFrame(results)


def policy_full_analysis(emp_map, office_counts, selected_items):
    desktop_recs = compute_employee_computer_procurement(emp_map, selected_items)
    shared_df = compute_shared_procurement(office_counts, selected_items)
    printer_df = compute_proportional_printer_allocation(emp_map, office_counts) if 'Printer' in selected_items else pd.DataFrame()
    desktop_only = desktop_recs[desktop_recs['item'] == 'Desktop Computer'] if not desktop_recs.empty else pd.DataFrame()
    ups_df = compute_ups_procurement(emp_map, desktop_only, selected_items)

    office_records = []
    for office, group in emp_map.groupby('officeDivision'):
        total_emps = len(group)
        w_desk = int(group['equipment_set'].apply(lambda s: 'Desktop Computer' in s).sum())
        w_lap = int(group['equipment_set'].apply(lambda s: any(v in s for v in ['Laptop Computer', 'Laptop Computers'])).sum())
        w_comp = int(group['equipment_set'].apply(lambda s: any(v in s for v in ['Desktop Computer', 'Laptop Computer', 'Laptop Computers'])).sum())
        wo_comp = total_emps - w_comp
        rec_items = {}
        for item in selected_items:
            if item in ('Desktop Computer', 'Laptop Computer'):
                cnt = int(desktop_recs[(desktop_recs['item'] == item) & (desktop_recs['office'] == office)].shape[0]) if not desktop_recs.empty else 0
            elif item == 'UPS':
                cnt = int(ups_df[ups_df['office'] == office].shape[0]) if not ups_df.empty else 0
            elif item == 'Printer':
                cnt = int(printer_df[(printer_df['office'] == office)]['units'].sum()) if not printer_df.empty else 0
            else:
                cnt = int(shared_df[(shared_df['item'] == item) & (shared_df['office'] == office)]['units'].sum()) if not shared_df.empty else 0
            rec_items[f'rec_{item}'] = cnt
        budget = sum(rec_items.get(f'rec_{item}', 0) * EQUIPMENT_COSTS.get(item, 0) for item in selected_items)
        o_counts = office_counts.get(office, {})
        existing_printers = o_counts.get('Printer', 0)
        existing_cctv = o_counts.get('CCTV', 0)
        office_records.append({
            'Office': office,
            'Total Employees': total_emps,
            'Employees with Desktop': w_desk,
            'Employees with Laptop': w_lap,
            'Employees with Computer': w_comp,
            'Employees without Computer': wo_comp,
            'Existing Printers': existing_printers,
            'Existing CCTV': existing_cctv,
            **rec_items,
            'Estimated Budget': budget,
        })
    offices_df = pd.DataFrame(office_records)
    if not offices_df.empty:
        offices_df = offices_df.sort_values('Employees without Computer', ascending=False).reset_index(drop=True)
        offices_df['Rank'] = range(1, len(offices_df) + 1)
        max_need = offices_df['Employees without Computer'].max()
        if max_need > 0:
            offices_df['Priority'] = offices_df['Employees without Computer'].apply(
                lambda n: 'HIGH' if n >= max_need * 0.5 else ('MEDIUM' if n >= max_need * 0.2 else 'LOW')
            )
        else:
            offices_df['Priority'] = 'LOW'

    proc_rows = []
    for _, r in desktop_recs.iterrows():
        proc_rows.append({'Employee': r['employeeName'], 'Office': r['office'], 'Item': r['item'], 'Type': 'Per-Employee', 'Quantity': 1, 'Unit Cost': r['cost'], 'Total Cost': r['cost'], 'Reason': r['reason']})
    for _, r in ups_df.iterrows():
        proc_rows.append({'Employee': r['employeeName'], 'Office': r['office'], 'Item': r['item'], 'Type': 'Per-Employee', 'Quantity': 1, 'Unit Cost': r['cost'], 'Total Cost': r['cost'], 'Reason': r['reason']})
    for _, r in printer_df.iterrows():
        total = r['units'] * r['cost_per_unit']
        proc_rows.append({'Employee': f'{r["office"]} (Office)', 'Office': r['office'], 'Item': r['item'], 'Type': 'Shared', 'Quantity': int(r['units']), 'Unit Cost': int(r['cost_per_unit']), 'Total Cost': int(total), 'Reason': r['reason']})
    for _, r in shared_df.iterrows():
        total = r['units'] * r['cost_per_unit']
        proc_rows.append({'Employee': f'{r["office"]} (Office)', 'Office': r['office'], 'Item': r['item'], 'Type': 'Shared', 'Quantity': int(r['units']), 'Unit Cost': int(r['cost_per_unit']), 'Total Cost': int(total), 'Reason': r['reason']})
    proc_list = pd.DataFrame(proc_rows)
    if not proc_list.empty:
        proc_list = proc_list.sort_values('Total Cost', ascending=False).reset_index(drop=True)

    summary = {}
    for item in selected_items:
        units = 0
        if not desktop_recs.empty:
            units += int(desktop_recs[desktop_recs['item'] == item].shape[0])
        if item == 'UPS' and not ups_df.empty:
            units += int(ups_df.shape[0])
        if not shared_df.empty:
            units += int(shared_df[shared_df['item'] == item]['units'].sum())
        if item == 'Printer' and not printer_df.empty:
            units += int(printer_df['units'].sum())
        cost = EQUIPMENT_COSTS.get(item, 0)
        summary[item] = {'units': units, 'unit_cost': cost, 'total': units * cost}
    total_units = sum(v['units'] for v in summary.values())
    total_budget = sum(v['total'] for v in summary.values())
    return desktop_recs, shared_df, ups_df, offices_df, proc_list, summary, total_units, total_budget, printer_df


def compute_priority_scores(emp_map):
    office_coverage = {}
    for office in emp_map['officeDivision'].unique():
        group = emp_map[emp_map['officeDivision'] == office]
        total = len(group)
        with_comp = group['equipment_set'].apply(lambda s: any(v in s for v in ['Desktop Computer', 'Laptop Computer', 'Laptop Computers'])).sum()
        office_coverage[office] = with_comp / total if total > 0 else 1.0
    avg_coverage = sum(office_coverage.values()) / len(office_coverage) if office_coverage else 1.0
    peripherals_set = {'Printer', 'Scanner', 'Monitor', 'UPS', 'CCTV / IP CAMERA', 'CCTV', 'Projector', 'Router', 'Switch'}
    results = []
    for _, row in emp_map.iterrows():
        equip_set = row['equipment_set']
        office = row['officeDivision']
        has_desktop = 'Desktop Computer' in equip_set
        has_laptop = any(v in equip_set for v in ['Laptop Computer', 'Laptop Computers'])
        has_computer = has_desktop or has_laptop
        if has_computer:
            results.append({
                'employeeName': row['employeeName'],
                'office': office,
                'natureOfWork': row.get('natureOfWork', ''),
                'equipment_set': equip_set,
                'priority_score': 0,
                'priority_reason': 'Already assigned a Desktop or Laptop.',
                'office_coverage_pct': office_coverage.get(office, 1.0),
            })
            continue
        candidates = []
        candidates.append((100, 'No Desktop or Laptop assigned.'))
        if not equip_set:
            candidates.append((90, 'No ICT equipment assigned.'))
        if equip_set and equip_set.issubset(peripherals_set):
            candidates.append((80, 'Only peripheral devices assigned.'))
        coverage = office_coverage.get(office, 1.0)
        if coverage < avg_coverage:
            candidates.append((70, f'Office has below-average ICT coverage ({coverage:.0%}).'))
        best = max(candidates, key=lambda x: x[0])
        results.append({
            'employeeName': row['employeeName'],
            'office': office,
            'natureOfWork': row.get('natureOfWork', ''),
            'equipment_set': equip_set,
            'priority_score': best[0],
            'priority_reason': best[1],
            'office_coverage_pct': office_coverage.get(office, 1.0),
        })
    return pd.DataFrame(results)


def optimize_budget(priority_df, emp_recs, ups_df, available_budget):
    merged = priority_df.merge(emp_recs[['employeeName', 'office', 'item', 'cost']], on=['employeeName', 'office'], how='inner')
    if merged.empty:
        return pd.DataFrame(), available_budget, 0
    merged = merged.sort_values(['priority_score', 'office_coverage_pct'], ascending=[False, True]).reset_index(drop=True)
    total_requirement = int(merged['cost'].sum())
    remaining = int(available_budget)
    rows = []
    for idx, (_, row) in enumerate(merged.iterrows()):
        cost = int(row['cost'])
        equip_set = row['equipment_set']
        current = ', '.join(sorted(equip_set)) if isinstance(equip_set, set) and equip_set else 'None'
        if cost <= remaining:
            remaining -= cost
            status = 'APPROVED'
        else:
            status = 'WAITING FOR NEXT BUDGET'
        rows.append({
            'Rank': idx + 1,
            'Priority Score': int(row['priority_score']),
            'Employee Name': row['employeeName'],
            'Office': row['office'],
            'Current Equipment': current,
            'Recommended Device': row['item'],
            'Estimated Cost': cost,
            'Reason': row['priority_reason'],
            'Status': status,
        })
    opt_df = pd.DataFrame(rows)
    if ups_df is not None and not ups_df.empty and remaining > 0:
        approved_emps = set(opt_df[opt_df['Status'] == 'APPROVED']['Employee Name'])
        ups_filtered = ups_df[ups_df['employeeName'].isin(approved_emps)]
        if not ups_filtered.empty:
            for _, row in ups_filtered.iterrows():
                cost = int(row['cost'])
                if cost <= remaining:
                    remaining -= cost
                    status = 'APPROVED'
                else:
                    status = 'WAITING FOR NEXT BUDGET'
                rows.append({
                    'Rank': len(rows) + 1,
                    'Priority Score': 0,
                    'Employee Name': row['employeeName'],
                    'Office': row['office'],
                    'Current Equipment': 'None',
                    'Recommended Device': row['item'],
                    'Estimated Cost': cost,
                    'Reason': 'Desktop recipient with no existing UPS.',
                    'Status': status,
                })
    opt_df = pd.DataFrame(rows)
    return opt_df, remaining, total_requirement


def load_models():
    import joblib
    clf = joblib.load(MODEL_DIR / 'replacement_model.pkl')
    reg = joblib.load(MODEL_DIR / 'health_score_model.pkl')
    encoder = joblib.load(MODEL_DIR / 'label_encoder.pkl')
    feature_cols = joblib.load(MODEL_DIR / 'feature_columns.pkl')
    return clf, reg, encoder, feature_cols


def load_data():
    import streamlit as st
    inv = pd.read_csv(DATA_DIR / 'inv_inventory.csv', low_memory=False)
    repair = pd.read_csv(DATA_DIR / 'repairhistory.csv', low_memory=False)
    div = pd.read_csv(DATA_DIR / 'division_counts.csv', low_memory=False)
    repl = pd.read_csv(OUTPUT_DIR / 'replacement_priority.csv') if (OUTPUT_DIR / 'replacement_priority.csv').exists() else None
    emp = compute_employee_priority(inv)
    if len(emp) > 0:
        emp = compute_employee_priority_score(emp)
    div_short = compute_division_shortage(inv, div)
    proc = compute_procurement(emp) if len(emp) > 0 else None
    return inv, repair, div, repl, emp, div_short, proc


def get_ollama_models():
    try:
        import ollama
        result = ollama.list()
        models = []
        for m in result.models:
            family = m.details.family if m.details and m.details.family else ''
            if family not in ('nomic-bert', ''):
                models.append(m.model)
        return models
    except Exception:
        return []


# =====================================================================
# ENHANCED FUNCTIONS (new, not modifying original)
# =====================================================================

def compute_enhanced_priority_scores(emp_map, repl_df=None):
    """
    Enhanced priority scoring with all tiers: 100, 90, 80, 70, 60, 50.
    Also considers employees WITH computers for replacement scoring.
    """
    office_coverage = {}
    for office in emp_map['officeDivision'].unique():
        group = emp_map[emp_map['officeDivision'] == office]
        total = len(group)
        with_comp = group['equipment_set'].apply(
            lambda s: any(v in s for v in ['Desktop Computer', 'Laptop Computer', 'Laptop Computers'])
        ).sum()
        office_coverage[office] = with_comp / total if total > 0 else 1.0
    avg_coverage = sum(office_coverage.values()) / len(office_coverage) if office_coverage else 1.0
    peripherals_set = {'Printer', 'Scanner', 'Monitor', 'UPS', 'CCTV / IP CAMERA', 'CCTV', 'Projector', 'Router', 'Switch'}
    results = []
    for _, row in emp_map.iterrows():
        equip_set = row['equipment_set']
        office = row['officeDivision']
        has_desktop = 'Desktop Computer' in equip_set
        has_laptop = any(v in equip_set for v in ['Laptop Computer', 'Laptop Computers'])
        has_computer = has_desktop or has_laptop
        if not has_computer:
            candidates = [(100, 'Employee has no Desktop or Laptop.')]
            if not equip_set:
                candidates.append((90, 'Employee has no ICT equipment.'))
            if equip_set and equip_set.issubset(peripherals_set):
                candidates.append((80, f'Employee only owns peripheral devices: {", ".join(sorted(equip_set))}.'))
            coverage = office_coverage.get(office, 1.0)
            if coverage < avg_coverage:
                candidates.append((70, f'Office ICT coverage ({coverage:.0%}) is below agency average ({avg_coverage:.0%}).'))
            best = max(candidates, key=lambda x: x[0])
            results.append({
                'employeeName': row['employeeName'],
                'office': office,
                'natureOfWork': row.get('natureOfWork', ''),
                'equipment_set': equip_set,
                'priority_score': best[0],
                'priority_reason': best[1],
                'office_coverage_pct': office_coverage.get(office, 1.0),
                'needs_procurement': True,
            })
        else:
            # Employee has a computer — check age/condition for replacement need
            if repl_df is not None:
                emp_assets = repl_df[repl_df['actualUser'] == row['employeeName']] if 'actualUser' in repl_df.columns else pd.DataFrame()
                if not emp_assets.empty:
                    oldest_age = emp_assets['equipment_age'].max() if 'equipment_age' in emp_assets.columns else 0
                    worst_health = emp_assets['asset_health_score'].min() if 'asset_health_score' in emp_assets.columns else 100
                    worst_priority = emp_assets['predicted_priority'].iloc[0] if 'predicted_priority' in emp_assets.columns else ''
                    has_old = oldest_age > 5 if oldest_age else False
                    has_poor = worst_health < 40 if worst_health else False
                    if has_old:
                        results.append({
                            'employeeName': row['employeeName'],
                            'office': office,
                            'natureOfWork': row.get('natureOfWork', ''),
                            'equipment_set': equip_set,
                            'priority_score': 60,
                            'priority_reason': f'Computer is over {int(oldest_age)} years old (exceeds 5-year useful life).',
                            'office_coverage_pct': office_coverage.get(office, 1.0),
                            'needs_procurement': True,
                        })
                    elif has_poor:
                        results.append({
                            'employeeName': row['employeeName'],
                            'office': office,
                            'natureOfWork': row.get('natureOfWork', ''),
                            'equipment_set': equip_set,
                            'priority_score': 50,
                            'priority_reason': f'Computer condition is Poor (health score: {worst_health:.0f}/100).',
                            'office_coverage_pct': office_coverage.get(office, 1.0),
                            'needs_procurement': True,
                        })
                    else:
                        results.append({
                            'employeeName': row['employeeName'],
                            'office': office,
                            'natureOfWork': row.get('natureOfWork', ''),
                            'equipment_set': equip_set,
                            'priority_score': 0,
                            'priority_reason': 'Already assigned a Desktop or Laptop.',
                            'office_coverage_pct': office_coverage.get(office, 1.0),
                            'needs_procurement': False,
                        })
                else:
                    results.append({
                        'employeeName': row['employeeName'],
                        'office': office,
                        'natureOfWork': row.get('natureOfWork', ''),
                        'equipment_set': equip_set,
                        'priority_score': 0,
                        'priority_reason': 'Already assigned a Desktop or Laptop.',
                        'office_coverage_pct': office_coverage.get(office, 1.0),
                        'needs_procurement': False,
                    })
            else:
                results.append({
                    'employeeName': row['employeeName'],
                    'office': office,
                    'natureOfWork': row.get('natureOfWork', ''),
                    'equipment_set': equip_set,
                    'priority_score': 0,
                    'priority_reason': 'Already assigned a Desktop or Laptop.',
                    'office_coverage_pct': office_coverage.get(office, 1.0),
                    'needs_procurement': False,
                })
    return pd.DataFrame(results)


# =====================================================================
# EXPLAINABLE AI ASSET HEALTH ASSESSMENT
# =====================================================================

ISSUE_KEYWORDS = {
    'Windows Update': r'(?i)(windows\s*update|windows\s+recovery|os\s+upgrade|operating\s+system)',
    'RAM Upgrade': r'(?i)(ram\s*upgrade|memory\s*upgrade|added\s*ram|low\s*memory|insufficient\s*memory)',
    'Hard Disk Replacement': r'(?i)(hard\s*disk|hdd|ssd\s*replace|storage\s*replace|blue\s*screen.*(?:ssd|hdd|disk)|replac(e|ed)\s+(?:ssd|hard\s*drive))',
    'Motherboard Replacement': r'(?i)(motherboard|mobo|system\s*board)',
    'Power Supply Replacement': r'(?i)(power\s*supply|psu|power\s*unit)',
    'Monitor Replacement': r'(?i)(monitor|display|vga\s*cable|loose\s*connection.*monitor|no\s*display)',
    'Network Issue': r'(?i)(network|ethernet|wifi|internet|connectivity|lan)',
    'Software Installation': r'(?i)(install|microsoft\s*365|office\s*license|application)',
    'Virus Infection': r'(?i)(virus|malware|infection|threat)',
    'Printer Driver Installation': r'(?i)(printer.*driver|driver.*printer|printer.*install|print.*driver)',
    'Cleaning': r'(?i)(cleaning|clean|dust|debris|preventive\s*maintenance|airflow)',
    'Operating System Reinstallation': r'(?i)(reformat|reinstall.*os|os.*reinstall|fresh\s*install|reinstallation)',
}

REPAIR_STATUS_WEIGHTS = {
    'Operational': 1.0,
    'For Repair': -0.5,
    'Not Operational': -1.0,
    'Unserviceable': -1.0,
    'PREVIOUSLY ISSUED': -0.3,
    'Previously issued': -0.3,
    'Computer Case only': -0.2,
}

RECURRING_ISSUE_THRESHOLD = 2


def _parse_shelf_life(val):
    if pd.isna(val):
        return 'Unknown'
    v = str(val).strip().lower()
    if 'beyond' in v:
        return 'Beyond'
    return 'Within'


def _compute_repair_score(repairs_count):
    if repairs_count <= 1:
        return 90, 'Excellent'
    elif repairs_count <= 3:
        return 75, 'Good'
    elif repairs_count <= 5:
        return 50, 'Fair'
    else:
        return 25, 'Critical'


def _compute_remarks_score(remarks_statuses):
    if not remarks_statuses:
        return 50
    scores = []
    for s in remarks_statuses:
        s_clean = str(s).strip()
        if s_clean in REPAIR_STATUS_WEIGHTS:
            scores.append(REPAIR_STATUS_WEIGHTS[s_clean])
        else:
            scores.append(0)
    avg = sum(scores) / len(scores) if scores else 0
    return max(0, min(100, 50 + avg * 50))


def _compute_shelf_life_penalty(shelf_life):
    parsed = _parse_shelf_life(shelf_life)
    if parsed == 'Beyond':
        return -10
    return 0


def _compute_age_penalty(year_acquired, equipment_type):
    if pd.isna(year_acquired):
        return 0
    try:
        year = int(float(year_acquired))
    except (ValueError, TypeError):
        return 0
    CURRENT_YEAR = 2026
    age = CURRENT_YEAR - year
    desktop_laptop = str(equipment_type) if not pd.isna(equipment_type) else ''
    if age >= 6:
        return -15
    elif age >= 4:
        return -5
    return 0


def _identify_recurring_issues(remarks_list):
    issues = []
    issue_counts = {}
    for r in remarks_list:
        if pd.isna(r):
            continue
        r_str = str(r)
        for issue, pattern in ISSUE_KEYWORDS.items():
            import re
            if re.search(pattern, r_str):
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
    recurring = {k: v for k, v in issue_counts.items() if v >= RECURRING_ISSUE_THRESHOLD}
    all_found = list(issue_counts.keys())
    return all_found, recurring


def _generate_ai_summary(remarks_filtered):
    if not remarks_filtered:
        return 'No repair history available for this asset.'
    all_text = ' '.join([str(r) for r in remarks_filtered if not pd.isna(r)])
    if not all_text.strip() or all_text.strip() in ('No remarks recorded', 'Completed', ''):
        return 'No significant repair details recorded.'

    issues_found, recurring = _identify_recurring_issues(remarks_filtered)
    has_recurring = len(recurring) > 0

    if 'RAM' in all_text and ('upgrad' in all_text.lower() or 'slow' in all_text.lower()):
        return 'The device experienced slow performance due to insufficient memory. The Help Desk upgraded the RAM and performed preventive maintenance. The issue was successfully resolved.'
    if 'blue screen' in all_text.lower() or 'ssd' in all_text.lower() and 'replace' in all_text.lower():
        return 'The device encountered critical system errors due to a failing storage device. The Help Desk replaced the SSD and reinstalled the necessary applications. The system is now stable.'
    if 'printer' in all_text.lower() and ('driver' in all_text.lower() or 'install' in all_text.lower()):
        return 'The device required printer driver installation. The Help Desk configured the printer software and verified proper functionality.'
    if 'windows' in all_text.lower() and ('update' in all_text.lower() or 'recovery' in all_text.lower()):
        return 'The device required operating system updates. The Help Desk performed Windows Update and system recovery procedures to restore normal operation.'
    if 'clean' in all_text.lower() or 'dust' in all_text.lower():
        return 'Preventive maintenance was performed on the device. The Help Desk cleaned internal components to remove dust and improve airflow and cooling.'
    if 'reset' in all_text.lower() or 'ink pad' in all_text.lower():
        return 'The printer required maintenance. The Help Desk performed a reset of the printer ink pad counter and verified proper functionality.'
    if 'ethernet' in all_text.lower() or 'wifi' in all_text.lower() or 'network' in all_text.lower() or 'dongle' in all_text.lower():
        return 'The device experienced network connectivity issues. The Help Desk diagnosed the hardware problem and replaced the affected network component.'
    if 'loose' in all_text.lower() or 'vga' in all_text.lower() or 'cable' in all_text.lower():
        return 'The device had a loose cable connection causing intermittent display issues. The Help Desk secured all connections and verified proper operation.'
    if 'beyond' in all_text.lower() and 'useful' in all_text.lower():
        return 'The device has exceeded its useful life and is experiencing hardware failures due to aging components. The Help Desk recommends replacement to ensure reliable operation.'

    parts = []
    if issues_found:
        parts.append(f"The Help Desk addressed the following issues: {', '.join(issues_found)}.")
    if has_recurring:
        recurring_str = ', '.join(recurring.keys())
        parts.append(f"Recurring issues detected: {recurring_str}.")
    if not parts:
        return 'The Help Desk performed routine maintenance and repairs. The device is now operational.'
    return ' '.join(parts)


def _determine_risk_level(health_score, repairs_count, remarks_statuses, recurring_issues):
    has_recurring = len(recurring_issues) > 0
    has_remarks_negative = any(
        str(s).strip() in ('For Repair', 'Not Operational', 'Unserviceable')
        for s in remarks_statuses if not pd.isna(s)
    )
    all_remarks_critical = all(
        str(s).strip() in ('Not Operational', 'Unserviceable')
        for s in remarks_statuses if not pd.isna(s) and str(s).strip() != ''
    ) if remarks_statuses else False

    if all_remarks_critical:
        return '🔴 Critical', 10
    if repairs_count >= 6 or has_recurring or has_remarks_negative:
        return '🔴 Critical', 20
    if repairs_count == 5:
        return '🟠 High Risk', 35
    if repairs_count >= 3 and health_score < 60:
        return '🟠 High Risk', 40
    if repairs_count in (3, 4):
        return '🟡 Moderate', 60
    if repairs_count <= 2 and not has_remarks_negative:
        return '🟢 Healthy', 85
    return '🟢 Healthy', 80


def _generate_recommendation(risk_level_label, equipment_type, issues_found, health_score):
    if risk_level_label == '🔴 Critical':
        if any('Hard Disk' in k or 'Motherboard' in k or 'Power Supply' in k for k in issues_found):
            return 'Replace Storage Device'
        return 'Prepare for Replacement'
    if risk_level_label == '🟠 High Risk':
        return 'Schedule Preventive Maintenance'
    if risk_level_label == '🟡 Moderate':
        return 'Continue Monitoring'
    return 'Continue Monitoring'


def _generate_explanation(record, equipment_type, actual_user):
    prop_no = record.get('propertyNumber', 'Unknown')
    repairs_count = record.get('repairs_count', 0)
    health_score = record.get('health_score', 50)
    risk_label = record.get('risk_label', '🟢 Healthy')
    issues_found = record.get('issues_found', [])
    remarks_status = record.get('current_status', 'Operational')
    shelf_life = record.get('shelf_life', '')
    year_acquired = record.get('year_acquired', '')

    equip_name = equipment_type if not pd.isna(equipment_type) else 'equipment'
    user = actual_user if not pd.isna(actual_user) else 'the assigned user'

    explanation_parts = []

    if repairs_count == 0:
        explanation_parts.append(
            f"This {equip_name} assigned to {user} has no repair history, indicating reliable performance."
        )
    else:
        explanation_parts.append(
            f"This {equip_name} assigned to {user} has been repaired {repairs_count} time(s)."
        )

    shelf_parsed = _parse_shelf_life(shelf_life)
    if shelf_parsed == 'Beyond':
        explanation_parts.append(
            f"The asset is beyond its expected shelf life, increasing the likelihood of component degradation."
        )

    if issues_found:
        issue_str = ', '.join(issues_found)
        explanation_parts.append(
            f"Repair records indicate issues including {issue_str}."
        )

    if repairs_count > 0:
        if repairs_count >= 4:
            explanation_parts.append(
                f"The high frequency of repairs suggests the asset may be nearing end-of-life."
            )
        else:
            explanation_parts.append(
                f"The repair frequency is within acceptable limits."
            )

    explanation_parts.append(
        f"The AI classifies this asset as {risk_label} with a health score of {health_score:.0f}/100."
    )

    return ' '.join(explanation_parts)


def compute_asset_health_analysis(inv_df, repair_df):
    results = []
    inv_copy = inv_df.copy()

    for _, asset in inv_copy.iterrows():
        prop_no = asset.get('propertyNumber', asset.get('Property No', ''))
        actual_user = asset.get('actualUser', asset.get('Actual User', ''))
        equipment_type = asset.get('equipmentType', asset.get('Equipment Type', ''))
        year_acquired = asset.get('yearAcquired', asset.get('Year Acquired', ''))
        shelf_life = asset.get('shelfLife', asset.get('Shelf Life', ''))
        employee = asset.get('employeeName', '')
        office = asset.get('officeDivision', '')
        remarks = asset.get('remarks', '')

        asset_repairs = repair_df[
            repair_df['Property No'].astype(str).str.strip() == str(prop_no).strip()
        ].copy() if 'Property No' in repair_df.columns else pd.DataFrame()

        if asset_repairs.empty:
            asset_repairs = repair_df[
                repair_df['Actual User'].astype(str).str.contains(
                    str(actual_user).strip(), case=False, na=False
                )
            ].copy() if 'Actual User' in repair_df.columns else pd.DataFrame()

        repairs_count = int(asset_repairs['Total Times Repaired'].iloc[0]) if not asset_repairs.empty and 'Total Times Repaired' in asset_repairs.columns else 0

        remarks_list = asset_repairs['Remarks / Action Taken'].tolist() if not asset_repairs.empty and 'Remarks / Action Taken' in asset_repairs.columns else []
        remarks_statuses = asset_repairs['Remarks'].tolist() if not asset_repairs.empty and 'Remarks' in asset_repairs.columns else []

        current_status = str(remarks).strip() if not pd.isna(remarks) else 'Operational'
        if remarks_statuses:
            non_empty = [s for s in remarks_statuses if not pd.isna(s) and str(s).strip() not in ('', 'N/A')]
            if non_empty:
                current_status = str(non_empty[-1]).strip()

        issues_found, recurring_issues = _identify_recurring_issues(remarks_list)

        repair_score, repair_rating = _compute_repair_score(repairs_count)
        remarks_score = _compute_remarks_score(remarks_statuses)
        shelf_penalty = _compute_shelf_life_penalty(shelf_life)
        age_penalty = _compute_age_penalty(year_acquired, equipment_type)

        health_score = max(0, min(100,
            repair_score * 0.40 +
            remarks_score * 0.25 +
            (100 + shelf_penalty) * 0.20 +
            (100 + age_penalty) * 0.15
        ))

        risk_label, risk_score = _determine_risk_level(
            health_score, repairs_count, remarks_statuses, recurring_issues
        )

        recommendation = _generate_recommendation(risk_label, equipment_type, issues_found, health_score)
        ai_summary = _generate_ai_summary(remarks_list)

        record = {
            'propertyNumber': prop_no,
            'repairs_count': repairs_count,
            'health_score': round(health_score, 1),
            'repair_rating': repair_rating,
            'risk_label': risk_label,
            'risk_score': risk_score,
            'issues_found': issues_found,
            'recurring_issues': recurring_issues,
            'ai_summary': ai_summary,
            'recommendation': recommendation,
            'current_status': current_status,
            'shelf_life': shelf_life,
            'year_acquired': year_acquired,
        }

        record['explanation'] = _generate_explanation(
            record, equipment_type, actual_user
        )

        record['employee'] = employee
        record['office'] = office
        record['equipment_type'] = equipment_type
        record['actual_user'] = actual_user
        record['remarks_history'] = remarks_list
        record['date_recorded_list'] = asset_repairs['Date Recorded'].tolist() if not asset_repairs.empty and 'Date Recorded' in asset_repairs.columns else []
        record['action_staff_list'] = asset_repairs['Action Staff'].tolist() if not asset_repairs.empty and 'Action Staff' in asset_repairs.columns else []

        results.append(record)

    df = pd.DataFrame(results)

    healthy = len(df[df['risk_label'] == '🟢 Healthy'])
    moderate = len(df[df['risk_label'] == '🟡 Moderate'])
    high_risk = len(df[df['risk_label'] == '🟠 High Risk'])
    critical = len(df[df['risk_label'] == '🔴 Critical'])

    return df, {
        'healthy': healthy,
        'moderate': moderate,
        'high_risk': high_risk,
        'critical': critical,
        'total': len(df),
    }


def generate_ai_summary_text(total_unique_emps, num_offices, emps_without_computer, offices_needing, available_budget, desktop_count, laptop_count, printer_count, total_budget):
    lines = [
        f"The AI analyzed **{total_unique_emps:,}** unique employees across all offices.",
        "",
        "Duplicate employee records were automatically consolidated.",
        "",
        f"**{emps_without_computer}** employees require ICT equipment.",
        f"**{offices_needing}** offices require procurement.",
        "",
        "Based on the available budget, the system recommends purchasing:",
    ]
    if desktop_count > 0:
        lines.append(f"- **{desktop_count}** Desktop Computers")
    if laptop_count > 0:
        lines.append(f"- **{laptop_count}** Laptop Computers")
    if printer_count > 0:
        lines.append(f"- **{printer_count}** Printers")
    lines.append("")
    lines.append("Priority was given to employees without any Desktop or Laptop.")
    return '\n'.join(lines)
