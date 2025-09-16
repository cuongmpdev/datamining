from __future__ import annotations

import itertools
from typing import Any, Dict, List, Tuple, Set


def get_equivalence_classes(rows: List[Dict[str, Any]], attrs: List[str]) -> List[List[int]]:
    if not attrs:
        return [[i] for i in range(len(rows))]
    
    groups: Dict[Tuple[Any, ...], List[int]] = {}
    for idx, row in enumerate(rows):
        key = tuple(row[a] for a in attrs)
        groups.setdefault(key, []).append(idx)
    
    return list(groups.values())


def is_reduct(rows: List[Dict[str, Any]], attrs: List[str], decision_attr: str) -> bool:
    if not attrs:
        return False
    
    # Group rows by conditional attributes
    groups: Dict[Tuple[Any, ...], List[int]] = {}
    for idx, row in enumerate(rows):
        key = tuple(row[a] for a in attrs)
        groups.setdefault(key, []).append(idx)
    
    # Check if each group has consistent decisions
    for group_indices in groups.values():
        decisions = set(rows[i][decision_attr] for i in group_indices)
        if len(decisions) > 1:  # Inconsistent decisions in this group
            return False
    
    return True


def find_all_reducts(rows: List[Dict[str, Any]], cond_attrs: List[str], decision_attr: str) -> Tuple[List[List[str]], Set[str]]:
    reducts: List[List[str]] = []
    
    # Try all possible combinations from smallest to largest
    for r in range(1, len(cond_attrs) + 1):
        for combo in itertools.combinations(cond_attrs, r):
            combo_list = list(combo)
            
            if is_reduct(rows, combo_list, decision_attr):
                # Check if this combination is minimal
                is_minimal = True
                for k in range(1, len(combo)):
                    for sub_combo in itertools.combinations(combo, k):
                        if is_reduct(rows, list(sub_combo), decision_attr):
                            is_minimal = False
                            break
                    if not is_minimal:
                        break
                
                if is_minimal:
                    reducts.append(combo_list)
    
    # Find core (intersection of all reducts)
    core: Set[str] = set(reducts[0]) if reducts else set()
    for reduct in reducts[1:]:
        core &= set(reduct)
    
    return reducts, core


def positive_region(rows: List[Dict[str, Any]], cond_attrs: List[str], decision_attr: str) -> Set[int]:
    pos: Set[int] = set()
    if not cond_attrs:
        return pos
    
    eq_classes = get_equivalence_classes(rows, cond_attrs)
    for class_indices in eq_classes:
        decisions = set(rows[i][decision_attr] for i in class_indices)
        if len(decisions) == 1:  # Consistent decision
            pos.update(class_indices)
    
    return pos


def dependency_degree(rows: List[Dict[str, Any]], cond_attrs: List[str], decision_attr: str) -> float:
    if not rows:
        return 0.0
    pos = positive_region(rows, cond_attrs, decision_attr)
    return len(pos) / len(rows)


def quick_reduct(rows: List[Dict[str, Any]], cond_attrs: List[str], decision_attr: str) -> Dict[str, Any]:
    if not rows or not cond_attrs:
        return {
            "reduct": [],
            "all_reducts": [],
            "core": [],
            "gamma_R": 0.0,
            "gamma_C": 0.0,
            "order": [],
            "equivalence_classes": [],
        }
    
    # Calculate initial dependency degree for all conditional attributes
    gamma_C = dependency_degree(rows, cond_attrs, decision_attr)
    
    # Find all minimal reducts and core using exhaustive search
    all_reducts, core = find_all_reducts(rows, cond_attrs, decision_attr)
    
    # Select the first (or smallest) reduct as the primary reduct
    primary_reduct = all_reducts[0] if all_reducts else []
    
    # Calculate dependency degree for the primary reduct
    gamma_R = dependency_degree(rows, primary_reduct, decision_attr)
    
    # Create order information (showing how reduct was built)
    # For comprehensive search, we show the final reduct composition
    order = [(attr, gamma_R) for attr in primary_reduct]
    
    # Get equivalence classes for visualization
    eq_classes = get_equivalence_classes(rows, primary_reduct)
    
    # Format equivalence classes for better readability
    formatted_eq_classes = []
    for i, class_indices in enumerate(eq_classes):
        class_objects = []
        class_decisions = set()
        
        for idx in class_indices:
            # Create object representation
            obj_values = {attr: rows[idx][attr] for attr in primary_reduct}
            obj_values['decision'] = rows[idx][decision_attr]
            obj_values['index'] = idx
            class_objects.append(obj_values)
            class_decisions.add(rows[idx][decision_attr])
        
        formatted_eq_classes.append({
            'class_id': i + 1,
            'objects': class_objects,
            'size': len(class_indices),
            'decisions': list(class_decisions),
            'is_consistent': len(class_decisions) == 1
        })
    
    return {
        "reduct": primary_reduct,
        "all_reducts": all_reducts,
        "core": list(core),
        "gamma_R": gamma_R,
        "gamma_C": gamma_C,
        "order": order,
        "equivalence_classes": formatted_eq_classes,
        "total_reducts_found": len(all_reducts),
        "reduct_summary": {
            "primary_reduct_size": len(primary_reduct),
            "core_size": len(core),
            "dependency_improvement": gamma_R - (0.0 if not primary_reduct else dependency_degree(rows, [], decision_attr)),
            "is_complete_reduct": gamma_R >= gamma_C - 1e-12
        }
    }

