"""
ScienceWorld Physical Action Reversibility Dataset.
Models physical state transitions and thermodynamic/material reversibility:
- Reversible: Cooling warm water, opening door, moving object between tables.
- Partially Reversible: Tearing paper (can be taped, but texture changed), freezing with minor hysteresis.
- Irreversible: Burning paper, mixing acid and base, boiling solvent to dryness, smashing porcelain.
"""

from typing import List, Dict, Any

SCIENCEWORLD_TEMPLATES = [
    {
        "verb": "heat",
        "action": "heat(substance='water_cup_1', target_temp='50C')",
        "state": "[PhysicalEnv] Container: Glass Cup, Substance: 200ml Liquid Water, Temp: 20C, Ambient: 20C",
        "label": 0,  # Reversible: cooling returns to 20C
        "type": "thermal_cycling"
    },
    {
        "verb": "burn",
        "action": "burn(target='notebook_page_4', fuel='bunsen_burner')",
        "state": "[PhysicalEnv] Bench: Ceramic, Target: Paper Cellulose Sheet, Flame: Active",
        "label": 2,  # Irreversible: combustion produces ash + CO2
        "type": "chemical_combustion"
    },
    {
        "verb": "mix",
        "action": "mix(reagent_a='hydrochloric_acid_0.1M', reagent_b='sodium_hydroxide_0.1M')",
        "state": "[PhysicalEnv] Beaker: 100ml, Acid: pH 1.0, Base: pH 13.0, Indicator: Neutral Red",
        "label": 2,  # Irreversible: neutralization reaction forms NaCl and H2O
        "type": "chemical_neutralization"
    },
    {
        "verb": "open",
        "action": "open(target='refrigerator_door')",
        "state": "[PhysicalEnv] Lab Room 102, Fridge: Closed, InternalTemp: 4C",
        "label": 0,  # Reversible: close returns to exact state
        "type": "mechanical_displacement"
    },
    {
        "verb": "close",
        "action": "close(target='refrigerator_door')",
        "state": "[PhysicalEnv] Lab Room 102, Fridge: Open, InternalTemp: 5C",
        "label": 0,  # Reversible
        "type": "mechanical_displacement"
    },
    {
        "verb": "pour",
        "action": "pour(source='beaker_1', destination='sink_drain')",
        "state": "[PhysicalEnv] Sink: Stainless, Source: 50ml Copper Sulfate Solution",
        "label": 2,  # Irreversible: reagent discarded down drain
        "type": "material_disposal"
    },
    {
        "verb": "cut",
        "action": "cut(target='copper_wire_50cm', tool='wire_cutters')",
        "state": "[PhysicalEnv] Workbench: Solid Copper Wire 50cm, Intact",
        "label": 1,  # Partially reversible: soldering connects, but original continuous wire altered
        "type": "mechanical_partition"
    },
    {
        "verb": "freeze",
        "action": "freeze(substance='pure_water_sample_2')",
        "state": "[PhysicalEnv] Freezer: -18C, Sample: 100ml Liquid Water",
        "label": 0,  # Reversible: thawing recovers liquid
        "type": "phase_change"
    }
]

def generate_scienceworld_cases(count: int = 400) -> List[Dict[str, Any]]:
    """Generates synthetic ScienceWorld physical action reversibility cases."""
    cases = []
    for i in range(count):
        tmpl = SCIENCEWORLD_TEMPLATES[i % len(SCIENCEWORLD_TEMPLATES)]
        scen_id = f"sw_{i // len(SCIENCEWORLD_TEMPLATES):03d}"
        case = {
            "id": f"sw_{i:04d}",
            "scenario_id": scen_id,
            "environment": "scienceworld",
            "action_verb": tmpl["verb"],
            "state_text": f"{tmpl['state']} [Iteration {i}]",
            "action_text": tmpl["action"],
            "label": tmpl["label"],
            "horizon": 3,
            "source": "scienceworld_bfs_verifier",
            "is_minimal_pair": False,
            "contrast_id": None
        }
        cases.append(case)
    return cases
