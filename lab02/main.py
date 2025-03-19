from aipython.stripsForwardPlanner import Forward_STRIPS
from aipython.stripsProblem import Planning_problem, STRIPS_domain, Strips

minerals = {"mineral_field_a", "mineral_field_b"}
sectors = {"sector_a", "sector_b"}
boolean = {True, False}
starcraft_domain = STRIPS_domain(
    feature_domain_dict={
        "Mineral": minerals,
        "Sector": sectors,
        "Area": minerals | sectors,
        "ScvLocation": minerals | sectors,
        # "HasTank": boolean,
        # "HasMarine": boolean,
        "HasMinerals": boolean,
        # "Barracks": None | sectors,
        # "Buildings": {"barracks", "factory", "starport"}
    },
    actions={
        Strips(
            "collect_minerals",
            preconds={"ScvLocation": minerals, "HasMinerals": False},
            effects={"HasMinerals": True},
        ),
    },
)

# Definicje problemów
collect_materials_problem = Planning_problem(
    prob_domain=starcraft_domain,
    initial_state={
        "Mineral": minerals,
        "Sector": sectors,
        "Area": minerals | sectors,
        "ScvLocation": "mineral_field_a",
        "HasMinerals": False,
    },
    goal={"HasMinerals": True},
)


# result_barracks = barracks_problem.solve()
# result_marine = marine_problem.solve()
# result_problem = tank_problem.solve()
solver = Forward_STRIPS(collect_materials_problem)
