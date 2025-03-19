from searchMPP import SearcherMPP
from stripsForwardPlanner import Forward_STRIPS
from stripsProblem import Planning_problem, STRIPS_domain, Strips

# minerals = {"mineral_field_a"}
sectors = {"sector_a", "sector_b"}
boolean = {True, False}
building = {"", "Barracks", "Depot"}
mineralField = {"", "Minerals"}


collect_minerals_actions = list(
    map(
        lambda x: Strips(
            f"collect_minerals_{x}",
            {"HasMinerals": False, f"Minerals_{x}": "Minerals"},
            {"HasMinerals": True, f"Minerals_{x}": ""},
        ),
        range(1, 3),
    )
)

starcraft_domain = STRIPS_domain(
    feature_domain_dict={
        "Minerals_1": mineralField,
        "Minerals_2": mineralField,
        "Sector_1": building,
        "HasMinerals": boolean,
    },
    actions={
        *collect_minerals_actions,
        Strips(
            "Build_Barracks_1",
            {"HasMinerals": True, "Sector_1": ""},
            {"HasMinerals": False, "Sector_1": "Barracks"},
        ),
    },
)

collect_materials_problem = Planning_problem(
    prob_domain=starcraft_domain,
    initial_state={
        "HasMinerals": False,
        "Minerals_1": "Minerals",
        "Minerals_2": "Minerals",
        "Sector_1": ""
    },
    goal={"Sector_1": "Barracks"},
)

s1 = SearcherMPP(Forward_STRIPS(collect_materials_problem))  # A*
s1.search()  # find another plan
