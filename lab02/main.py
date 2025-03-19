from searchMPP import SearcherMPP
from stripsForwardPlanner import Forward_STRIPS
from stripsProblem import Planning_problem, STRIPS_domain, Strips

# minerals = {"mineral_field_a"}
sectors = {"sector_a", "sector_b"}
boolean = {True, False}
building = {"", "Barracks"}
mineralField = {"", "Minerals"}


collect_minerals_actions = list(
    map(
        lambda x: Strips(
            f"collect_minerals_{x}",
            {"HasMinerals": False, f"Minerals_{x}": "Minerals"},
            {"HasMinerals": True, f"Minerals_{x}": ""},
        ),
        range(1, 3)
    )
)

starcraft_domain = STRIPS_domain(
    feature_domain_dict={
        "Minerals_1": building,
        "Minerals_2": building,
        "HasMinerals": boolean,
    },
    actions={
        *collect_minerals_actions
    },
)

collect_materials_problem = Planning_problem(
    prob_domain=starcraft_domain,
    initial_state={
        "HasMinerals": False,
        "Minerals_1": "Minerals",
        "Minerals_2": "Minerals",
    },
    goal={"HasMinerals": True},
)

s1 = SearcherMPP(Forward_STRIPS(collect_materials_problem))  # A*
s1.search()  # find another plan
