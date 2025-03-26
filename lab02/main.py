from searchMPP import SearcherMPP
from stripsForwardPlanner import Forward_STRIPS
from stripsProblem import Planning_problem, STRIPS_domain, Strips

# minerals = {"mineral_field_a"}
sectors = set(map(lambda x: f"Sector_{x}", range(1, 3)))
boolean = {True, False}
building = {"", "Barracks", "Depot"}
mineralField = {"", "Minerals"}


collect_minerals_actions = list(
    map(
        lambda x: Strips(
            f"collect_minerals_{x}",
            {
                "HasMinerals": False,
                f"Minerals_{x}": "Minerals",
                "SCV_location": f"Minerals_{x}",
            },
            {
                "HasMinerals": True,
                f"Minerals_{x}": "",
            },
        ),
        range(1, 3),
    )
)

move_scv_actions = []


mineral_fields = set(map(lambda x: f"Minerals_{x}", range(1, 3)))
locations = mineral_fields | sectors

for l_1 in locations:
    for l_2 in locations:
        if l_1 != l_2:
            move_scv_actions.append(
                Strips(
                    f"move_scv_from_{l_1}_to_{l_2}",
                    {"SCV_location": l_1},
                    {"SCV_location": l_2},
                )
            )

build_depot_actions = list(map(
    lambda x: Strips(
        f"Build_Depot_{x}",
        {"SCV_location": f"Sector_{x}", f"Sector_{x}": "", "HasMinerals": True},
        {f"Sector_{x}": "Depot", "HasMinerals": False},
    ),
    range(1, len(sectors) + 1),
))

starcraft_domain = STRIPS_domain(
    feature_domain_dict={
        "Minerals_1": mineralField,
        "Minerals_2": mineralField,
        "Sector_1": building,
        "Sector_2": building,
        "HasMinerals": boolean,
        "SCV_location": locations,
    },
    actions={
        *collect_minerals_actions,
        *move_scv_actions,
        *build_depot_actions,
        Strips(
            "Build_Barracks_1",
            {
                "HasMinerals": True,
                "Sector_1": "",
                "SCV_location": "Sector_1",
                "Sector_2": "Depot",
            },
            {"HasMinerals": False, "Sector_1": "Barracks"},
        ),
    },
)

build_barracks = Planning_problem(
    prob_domain=starcraft_domain,
    initial_state={
        "HasMinerals": False,
        "Minerals_1": "Minerals",
        "Minerals_2": "Minerals",
        "Sector_1": "",
        "Sector_2": "",
        "SCV_location": "Sector_1",
    },
    goal={"Sector_1": "Barracks"},
)

s1 = SearcherMPP(Forward_STRIPS(build_barracks))  # A*
s1.search()  # find another plan
