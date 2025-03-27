from enum import Enum
from searchMPP import SearcherMPP
from stripsForwardPlanner import Forward_STRIPS
from stripsProblem import Planning_problem, STRIPS_domain, Strips


class Building(Enum):
    EMPTY = ""
    BARRACKS = "Barracks"
    DEPOT = "Depot"


# minerals = {"mineral_field_a"}
sectors = set(map(lambda x: f"Sector_{x}", range(1, 3)))
boolean = {True, False}
building = set(e.value for e in Building)
units = {"", "Marine", "Wraith", "Tank"}
mineralField = {"", "Minerals"}
mineral_fields = set(map(lambda x: f"Minerals_{x}", range(1, 4)))


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
        range(1, len(mineral_fields) + 1),
    )
)

move_scv_actions = []

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

build_depot_actions = list(
    map(
        lambda x: Strips(
            f"Build_Depot_{x}",
            {
                "SCV_location": f"Sector_{x}",
                f"Sector_{x}": Building.EMPTY,
                "HasMinerals": True,
            },
            {f"Sector_{x}": Building.DEPOT, "HasMinerals": False},
        ),
        range(1, len(sectors) + 1),
    )
)

build_barracks_actions = []
for i in range(1, len(sectors) + 1):
    for j in range(1, len(sectors) + 1):
        if i != j:
            build_barracks_actions.append(
                Strips(
                    f"Build_Barracks_{i}_With_Depot_At_{j}",
                    {
                        "HasMinerals": True,
                        f"Sector_{i}": Building.EMPTY,
                        "SCV_location": f"Sector_{i}",
                        f"Sector_{j}": Building.DEPOT,
                    },
                    {"HasMinerals": False, f"Sector_{i}": Building.BARRACKS},
                )
            )


train_marine_actions = list(
    map(
        lambda x: Strips(
            f"train_marine_{x}",
            {
                f"Sector_{x}": Building.BARRACKS,
                "HasMinerals": True,
                f"Unit_{x}": "",
            },
            {
                f"Unit_{x}": "Marine",
                "HasMinerals": False,
            },
        ),
        range(1, len(sectors) + 1),
    )
)

starcraft_domain = STRIPS_domain(
    feature_domain_dict={
        "Minerals_1": mineralField,
        "Minerals_2": mineralField,
        "Minerals_3": mineralField,
        "Sector_1": building,
        "Sector_2": building,
        "HasMinerals": boolean,
        "SCV_location": locations,
        "Unit_1": units,
        "Unit_2": units,
        "Unit_3": units,
    },
    actions={
        *collect_minerals_actions,
        *move_scv_actions,
        *build_depot_actions,
        *build_barracks_actions,
        *train_marine_actions,
        Strips(
            "Build_Barracks_1",
            {
                "HasMinerals": True,
                "Sector_1": Building.EMPTY,
                "SCV_location": "Sector_1",
                "Sector_2": Building.DEPOT,
            },
            {"HasMinerals": False, "Sector_1": Building.BARRACKS},
        ),
    },
)

initial_units_state = {
    "Unit_1": "",
    "Unit_2": "",
    "Unit_3": "",
}

initial_sector_state = {
    "Sector_1": Building.EMPTY,
    "Sector_2": Building.EMPTY,
}

initial_minerals_state = {
    "HasMinerals": False,
    "Minerals_1": "Minerals",
    "Minerals_2": "Minerals",
    "Minerals_3": "Minerals",
}

initial_state = {
    **initial_minerals_state,
    **initial_sector_state,
    "SCV_location": "Sector_1",
    **initial_units_state,
}

build_barracks_problem = Planning_problem(
    prob_domain=starcraft_domain,
    initial_state=initial_state,
    goal={"Sector_1": Building.BARRACKS},
)

build_depot_problem = Planning_problem(
    prob_domain=starcraft_domain,
    initial_state=initial_state,
    goal={"Sector_1": Building.DEPOT},
)

train_marine_problem = Planning_problem(
    prob_domain=starcraft_domain,
    initial_state=initial_state,
    goal={"Unit_1": "Marine"},
)

s1 = SearcherMPP(Forward_STRIPS(train_marine_problem))  # A*
s1.search()  # find another plan
