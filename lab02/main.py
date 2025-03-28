from enum import Enum
import time
from searchGeneric import Searcher
from searchMPP import SearcherMPP
from stripsForwardPlanner import Forward_STRIPS
from stripsProblem import Planning_problem, STRIPS_domain, Strips


def goal_count_heuristic(state, goal):
    scv_location = state.get("SCV_location")
    location_content = state.get(scv_location)
    penalty_for_going_into_empty_mineral_field = (
        2 if scv_location.startswith("Minerals") and location_content == "" else 0
    )
    penalty_for_going_into_full_sector = (
        2
        if scv_location.startswith("Sector") and location_content != Building.EMPTY
        else 0
    )
    return (
        sum(4 for key, value in goal.items() if state.get(key) != value)
        + penalty_for_going_into_empty_mineral_field
        + penalty_for_going_into_full_sector
    )


class Building(Enum):
    EMPTY = ""
    BARRACKS = "Barracks"
    DEPOT = "Depot"
    FACTORY = "Factory"


class Unit(Enum):
    NONE = ""
    MARINE = "Marine"
    VULTURE = "Vulture"
    SIEGE_TANK = "Siege Tank"


sectors = set(map(lambda x: f"Sector_{x}", range(1, 5)))
mineral_fields = set(map(lambda x: f"Minerals_{x}", range(1, 8)))
unit_slots = set(map(lambda x: f"Unit_{x}", range(1, 3)))

boolean = {True, False}
building = set(e.value for e in Building)
units = set(e.value for e in Unit)
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
            {
                f"Sector_{x}": Building.DEPOT,
                "HasMinerals": False,
            },
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
                    {
                        "HasMinerals": False,
                        f"Sector_{i}": Building.BARRACKS,
                    },
                )
            )

build_factory_actions = []
for i in range(1, len(sectors) + 1):
    for j in range(1, len(sectors) + 1):
        if i != j:
            build_factory_actions.append(
                Strips(
                    f"Build_Factory_{i}_With_Barracks_At_{j}",
                    {
                        "HasMinerals": True,
                        f"Sector_{i}": Building.EMPTY,
                        "SCV_location": f"Sector_{i}",
                        f"Sector_{j}": Building.BARRACKS,
                    },
                    {
                        "HasMinerals": False,
                        f"Sector_{i}": Building.FACTORY,
                    },
                )
            )
train_marine_actions = []
for i in range(1, len(sectors) + 1):
    for j in range(1, len(unit_slots) + 1):
        train_marine_actions.append(
            Strips(
                f"Build_Marine_{j}_from_Barracks_{i}",
                {
                    f"Sector_{i}": Building.BARRACKS,
                    "HasMinerals": True,
                    f"Unit_{j}": Unit.NONE,
                },
                {
                    f"Unit_{j}": Unit.MARINE,
                    "HasMinerals": False,
                },
            )
        )

train_siege_tank_actions = []
for i in range(1, len(sectors) + 1):
    for j in range(1, len(unit_slots) + 1):
        train_siege_tank_actions.append(
            Strips(
                f"Train_Siege_Tank_{j}_from_Factory_{i}",
                {
                    f"Sector_{i}": Building.FACTORY,
                    "HasMinerals": True,
                    f"Unit_{j}": Unit.NONE,
                },
                {
                    f"Unit_{j}": Unit.SIEGE_TANK,
                    "HasMinerals": False,
                },
            )
        )

train_wraith_actions = []
for i in range(1, len(sectors) + 1):
    for j in range(1, len(unit_slots) + 1):
        train_wraith_actions.append(
            Strips(
                f"Train_Vulture_{j}_from_Factory_{i}",
                {
                    f"Sector_{i}": Building.FACTORY,
                    "HasMinerals": True,
                    f"Unit_{j}": Unit.NONE,
                },
                {
                    f"Unit_{j}": Unit.VULTURE,
                    "HasMinerals": False,
                },
            )
        )

starcraft_domain = STRIPS_domain(
    feature_domain_dict={
        **{f"Minerals_{x}": mineralField for x in range(1, len(mineral_fields) + 1)},
        **{f"Sector_{x}": building for x in range(1, len(sectors) + 1)},
        "HasMinerals": boolean,
        "SCV_location": locations,
        **{f"Unit_{x}": units for x in range(1, len(unit_slots) + 1)},
    },
    actions={
        *collect_minerals_actions,
        *move_scv_actions,
        *build_depot_actions,
        *build_barracks_actions,
        *build_factory_actions,
        *train_marine_actions,
        *train_siege_tank_actions,
        *train_wraith_actions,
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

initial_units_state = {f"Unit_{x}": Unit.NONE for x in range(1, len(unit_slots) + 1)}

initial_sector_state = {
    f"Sector_{x}": Building.EMPTY for x in range(1, len(sectors) + 1)
}

initial_minerals_state = {
    "HasMinerals": False,
}
for field in mineral_fields:
    initial_minerals_state[field] = "Minerals"


initial_state = {
    **initial_minerals_state,
    **initial_sector_state,
    "SCV_location": "Sector_1",
    **initial_units_state,
}

build_barracks_problem = Planning_problem(
    prob_domain=starcraft_domain,
    initial_state=initial_state,
    goal={
        "Sector_1": Building.BARRACKS,
        "Sector_4": Building.DEPOT,
        "SCV_location": "Sector_2",
    },
)

build_depot_problem = Planning_problem(
    prob_domain=starcraft_domain,
    initial_state=initial_state,
    goal={
        "Sector_1": Building.DEPOT,
        "SCV_location": "Sector_3",
        "HasMinerals": True,
    },
)

train_marine_problem = Planning_problem(
    prob_domain=starcraft_domain,
    initial_state=initial_state,
    goal={
        "Sector_1": Building.DEPOT,
        "Unit_1": Unit.MARINE,
        "Unit_2": Unit.MARINE,
        "Sector_3": Building.BARRACKS,
    },
)

train_siege_tank_problem = Planning_problem(
    prob_domain=starcraft_domain,
    initial_state=initial_state,
    goal={
        "Sector_1": Building.DEPOT,
        "Sector_2": Building.BARRACKS,
        "Sector_3": Building.FACTORY,
        "Sector_4": Building.FACTORY,
        "Unit_1": Unit.SIEGE_TANK,
        "HasMinerals": True,
    },
)

train_vulture_problem = Planning_problem(
    prob_domain=starcraft_domain,
    initial_state=initial_state,
    goal={
        "Sector_1": Building.DEPOT,
        "Sector_2": Building.BARRACKS,
        "Sector_3": Building.FACTORY,
        "Sector_4": Building.DEPOT,
        "Unit_1": Unit.VULTURE,
        "HasMinerals": True,
    },
)

train_vulture_and_tank_problem = Planning_problem(
    prob_domain=starcraft_domain,
    initial_state=initial_state,
    goal={
        "Sector_1": Building.DEPOT,
        "Sector_2": Building.BARRACKS,
        "Sector_3": Building.FACTORY,
        "Unit_1": Unit.VULTURE,
        "Unit_2": Unit.SIEGE_TANK,
        "HasMinerals": True,
    },
)

problem = train_vulture_and_tank_problem

# start = time.time()
# A*
# s1 = SearcherMPP(Forward_STRIPS(problem))
# s1.search()

# end = time.time()
# print(f"Elapsed time: {end - start}")


start = time.time()
# A* z heurystyką
s1 = SearcherMPP(Forward_STRIPS(problem, goal_count_heuristic))
s1.search()

end = time.time()
print(f"Elapsed time with heuristic: {end - start:.4f} seconds")
