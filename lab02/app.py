from aipython.stripsProblem import Planning_problem, STRIPS_domain, Strips

# Definicja domeny STRIPS dla Starcrafta
starcraft_domain = STRIPS_domain(
    feature_domain_dict={
        "Area": {
            "mineral_field_a", "mineral_field_b", "sector_a", "sector_b"
        },
        "Unit": {
            "scv", "marine", "tank"
        },
        "Building": {
            "barracks", "factory"
        },
        "Resource": {
            "minerals"
        }
    },
    actions={
        # Akcja ruchu SCV do pola minerałów
        Strips(
            "move_scv_to_minerals", 
            preconds={"at": "scv", "location": "sector_a", "minerals": "mineral_field_a"},
            effects={"at": "scv", "location": "mineral_field_a", "minerals": "mineral_field_a"}
        ),
        # Akcja zbierania minerałów przez SCV
        Strips(
            "collect_minerals", 
            preconds={"at": "scv", "location": "mineral_field_a", "minerals": "mineral_field_a"},
            effects={"minerals": "empty", "scv_collected_minerals": True}
        ),
        # Budowanie koszar
        Strips(
            "build_barracks", 
            preconds={"scv_collected_minerals": True, "at": "scv", "location": "sector_a", "minerals": "empty"},
            effects={"barracks": "sector_a", "scv_collected_minerals": False}
        ),
        # Trenowanie Marynarza
        Strips(
            "train_marine", 
            preconds={"scv_collected_minerals": True, "at": "scv", "location": "sector_a", "barracks": "sector_a"},
            effects={"marine": "sector_a", "scv_collected_minerals": False}
        ),
        # Trenowanie Czołgu
        Strips(
            "train_tank", 
            preconds={"scv_collected_minerals": True, "at": "scv", "location": "sector_a", "factory": "sector_b"},
            effects={"tank": "sector_a", "scv_collected_minerals": False}
        )
    }
)

# Definicja problemów

# Problem: Budowanie koszar
barracks_problem = Planning_problem(
    prob_domain=starcraft_domain,
    initial_state={
        "at": "scv", "location": "sector_a", "minerals": "mineral_field_a", "scv_collected_minerals": False
    },
    goal_state={
        "barracks": "sector_a"
    }
)

# Problem: Trenowanie Marynarza
marine_problem = Planning_problem(
    prob_domain=starcraft_domain,
    initial_state={
        "at": "scv", "location": "sector_a", "minerals": "mineral_field_a", "scv_collected_minerals": False,
        "barracks": "sector_a"
    },
    goal_state={
        "marine": "sector_a"
    }
)

# Problem: Trenowanie Czołgu
tank_problem = Planning_problem(
    prob_domain=starcraft_domain,
    initial_state={
        "at": "scv", "location": "sector_a", "minerals": "mineral_field_a", "scv_collected_minerals": False,
        "factory": "sector_b"
    },
    goal_state={
        "tank": "sector_a"
    }
)

# Rozwiązywanie problemów
result_barracks = barracks_problem.solve()
result_marine = marine_problem.solve()
result_tank = tank_problem.solve()

print("Wynik budowy koszar:", result_barracks)
print("Wynik trenowania Marynarza:", result_marine)
print("Wynik trenowania Czołgu:", result_tank)

# Przykład innego problemu, nieistniejącego w grze Starcraft
problem1 = Planning_problem(
    delivery_domain,  # Zakładając, że domain 'delivery_domain' jest wcześniej zdefiniowana
    {'RLoc': 'lab', 'MW': True, 'SamWantsCoffee': True, 'RHC': False, 'RHM': False}, 
    {'SamWantsCoffee': False}
)
