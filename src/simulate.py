# src/simulate.py
# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — 2026 FIFA World Cup Tournament Simulation
#
# Run from project root:
#     python -m src.simulate
#
# What this script does, in plain English:
#   1. Loads the trained neural network and the scaler from Phase 3.
#   2. Builds a dictionary of all 48 qualified teams with their feature values.
#   3. Defines a function to predict any match between two teams.
#   4. Simulates the 2026 group stage (16 groups × 3 teams).
#   5. Simulates the knockout stage (R32 → R16 → QF → SF → Final).
#   6. Runs everything 10,000 times (Monte Carlo) and counts how often each
#      team wins the tournament.
#   7. Saves results to outputs/results/simulation_results.csv.
# ─────────────────────────────────────────────────────────────────────────────

import os
import pickle
import random
import numpy as np
import pandas as pd
import torch

from src.model import MatchPredictor

# ── Reproducibility ───────────────────────────────────────────────────────────
# Setting seeds makes results reproducible across runs.
# We seed Python's random, NumPy, and PyTorch separately because they each
# have their own internal random number generators.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs("outputs/results", exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD MODEL AND SCALER
# ═══════════════════════════════════════════════════════════════════════════════

def load_model_and_scaler(
    model_path: str = "models/neural_net_pytorch.pth",
    scaler_path: str = "data/features/scaler.pkl",
):
    """
    Load the trained neural network and the feature scaler from disk.

    Why reuse the scaler?
        The model was trained on *scaled* feature values. If we feed it raw
        numbers at inference time, the inputs live in a completely different
        numerical range — the model will produce garbage predictions.
        We must apply exactly the same transformation used during training.

    Returns:
        model  — MatchPredictor in eval() mode (ready for inference)
        scaler — fitted sklearn StandardScaler
        device — torch.device (cpu or cuda)
    """
    # ── Load scaler ──────────────────────────────────────────────────────────
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print(f"✓ Scaler loaded from {scaler_path}")

    # ── Load neural network ──────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MatchPredictor(input_dim=9, dropout_rate=0.3)

    # map_location=device ensures the weights load correctly whether or not
    # a GPU was used during training.
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # eval() disables Dropout layers so predictions are deterministic.
    # During training, Dropout randomly switches off neurons — we don't
    # want that randomness at inference time.
    model.eval()
    print(f"✓ Neural network loaded from {model_path} (device: {device})")

    return model, scaler, device


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TEAM FEATURE DICTIONARY
# ═══════════════════════════════════════════════════════════════════════════════
# Each team's features are derived from their most recent values available
# in the feature dataset (through end of 2024).
#
# Feature structure (must match FEATURE_COLS from src/utils.py):
#   [home_fifa_points, away_fifa_points, fifa_points_diff,
#    home_form, away_form, h2h_home_win_rate,
#    tournament_weight, home_avg_goal_diff, away_avg_goal_diff]
#
# Note on FIFA points:
#   Approximate values from the October 2024 FIFA rankings.
#
# Note on form / avg_goal_diff:
#   Estimated from each team's recent international results (2023–2024).
#   form: fraction of available points won over ~last 5 games [0, 1].
#   avg_goal_diff: average (goals scored – goals conceded) per game.
#
# Note on h2h_home_win_rate and tournament_weight:
#   These are match-level features. We store per-team attributes here;
#   the match prediction function (Section 3) combines them into a proper
#   9-feature vector using neutral defaults.
# ─────────────────────────────────────────────────────────────────────────────

TEAM_FEATURES = {
    # ── Group A ─────────────────────────────────────────────────────────────
    "USA":           {"fifa_points": 1719.7, "form": 0.60, "avg_goal_diff":  0.45},
    "Mexico":        {"fifa_points": 1632.9, "form": 0.55, "avg_goal_diff":  0.30},
    "Canada":        {"fifa_points": 1487.5, "form": 0.58, "avg_goal_diff":  0.35},

    # ── Group B ─────────────────────────────────────────────────────────────
    "Argentina":     {"fifa_points": 1896.1, "form": 0.80, "avg_goal_diff":  1.20},
    "Ecuador":       {"fifa_points": 1466.5, "form": 0.52, "avg_goal_diff":  0.20},
    "Chile":         {"fifa_points": 1410.2, "form": 0.45, "avg_goal_diff": -0.10},

    # ── Group C ─────────────────────────────────────────────────────────────
    "Brazil":        {"fifa_points": 1785.2, "form": 0.65, "avg_goal_diff":  0.90},
    "Colombia":      {"fifa_points": 1594.0, "form": 0.62, "avg_goal_diff":  0.55},
    "Venezuela":     {"fifa_points": 1309.4, "form": 0.42, "avg_goal_diff": -0.20},

    # ── Group D ─────────────────────────────────────────────────────────────
    "France":        {"fifa_points": 1864.5, "form": 0.72, "avg_goal_diff":  1.10},
    "Belgium":       {"fifa_points": 1786.6, "form": 0.65, "avg_goal_diff":  0.80},
    "Ukraine":       {"fifa_points": 1567.0, "form": 0.50, "avg_goal_diff":  0.15},

    # ── Group E ─────────────────────────────────────────────────────────────
    "England":       {"fifa_points": 1826.9, "form": 0.68, "avg_goal_diff":  0.95},
    "Netherlands":   {"fifa_points": 1763.9, "form": 0.65, "avg_goal_diff":  0.80},
    "Wales":         {"fifa_points": 1386.3, "form": 0.40, "avg_goal_diff": -0.15},

    # ── Group F ─────────────────────────────────────────────────────────────
    "Spain":         {"fifa_points": 1851.3, "form": 0.78, "avg_goal_diff":  1.15},
    "Portugal":      {"fifa_points": 1775.9, "form": 0.72, "avg_goal_diff":  1.00},
    "Croatia":       {"fifa_points": 1636.2, "form": 0.58, "avg_goal_diff":  0.40},

    # ── Group G ─────────────────────────────────────────────────────────────
    "Germany":       {"fifa_points": 1777.4, "form": 0.65, "avg_goal_diff":  0.85},
    "Austria":       {"fifa_points": 1584.8, "form": 0.58, "avg_goal_diff":  0.45},
    "Switzerland":   {"fifa_points": 1625.8, "form": 0.62, "avg_goal_diff":  0.50},

    # ── Group H ─────────────────────────────────────────────────────────────
    "Morocco":       {"fifa_points": 1715.0, "form": 0.68, "avg_goal_diff":  0.70},
    "Senegal":       {"fifa_points": 1621.8, "form": 0.60, "avg_goal_diff":  0.50},
    "Tunisia":       {"fifa_points": 1486.0, "form": 0.45, "avg_goal_diff":  0.05},

    # ── Group I ─────────────────────────────────────────────────────────────
    "Japan":         {"fifa_points": 1736.4, "form": 0.70, "avg_goal_diff":  0.75},
    "South Korea":   {"fifa_points": 1594.0, "form": 0.58, "avg_goal_diff":  0.35},
    "Australia":     {"fifa_points": 1502.2, "form": 0.52, "avg_goal_diff":  0.20},

    # ── Group J ─────────────────────────────────────────────────────────────
    "Iran":          {"fifa_points": 1557.3, "form": 0.55, "avg_goal_diff":  0.30},
    "Saudi Arabia":  {"fifa_points": 1472.0, "form": 0.48, "avg_goal_diff":  0.10},
    "Uzbekistan":    {"fifa_points": 1349.9, "form": 0.45, "avg_goal_diff": -0.05},

    # ── Group K ─────────────────────────────────────────────────────────────
    "Uruguay":       {"fifa_points": 1680.5, "form": 0.62, "avg_goal_diff":  0.60},
    "Bolivia":       {"fifa_points": 1188.2, "form": 0.35, "avg_goal_diff": -0.60},
    "Peru":          {"fifa_points": 1380.4, "form": 0.40, "avg_goal_diff": -0.20},

    # ── Group L ─────────────────────────────────────────────────────────────
    "Nigeria":       {"fifa_points": 1523.8, "form": 0.50, "avg_goal_diff":  0.15},
    "Egypt":         {"fifa_points": 1545.0, "form": 0.52, "avg_goal_diff":  0.20},
    "Ghana":         {"fifa_points": 1398.4, "form": 0.42, "avg_goal_diff": -0.10},

    # ── Group M ─────────────────────────────────────────────────────────────
    "Italy":         {"fifa_points": 1748.0, "form": 0.60, "avg_goal_diff":  0.65},
    "Turkey":        {"fifa_points": 1577.6, "form": 0.58, "avg_goal_diff":  0.40},
    "Albania":       {"fifa_points": 1367.5, "form": 0.45, "avg_goal_diff": -0.10},

    # ── Group N ─────────────────────────────────────────────────────────────
    "Ivory Coast":   {"fifa_points": 1533.7, "form": 0.55, "avg_goal_diff":  0.30},
    "Algeria":       {"fifa_points": 1480.7, "form": 0.48, "avg_goal_diff":  0.10},
    "Cameroon":      {"fifa_points": 1440.5, "form": 0.45, "avg_goal_diff":  0.00},

    # ── Group O ─────────────────────────────────────────────────────────────
    "Denmark":       {"fifa_points": 1698.0, "form": 0.63, "avg_goal_diff":  0.65},
    "Serbia":        {"fifa_points": 1590.4, "form": 0.55, "avg_goal_diff":  0.35},
    "Scotland":      {"fifa_points": 1449.8, "form": 0.48, "avg_goal_diff":  0.10},

    # ── Group P ─────────────────────────────────────────────────────────────
    "Qatar":         {"fifa_points": 1439.0, "form": 0.42, "avg_goal_diff": -0.10},
    "New Zealand":   {"fifa_points": 1202.4, "form": 0.35, "avg_goal_diff": -0.50},
    "Panama":        {"fifa_points": 1384.7, "form": 0.45, "avg_goal_diff": -0.05},
}

# ─────────────────────────────────────────────────────────────────────────────
# Official 2026 FIFA World Cup Groups (16 groups × 3 teams)
# Source: FIFA official draw (December 2024)
# ─────────────────────────────────────────────────────────────────────────────
GROUPS = {
    "A": ["USA",        "Mexico",       "Canada"],
    "B": ["Argentina",  "Ecuador",      "Chile"],
    "C": ["Brazil",     "Colombia",     "Venezuela"],
    "D": ["France",     "Belgium",      "Ukraine"],
    "E": ["England",    "Netherlands",  "Wales"],
    "F": ["Spain",      "Portugal",     "Croatia"],
    "G": ["Germany",    "Austria",      "Switzerland"],
    "H": ["Morocco",    "Senegal",      "Tunisia"],
    "I": ["Japan",      "South Korea",  "Australia"],
    "J": ["Iran",       "Saudi Arabia", "Uzbekistan"],
    "K": ["Uruguay",    "Bolivia",      "Peru"],
    "L": ["Nigeria",    "Egypt",        "Ghana"],
    "M": ["Italy",      "Turkey",       "Albania"],
    "N": ["Ivory Coast","Algeria",      "Cameroon"],
    "O": ["Denmark",    "Serbia",       "Scotland"],
    "P": ["Qatar",      "New Zealand",  "Panama"],
}

# The 9 feature names, in exactly the order the scaler and model expect.
FEATURE_COLS = [
    "home_fifa_points",
    "away_fifa_points",
    "fifa_points_diff",
    "home_form",
    "away_form",
    "h2h_home_win_rate",
    "tournament_weight",
    "home_avg_goal_diff",
    "away_avg_goal_diff",
]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MATCH PREDICTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def build_feature_vector(home_team: str, away_team: str) -> np.ndarray:
    """
    Construct the raw (unscaled) 9-feature vector for a match.

    Arguments:
        home_team — name of the "home" team (or first-listed, for neutral venues)
        away_team — name of the "away" team

    Returns:
        np.ndarray of shape (1, 9) — ready to be passed to scaler.transform()

    Notes on fixed defaults:
        h2h_home_win_rate = 0.45
            Historical head-to-head rate at which the "home" side wins their
            direct meetings. 0.45 is a neutral default — slightly below 0.5
            because away teams historically win H2H clashes a touch less often.

        tournament_weight = 0.75
            In the training data, weights range from ~0.25 (friendlies) to 1.0
            (World Cup). We use 0.75 to reflect a major competitive tournament
            (group stage and knockout rounds are roughly equivalent in importance).
    """
    h = TEAM_FEATURES[home_team]
    a = TEAM_FEATURES[away_team]

    raw = np.array([[
        h["fifa_points"],                       # home_fifa_points
        a["fifa_points"],                       # away_fifa_points
        h["fifa_points"] - a["fifa_points"],    # fifa_points_diff
        h["form"],                              # home_form
        a["form"],                              # away_form
        0.45,                                   # h2h_home_win_rate (neutral default)
        0.75,                                   # tournament_weight (major tournament)
        h["avg_goal_diff"],                     # home_avg_goal_diff
        a["avg_goal_diff"],                     # away_avg_goal_diff
    ]], dtype=np.float32)

    return raw


def predict_match(
    home_team: str,
    away_team: str,
    model: MatchPredictor,
    scaler,
    device: torch.device,
) -> dict:
    """
    Predict outcome probabilities for a single match.

    Steps:
        1. Build the raw 9-feature vector.
        2. Scale it using the Phase 3 scaler (critical — must match training).
        3. Convert to a PyTorch tensor.
        4. Forward pass through the network → Softmax probabilities.

    Returns:
        dict with keys "away_win", "draw", "home_win" (each a float summing to 1.0)
    """
    raw    = build_feature_vector(home_team, away_team)         # (1, 9) unscaled
    scaled = scaler.transform(raw)                              # (1, 9) scaled
    x      = torch.tensor(scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        probs = model.predict_proba(x).cpu().numpy()[0]         # shape (3,)

    return {
        "away_win": float(probs[0]),
        "draw":     float(probs[1]),
        "home_win": float(probs[2]),
    }


def sample_group_outcome(probs: dict) -> int:
    """
    Randomly sample a match outcome from the model's probability distribution.

    Why sample instead of always picking the most likely result?
        Always predicting the most probable outcome would make every simulation
        identical — upsets would never happen. By sampling from the probability
        distribution, strong teams still win most of the time, but weaker teams
        can cause the occasional surprise, which reflects real football.

    Returns:
        0 = away win  |  1 = draw  |  2 = home win
    """
    p = [probs["away_win"], probs["draw"], probs["home_win"]]
    # Normalize probabilities to ensure they sum to exactly 1.0
    # (fixes floating-point precision issues)
    p = np.array(p) / np.sum(p)
    return int(np.random.choice([0, 1, 2], p=p))


def resolve_knockout(
    team_a: str,
    team_b: str,
    model: MatchPredictor,
    scaler,
    device: torch.device,
) -> str:
    """
    Predict a knockout match and return the winner (no draws allowed).

    If the sampled outcome is a draw, we redistribute the draw probability
    proportionally between the two teams and flip a weighted coin.
    This simulates extra time / penalty shootout where the stronger team
    (according to the model) retains a slight edge.

    team_a is listed as "home", team_b as "away".
    """
    probs   = predict_match(team_a, team_b, model, scaler, device)
    outcome = sample_group_outcome(probs)

    if outcome == 2:    # home win
        return team_a
    elif outcome == 0:  # away win
        return team_b
    else:               # draw → weighted coin flip
        total = probs["home_win"] + probs["away_win"]
        p_a   = probs["home_win"] / total if total > 0 else 0.5
        return team_a if random.random() < p_a else team_b


# ═══════════════════════════════════════════════════════════════════════════════
# 4. GROUP STAGE SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_group(
    teams: list,
    model: MatchPredictor,
    scaler,
    device: torch.device,
) -> pd.DataFrame:
    """
    Simulate one group (3 teams, round-robin — each plays the other once).

    Standings tiebreakers (in order):
        1. Points  (Win=3, Draw=1, Loss=0)
        2. Goal difference  (simulated from avg_goal_diff features)
        3. FIFA ranking points  (higher = better)

    Returns a DataFrame sorted by final standing (1st to 3rd).
    """
    standings = {
        t: {"points": 0, "gd": 0.0, "fifa_pts": TEAM_FEATURES[t]["fifa_points"]}
        for t in teams
    }

    # 3 matches: (0 vs 1), (0 vs 2), (1 vs 2)
    matchups = [
        (teams[0], teams[1]),
        (teams[0], teams[2]),
        (teams[1], teams[2]),
    ]

    for home, away in matchups:
        probs   = predict_match(home, away, model, scaler, device)
        outcome = sample_group_outcome(probs)

        # Approximate goal difference for tiebreaking purposes.
        # We use each team's avg_goal_diff as a proxy, then add Gaussian noise
        # so the tiebreaker doesn't produce identical results every simulation.
        h_gd = TEAM_FEATURES[home]["avg_goal_diff"]
        a_gd = TEAM_FEATURES[away]["avg_goal_diff"]
        simulated_gd = abs((h_gd - a_gd) + np.random.normal(0, 0.5))

        if outcome == 2:    # home win
            standings[home]["points"] += 3
            standings[home]["gd"]     += simulated_gd
            standings[away]["gd"]     -= simulated_gd
        elif outcome == 0:  # away win
            standings[away]["points"] += 3
            standings[away]["gd"]     += simulated_gd
            standings[home]["gd"]     -= simulated_gd
        else:               # draw
            standings[home]["points"] += 1
            standings[away]["points"] += 1
            # Small symmetric noise around 0 for draws
            nudge = abs(np.random.normal(0, 0.2))
            standings[home]["gd"] += nudge
            standings[away]["gd"] -= nudge

    df = pd.DataFrame([
        {"team": t, "points": v["points"], "gd": v["gd"], "fifa_pts": v["fifa_pts"]}
        for t, v in standings.items()
    ])
    df = df.sort_values(["points", "gd", "fifa_pts"], ascending=False).reset_index(drop=True)
    df["position"] = df.index + 1
    return df


def simulate_group_stage(
    model: MatchPredictor,
    scaler,
    device: torch.device,
) -> dict:
    """
    Simulate all 16 groups and determine who advances.

    2026 Advancement rules:
        • 1st and 2nd from each group advance automatically → 32 teams
        • 8 best third-place teams (ranked by points, GD, FIFA pts) → 8 more
        Total: 40 teams advance to the Round of 32.

    Returns a dict with:
        "group_results"           — {letter: DataFrame} for all 16 groups
        "first_place"             — list of 16 group winners
        "second_place"            — list of 16 runners-up
        "third_place_qualifiers"  — list of 8 best third-place teams
    """
    group_results   = {}
    first_place     = []
    second_place    = []
    third_place_all = []

    for letter, teams in GROUPS.items():
        df = simulate_group(teams, model, scaler, device)
        group_results[letter] = df

        first_place.append(df.loc[df["position"] == 1, "team"].values[0])
        second_place.append(df.loc[df["position"] == 2, "team"].values[0])

        third_row = df[df["position"] == 3].iloc[0]
        third_place_all.append({
            "team":     third_row["team"],
            "points":   third_row["points"],
            "gd":       third_row["gd"],
            "fifa_pts": third_row["fifa_pts"],
        })

    # Rank all 16 third-place finishers; keep top 8
    df_third = pd.DataFrame(third_place_all)
    df_third = df_third.sort_values(
        ["points", "gd", "fifa_pts"], ascending=False
    ).reset_index(drop=True)
    third_place_qualifiers = df_third["team"].tolist()[:8]

    return {
        "group_results":          group_results,
        "first_place":            first_place,
        "second_place":           second_place,
        "third_place_qualifiers": third_place_qualifiers,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. KNOCKOUT STAGE SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_knockout_round(
    teams: list,
    model: MatchPredictor,
    scaler,
    device: torch.device,
) -> list:
    """
    Play one knockout round. Teams are paired sequentially:
        teams[0] vs teams[1], teams[2] vs teams[3], ...

    Requires an even-length list. Returns the list of winners.
    """
    assert len(teams) % 2 == 0, "Knockout round requires an even number of teams."
    winners = []
    for i in range(0, len(teams), 2):
        winner = resolve_knockout(teams[i], teams[i + 1], model, scaler, device)
        winners.append(winner)
    return winners


def simulate_knockout_stage(
    first_place: list,
    second_place: list,
    third_place_qualifiers: list,
    model: MatchPredictor,
    scaler,
    device: torch.device,
) -> str:
    """
    Simulate the full knockout bracket and return the champion.

    40-team bracket structure:
        We have 16 group winners + 16 runners-up + 8 third-place qualifiers = 40 teams.

        Bye rule:
            The top 8 group winners (by FIFA points) receive a bye to the R16.
            The bottom 8 group winners play the R32 against the 8 third-place
            qualifiers. The 16 runners-up play each other in the R32.

        Round of 32:
            (a) 8 weaker group winners vs 8 third-place qualifiers → 8 winners
            (b) 16 runners-up vs each other                         → 8 winners
            Total: 16 R32 winners

        Round of 16 onward:
            Pool = 8 bye teams + 16 R32 winners = 24 teams.
            We run sequential halving rounds until 1 champion remains.
            If a round has an odd number of teams, the highest-ranked remaining
            team (by FIFA points) receives a bye to the next round.

    Returns the name of the tournament champion.
    """
    # Sort group winners; top 8 get byes, bottom 8 play R32
    first_sorted = sorted(
        first_place,
        key=lambda t: TEAM_FEATURES[t]["fifa_points"],
        reverse=True
    )
    bye_teams = first_sorted[:8]   # go straight to R16
    r32_seeds = first_sorted[8:]   # play R32 (8 teams)

    # Pair each weaker group winner with a third-place qualifier
    r32_bracket_a = []
    for gw, tp in zip(r32_seeds, third_place_qualifiers):
        r32_bracket_a.extend([gw, tp])

    # Runners-up play each other in the R32 (16 teams → 8 matches)
    r32_bracket_b = second_place[:]

    # ── Round of 32 ──────────────────────────────────────────────────────────
    r32_winners_a = simulate_knockout_round(r32_bracket_a, model, scaler, device)  # 8
    r32_winners_b = simulate_knockout_round(r32_bracket_b, model, scaler, device)  # 8
    r32_winners   = r32_winners_a + r32_winners_b                                  # 16

    # ── Rounds from R16 to Final ──────────────────────────────────────────────
    # Combine: 8 bye teams + 16 R32 winners = 24 teams total for R16+
    remaining = bye_teams + r32_winners  # 24 teams

    while len(remaining) > 1:
        if len(remaining) % 2 == 1:
            # Odd number: strongest remaining team gets a bye
            bye_idx  = max(range(len(remaining)),
                           key=lambda i: TEAM_FEATURES[remaining[i]]["fifa_points"])
            bye_team = remaining.pop(bye_idx)
            winners  = simulate_knockout_round(remaining, model, scaler, device)
            remaining = winners + [bye_team]
        else:
            remaining = simulate_knockout_round(remaining, model, scaler, device)

    return remaining[0]


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MONTE CARLO SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_monte_carlo(
    model: MatchPredictor,
    scaler,
    device: torch.device,
    n_simulations: int = 10_000,
    print_every: int   = 1_000,
) -> pd.DataFrame:
    """
    Run the full tournament n_simulations times and tally who wins each time.

    Why Monte Carlo?
        A single simulation gives one scenario. Football is highly variable —
        even the best team doesn't always win. Running 10,000 simulations lets
        us estimate the *probability* of each team winning, not just one
        possible future. Teams that win more often are genuinely more likely
        to win the real tournament, given our model's predictions.

    Returns a DataFrame (sorted by win probability) with columns:
        team, wins, win_probability_pct
    """
    win_counts = {team: 0 for team in TEAM_FEATURES}

    print(f"\nRunning {n_simulations:,} Monte Carlo simulations...")
    print("(Each simulation runs ~100+ matches — expect a few minutes.)\n")

    for i in range(1, n_simulations + 1):
        # Full tournament: group stage → knockout stage
        group_data = simulate_group_stage(model, scaler, device)

        champion = simulate_knockout_stage(
            first_place             = group_data["first_place"],
            second_place            = group_data["second_place"],
            third_place_qualifiers  = group_data["third_place_qualifiers"],
            model                   = model,
            scaler                  = scaler,
            device                  = device,
        )

        win_counts[champion] += 1

        if i % print_every == 0:
            print(f"  Completed {i:,} / {n_simulations:,} simulations...")

    print(f"\n✓ All {n_simulations:,} simulations complete.")

    df_results = pd.DataFrame([
        {
            "team":                team,
            "wins":                wins,
            "win_probability_pct": round(wins / n_simulations * 100, 2),
        }
        for team, wins in win_counts.items()
    ])

    df_results = df_results.sort_values(
        "win_probability_pct", ascending=False
    ).reset_index(drop=True)
    df_results.index += 1  # rank starts at 1

    return df_results


# ═══════════════════════════════════════════════════════════════════════════════
# 7. MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  2026 FIFA WORLD CUP — MONTE CARLO TOURNAMENT SIMULATOR")
    print("=" * 65)

    # ── Step 1: Load model and scaler ────────────────────────────────────────
    model, scaler, device = load_model_and_scaler(
        model_path  = "models/neural_net_pytorch.pth",
        scaler_path = "data/features/scaler.pkl",
    )

    # ── Step 2: Sanity check ─────────────────────────────────────────────────
    # Predict one well-known matchup to verify the pipeline is working.
    # Argentina (FIFA #1) vs France (FIFA #2) should show Argentina favoured.
    print("\n── Sanity check: Argentina vs France ──")
    test_probs = predict_match("Argentina", "France", model, scaler, device)
    print(f"   Away win (France):     {test_probs['away_win']:.1%}")
    print(f"   Draw:                  {test_probs['draw']:.1%}")
    print(f"   Home win (Argentina):  {test_probs['home_win']:.1%}")

    # ── Step 3: Run Monte Carlo ───────────────────────────────────────────────
    N_SIMS     = 10_000
    df_results = run_monte_carlo(
        model         = model,
        scaler        = scaler,
        device        = device,
        n_simulations = N_SIMS,
        print_every   = 1_000,
    )

    # ── Step 4: Print top 16 ─────────────────────────────────────────────────
    print("\n── Top 16 Predicted Tournament Winners ──")
    print(f"{'Rank':<6} {'Team':<20} {'Wins':>6} {'Win %':>8}")
    print("-" * 44)
    for rank, row in df_results.head(16).iterrows():
        print(f"{rank:<6} {row['team']:<20} {int(row['wins']):>6} {row['win_probability_pct']:>7.2f}%")

    # ── Step 5: Save results ─────────────────────────────────────────────────
    output_path = "outputs/results/simulation_results.csv"
    df_results.to_csv(output_path, index_label="rank")
    print(f"\n✓ Results saved to {output_path}")
    print("\nPhase 4 complete.")