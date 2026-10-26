# F1 podium predictor
# idea: predict the top 3 of a GP using only what we knew before the race
# data from jolpica api (the new ergast), 2018 to 2026
#
# how to run (colab):
#   !pip -q install pandas scikit-learn requests
#   !python austin_podium_model.py          -> austin prediction only
#   !python austin_podium_model.py score    -> compare with real result
#   !python austin_podium_model.py next     -> predict next race (run before the race!)

import os
import sys
import time

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

URL = "https://api.jolpi.ca/ergast/f1"
YEARS = list(range(2018, 2027))

# columns the model learns from
FEATURES = ["grid", "q_pos", "form_5", "podium_rate_10", "team_pts_5",
            "track_avg_finish", "track_podiums", "pts_before", "team_pts_before"]

# colours for the output, colab shows these fine
RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
GREEN, RED, YELLOW = "\033[92m", "\033[91m", "\033[93m"
GOLD, SILVER, BRONZE = "\033[38;5;220m", "\033[38;5;250m", "\033[38;5;208m"


def get_json(url, params, tries=8):
    # the api blocks us if we ask too fast (429) so just wait and try again
    for i in range(tries):
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 429 or r.status_code >= 500:
            wait = float(r.headers.get("Retry-After", 0)) or 5 * (i + 1)
            print(f"  too many requests ({r.status_code}), waiting {wait:.0f}s...")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"gave up after {tries} tries: {url}")


def get_data(year, kind):
    # kind is "results" or "qualifying"
    cache_file = f"cache_{year}_{kind}.csv"

    # old seasons never change so save them, 2026 gets fetched fresh every time
    if year < 2026 and os.path.exists(cache_file):
        return pd.read_csv(cache_file)

    rows = []
    offset = 0
    while True:
        print(f"fetching {year} {kind} offset {offset}")
        data = get_json(f"{URL}/{year}/{kind}.json", {"limit": 100, "offset": offset})["MRData"]
        key = "Results" if kind == "results" else "QualifyingResults"

        for race in data["RaceTable"]["Races"]:
            for x in race[key]:
                rows.append(dict(
                    season=year,
                    rnd=int(race["round"]),
                    circuit=race["Circuit"]["circuitId"],
                    driver=x["Driver"]["driverId"],
                    team=x["Constructor"]["constructorId"],
                    pos=int(x["position"]),
                    grid=int(x.get("grid", 0)) if kind == "results" else np.nan,
                    pts=float(x.get("points", 0)),
                ))

        offset += 100
        if offset >= int(data["total"]):
            break
        time.sleep(1.5)

    df = pd.DataFrame(rows)
    if year < 2026:
        df.to_csv(cache_file, index=False)
    return df


def make_features(results, upcoming=None):
    # every feature uses shift(1) so a race never sees its own result
    df = results.copy()
    df["podium"] = (df["pos"] <= 3).astype(float)
    if upcoming is not None:
        df = pd.concat([df, upcoming], ignore_index=True)
    df = df.sort_values(["season", "rnd"]).reset_index(drop=True)
    df["t"] = df["season"] * 100 + df["rnd"]  # simple time stamp to order races

    # how the driver has been finishing lately
    by_driver = df.groupby("driver")
    df["form_5"] = by_driver["pos"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    df["podium_rate_10"] = by_driver["podium"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())

    # how strong the team car has been
    team = df.groupby(["team", "t"], as_index=False)["pts"].sum().sort_values("t")
    team["team_pts_5"] = team.groupby("team")["pts"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    df = df.merge(team[["team", "t", "team_pts_5"]], on=["team", "t"], how="left")

    # how the driver does at this track
    by_track = df.groupby(["driver", "circuit"])
    df["track_avg_finish"] = by_track["pos"].transform(lambda s: s.shift(1).expanding().mean())
    df["track_podiums"] = by_track["podium"].transform(lambda s: s.shift(1).expanding().sum())

    # points so far this season (driver and team)
    df["pts_before"] = df.groupby(["driver", "season"])["pts"].transform(lambda s: s.shift(1).cumsum())
    team_season = df.groupby(["team", "season", "t"], as_index=False)["pts"].sum().sort_values("t")
    team_season["team_pts_before"] = team_season.groupby(["team", "season"])["pts"].transform(lambda s: s.shift(1).cumsum())
    df = df.merge(team_season[["team", "t", "team_pts_before"]], on=["team", "t"], how="left")

    # fill the blanks (first race of a driver etc)
    for c in ["pts_before", "team_pts_before", "track_podiums"]:
        df[c] = df[c].fillna(0)
    for c in FEATURES:
        df[c] = df[c].fillna(df[c].median() if df[c].notna().any() else 0)

    df.loc[df["grid"] == 0, "grid"] = 20  # pit lane start
    return df


def sample_weights(d):
    # 2026 has new rules so recent seasons count more
    return d["season"].map(lambda s: 4.0 if s == 2026 else 2.0 if s == 2025 else 1.0).values


def train_and_predict(train, test):
    # model 1 says who will be on the podium, model 2 guesses the finishing order
    clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=250,
                                         l2_regularization=1.0, random_state=0)
    clf.fit(train[FEATURES], train["podium"], sample_weight=sample_weights(train))

    reg = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=250, random_state=0)
    reg.fit(train[FEATURES], train["pos"], sample_weight=sample_weights(train))

    out = test[["driver", "team", "grid"]].copy()
    out["p_podium"] = clf.predict_proba(test[FEATURES])[:, 1]
    out["exp_pos"] = reg.predict(test[FEATURES])
    return out


def pick_top3(out):
    best = out.sort_values("p_podium", ascending=False).head(3)  # pick the 3 most likely
    return best.sort_values("exp_pos")                           # then order them


def backtest(df, season=2026):
    # go through every race of the season, train on the races before it, predict it
    hits, base_hits = [], []
    exact, base_exact = 0, 0

    rounds = sorted(df[(df.season == season) & df.pos.notna()].rnd.unique())
    for r in rounds:
        test = df[(df.season == season) & (df.rnd == r)]
        train = df[(df.t < season * 100 + r) & df.pos.notna()]
        if len(train) < 200:
            continue

        pred = pick_top3(train_and_predict(train, test))
        real = test[test.pos <= 3].sort_values("pos")["driver"].tolist()
        grid_top3 = test.sort_values("grid").head(3)["driver"].tolist()

        hits.append(len(set(pred.driver) & set(real)))
        base_hits.append(len(set(grid_top3) & set(real)))
        exact += pred.driver.tolist() == real
        base_exact += grid_top3 == real

    n = max(len(hits), 1)
    print(f"Backtest on {len(hits)} races of {season} (train only on data before each race)")
    print(f"  model   : {np.mean(hits):.2f}/3 podium drivers right, exact order {exact}/{n}")
    print(f"  grid top3 baseline: {np.mean(base_hits):.2f}/3, exact order {base_exact}/{n}")


def predict_next():
    # use this one BEFORE a race, after qualifying is done
    results = pd.concat([get_data(y, "results") for y in YEARS], ignore_index=True)
    quali_2026 = get_data(2026, "qualifying").rename(columns={"pos": "q_pos"})
    quali_old = pd.concat([get_data(y, "qualifying") for y in YEARS[:-1]], ignore_index=True).rename(columns={"pos": "q_pos"})
    quali = pd.concat([quali_old, quali_2026], ignore_index=True)[["season", "rnd", "driver", "q_pos"]]
    results = results.merge(quali, on=["season", "rnd", "driver"], how="left")

    last_done = results[results.season == 2026].rnd.max()
    next_quali = quali_2026[quali_2026.rnd > last_done]
    if next_quali.empty:
        sys.exit("no qualifying found for the next race yet, try again after quali")

    rnd = next_quali.rnd.min()
    upcoming = next_quali[next_quali.rnd == rnd].copy()
    upcoming["grid"] = upcoming["q_pos"]  # penalties not counted, change by hand if needed
    upcoming["pos"], upcoming["podium"] = np.nan, np.nan
    print(f"Predicting 2026 round {rnd} ({upcoming.circuit.iloc[0]})")

    df = make_features(results, upcoming)
    backtest(df)

    train = df[df.pos.notna()]
    test = df[(df.season == 2026) & (df.rnd == rnd)]
    out = train_and_predict(train, test).sort_values("p_podium", ascending=False)

    print("\nPodium probabilities:")
    print(out.head(8).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\nPREDICTED PODIUM:")
    for i, (_, r) in enumerate(pick_top3(out).iterrows(), 1):
        print(f"  P{i}: {r.driver} ({r.team})  p={r.p_podium:.2f}")


def test_on_race(track="americas", season=2026, score=False):
    # pretend we are before the race: train only on older races, predict, then (optionally) check
    results = pd.concat([get_data(y, "results") for y in YEARS], ignore_index=True)
    quali = pd.concat([get_data(y, "qualifying") for y in YEARS], ignore_index=True).rename(columns={"pos": "q_pos"})
    results = results.merge(quali[["season", "rnd", "driver", "q_pos"]], on=["season", "rnd", "driver"], how="left")

    race = results[(results.season == season) & results.circuit.str.contains(track)]
    if race.empty:
        sys.exit(f"{track} {season} not in the data yet, api takes a few hours after a race")
    rnd = race.rnd.iloc[0]

    df = make_features(results)
    train = df[(df.t < season * 100 + rnd) & df.pos.notna()]  # nothing from the race itself
    test = df[(df.season == season) & (df.rnd == rnd)]
    out = train_and_predict(train, test).sort_values("p_podium", ascending=False)

    pred = pick_top3(out)["driver"].tolist()
    real = test[test.pos <= 3].sort_values("pos")["driver"].tolist()
    grid_top3 = test.sort_values("grid").head(3)["driver"].tolist()
    nice = lambda d: d.replace("_", " ").title()

    print(f"{DIM}Trained on {len(train)} driver-races before {track} {season} (round {rnd}){RESET}\n")

    print(f"{BOLD}PODIUM PROBABILITIES{RESET}")
    for _, r in out.head(8).iterrows():
        bar = "█" * int(r.p_podium * 20)
        col = GREEN if r.p_podium >= 0.5 else YELLOW if r.p_podium >= 0.25 else DIM
        print(f"  {nice(r.driver):<16}{col}{bar:<20}{RESET} {r.p_podium:4.0%}  {DIM}grid P{int(r.grid)}{RESET}")

    print(f"\n{BOLD}PREDICTED PODIUM{RESET}")
    for i, (d, c) in enumerate(zip(pred, [GOLD, SILVER, BRONZE]), 1):
        print(f"  {c}{BOLD}P{i}  {nice(d)}{RESET}")

    if not score:
        print(f"\n{DIM}(Prediction only. Run with `score` to compare with the real result.){RESET}")
        return

    print(f"\n{BOLD}ACTUAL PODIUM{RESET}")
    for i, d in enumerate(real, 1):
        ok = d in pred
        print(f"  P{i}  {nice(d)}  {(GREEN + '✔ in our podium') if ok else (RED + '✘ missed')}{RESET}")

    model_hits = len(set(pred) & set(real))
    grid_hits = len(set(grid_top3) & set(real))
    print(f"\n{BOLD}SCORE{RESET}")
    print(f"  Model    : {GREEN if model_hits >= grid_hits else RED}{model_hits}/3{RESET} podium drivers correct")
    print(f"  Baseline : {grid_hits}/3 (grid top 3: {', '.join(nice(d) for d in grid_top3)})")
    print(f"  Exact order: {GREEN + 'yes' if pred == real else YELLOW + 'no'}{RESET}\n")
    backtest(df)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "next":
        predict_next()
    else:
        test_on_race(score=(arg == "score"))
