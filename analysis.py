"""NYPD 2022 Stop, Question & Frisk - aggregate analysis.
Usage: python3 analysis.py <input.xlsx> <output.json>
"""
import sys, json
import numpy as np
import pandas as pd

src, out = sys.argv[1], sys.argv[2]
df = pd.read_excel(src)
df = df.replace({"(null)": np.nan, "": np.nan})
N = len(df)

def yes(c):
    return (df[c] == "Y").astype(int)

df["arrest"] = yes("SUSPECT_ARRESTED_FLAG")
df["summons"] = yes("SUMMONS_ISSUED_FLAG")
df["noaction"] = ((df.arrest == 0) & (df.summons == 0)).astype(int)
df["frisk"] = yes("FRISKED_FLAG")
df["search"] = yes("SEARCHED_FLAG")
df["firearm"] = yes("FIREARM_FLAG")
df["knife"] = yes("KNIFE_CUTTER_FLAG")
df["otherweapon"] = yes("OTHER_WEAPON_FLAG")
df["weapon"] = yes("WEAPON_FOUND_FLAG")
df["contraband"] = yes("OTHER_CONTRABAND_FLAG")
df["anyfind"] = ((df.weapon == 1) | (df.contraband == 1)).astype(int)

race_map = {
    "BLACK": "Black",
    "WHITE HISPANIC": "Hispanic",
    "BLACK HISPANIC": "Hispanic",
    "WHITE": "White",
    "ASIAN / PACIFIC ISLANDER": "Asian / Pacific Islander",
    "MIDDLE EASTERN/SOUTHWEST ASIAN": "Other",
    "AMERICAN INDIAN/ALASKAN NATIVE": "Other",
}
df["race"] = df.SUSPECT_RACE_DESCRIPTION.map(race_map).fillna("Unknown")
df["sex"] = df.SUSPECT_SEX.map({"MALE": "Male", "FEMALE": "Female"}).fillna("Unknown")
df["age"] = pd.to_numeric(df.SUSPECT_REPORTED_AGE, errors="coerce")
df.loc[(df.age < 10) | (df.age > 90), "age"] = np.nan
bins = [0, 14, 17, 24, 34, 44, 54, 120]
labels = ["≤14", "15–17", "18–24", "25–34", "35–44", "45–54", "55+"]
df["agegrp"] = pd.cut(df.age, bins=bins, labels=labels)
df["hour"] = pd.to_datetime(df.STOP_FRISK_TIME.astype(str), format="%H:%M:%S", errors="coerce").dt.hour
df["boro"] = df.STOP_LOCATION_BORO_NAME.str.title()
df["crime"] = df.SUSPECTED_CRIME_DESCRIPTION.str.title()

def rates(g):
    n = len(g)
    return dict(
        n=int(n),
        arrest=round(100 * g.arrest.mean(), 1),
        summons=round(100 * g.summons.mean(), 1),
        noaction=round(100 * g.noaction.mean(), 1),
        frisk=round(100 * g.frisk.mean(), 1),
        search=round(100 * g.search.mean(), 1),
        weapon=round(100 * g.weapon.mean(), 1),
        firearm=round(100 * g.firearm.mean(), 1),
        contraband=round(100 * g.contraband.mean(), 1),
        anyfind=round(100 * g.anyfind.mean(), 1),
    )

R = {"N": N}

# ---------- Q1 demographics
race_order = ["Black", "Hispanic", "White", "Asian / Pacific Islander", "Other", "Unknown"]
pop = {  # NYC DCP 2020 Census, Hispanic of any race; others non-Hispanic
    "Black": 1776891, "Hispanic": 2490350, "White": 2719856,
    "Asian / Pacific Islander": 1373502, "Other": 443591}
poptot = 8804190
R["race"] = []
for r in race_order:
    n = int((df.race == r).sum())
    d = dict(group=r, n=n, share=round(100 * n / N, 1))
    if r in pop:
        d["pop_share"] = round(100 * pop[r] / poptot, 1)
        d["per1000"] = round(1000 * n / pop[r], 2)
    R["race"].append(d)
R["race_detail"] = df.SUSPECT_RACE_DESCRIPTION.fillna("Unknown").value_counts().to_dict()
R["sex"] = [dict(group=s, n=int((df.sex == s).sum()), share=round(100 * (df.sex == s).mean(), 1)) for s in ["Male", "Female", "Unknown"]]
ct = pd.crosstab(df.race, df.sex)
R["race_sex"] = {r: {s: int(ct.loc[r, s]) for s in ct.columns} for r in race_order if r in ct.index}
R["female_share_by_race"] = {r: round(100 * ct.loc[r, "Female"] / ct.loc[r].sum(), 1) for r in race_order if r in ct.index}

# ---------- Q2 temporal
months = ["January","February","March","April","May","June","July","August","September","October","November","December"]
days_in = [31,28,31,30,31,30,31,31,30,31,30,31]
R["month"] = [dict(m=m[:3], n=int((df.MONTH2 == m).sum()), perday=round((df.MONTH2 == m).sum() / d, 1)) for m, d in zip(months, days_in)]
wd = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
R["weekday"] = [dict(d=w[:3], n=int((df.DAY2 == w).sum())) for w in wd]
R["hour"] = [int((df.hour == h).sum()) for h in range(24)]
hm = pd.crosstab(df.DAY2, df.hour).reindex(index=wd, columns=range(24), fill_value=0)
R["heat_wd_hour"] = hm.values.tolist()
# month x race share
R["month_race_black_share"] = [round(100 * (df[df.MONTH2 == m].race == "Black").mean(), 1) for m in months]
R["initiated"] = df.STOP_WAS_INITIATED.str.replace("Based on ", "").value_counts().to_dict()
R["time_buckets"] = {
    "Overnight (12–5am)": int(df.hour.between(0, 5).sum()),
    "Morning (6–11am)": int(df.hour.between(6, 11).sum()),
    "Afternoon (12–5pm)": int(df.hour.between(12, 17).sum()),
    "Evening (6–11pm)": int(df.hour.between(18, 23).sum()),
}

# ---------- Q3 reasons
top = df.crime.value_counts()
R["reasons"] = [dict(reason=k, **rates(df[df.crime == k])) for k in top.index]
R["reasons_top"] = R["reasons"][:12]
act_cols = {
    "Concealed weapon (suspected)": "SUSPECTS_ACTIONS_CONCEALED_POSSESSION_WEAPON_FLAG",
    "Fits a description": "SUSPECTS_ACTIONS_DECRIPTION_FLAG",
    "Proximity to crime scene": "SUSPECTS_ACTIONS_PROXIMITY_TO_SCENE_FLAG",
    "Casing a location": "SUSPECTS_ACTIONS_CASING_FLAG",
    "Acting as lookout": "SUSPECTS_ACTIONS_LOOKOUT_FLAG",
    "Drug transaction": "SUSPECTS_ACTIONS_DRUG_TRANSACTIONS_FLAG",
    "Fits crime pattern": "SUSPECTS_ACTIONS_IDENTIFY_CRIME_PATTERN_FLAG",
    "Other": "SUSPECTS_ACTIONS_OTHER_FLAG",
    "Known to carry weapon": "BACKROUND_CIRCUMSTANCES_SUSPECT_KNOWN_TO_CARRY_WEAPON_FLAG",
    "Violent-crime background": "BACKROUND_CIRCUMSTANCES_VIOLENT_CRIME_FLAG",
}
R["stop_basis"] = []
for k, c in act_cols.items():
    g = df[df[c] == "Y"]
    R["stop_basis"].append(dict(reason=k, **rates(g), share=round(100 * len(g) / N, 1)))
R["stop_basis"].sort(key=lambda d: -d["n"])
R["initiation_outcomes"] = {k: rates(g) for k, g in df.assign(i=df.STOP_WAS_INITIATED.str.replace("Based on ", "")).groupby("i")}
R["explained"] = round(100 * yes("OFFICER_EXPLAINED_STOP_FLAG").mean(), 1)

# ---------- Q4 outcomes
R["overall"] = rates(df)
both = int(((df.arrest == 1) & (df.summons == 1)).sum())
R["outcome_mix"] = dict(
    arrest_only=int(((df.arrest == 1) & (df.summons == 0)).sum()),
    summons_only=int(((df.arrest == 0) & (df.summons == 1)).sum()),
    both=both,
    none=int(df.noaction.sum()),
)
R["by_race"] = {r: rates(df[df.race == r]) for r in race_order if (df.race == r).any()}
R["by_sex"] = {s: rates(df[df.sex == s]) for s in ["Male", "Female"]}
R["by_race_sex"] = {f"{r}|{s}": rates(df[(df.race == r) & (df.sex == s)]) for r in race_order[:5] for s in ["Male", "Female"] if ((df.race == r) & (df.sex == s)).sum() >= 20}
R["force_flags"] = {
    "Handcuffed": round(100 * yes("PHYSICAL_FORCE_HANDCUFF_SUSPECT_FLAG").mean(), 1),
    "Verbal instruction": round(100 * yes("PHYSICAL_FORCE_VERBAL_INSTRUCTION_FLAG").mean(), 1),
    "Weapon drawn": round(100 * yes("PHYSICAL_FORCE_DRAW_POINT_FIREARM_FLAG").mean(), 1),
    "Restraint": round(100 * yes("PHYSICAL_FORCE_RESTRAINT_USED_FLAG").mean(), 1),
    "Other force": round(100 * yes("PHYSICAL_FORCE_OTHER_FLAG").mean(), 1),
}
R["force_by_race"] = {r: round(100 * df[df.race == r].PHYSICAL_FORCE_DRAW_POINT_FIREARM_FLAG.eq("Y").mean(), 1) for r in race_order[:5]}

# ---------- Q5 location
R["boro"] = []
bpop = {"Bronx": 1472654, "Brooklyn": 2736074, "Manhattan": 1694251, "Queens": 2405464, "Staten Island": 495747}
bpct = {  # share of residents: Hispanic, White NH, Black NH, Asian NH
    "Bronx": [54.8, 8.9, 28.5, 4.6], "Brooklyn": [18.9, 35.4, 26.7, 13.6],
    "Manhattan": [23.8, 46.8, 11.8, 13.0], "Queens": [27.8, 22.8, 15.9, 27.3],
    "Staten Island": [19.6, 56.1, 9.4, 11.9]}
for b, p in bpop.items():
    g = df[df.boro == b]
    R["boro"].append(dict(boro=b, n=int(len(g)), per1000=round(1000 * len(g) / p, 2),
        pop=p, pop_black=bpct[b][2], pop_hisp=bpct[b][0], pop_white=bpct[b][1], pop_asian=bpct[b][3],
        stop_black=round(100 * (g.race == "Black").mean(), 1), stop_hisp=round(100 * (g.race == "Hispanic").mean(), 1),
        stop_white=round(100 * (g.race == "White").mean(), 1), stop_asian=round(100 * (g.race == "Asian / Pacific Islander").mean(), 1),
        arrest=round(100 * g.arrest.mean(), 1)))
bd = pd.DataFrame(R["boro"])
R["boro_corr"] = {
    "black_pop_vs_rate": round(float(np.corrcoef(bd.pop_black, bd.per1000)[0, 1]), 2),
    "white_pop_vs_rate": round(float(np.corrcoef(bd.pop_white, bd.per1000)[0, 1]), 2),
    "hisp_pop_vs_rate": round(float(np.corrcoef(bd.pop_hisp, bd.per1000)[0, 1]), 2),
    "black_pop_vs_black_stop": round(float(np.corrcoef(bd.pop_black, bd.stop_black)[0, 1]), 2),
}
pc = df.STOP_LOCATION_PRECINCT.value_counts()
R["precincts"] = []
for p in pc.index[:15]:
    g = df[df.STOP_LOCATION_PRECINCT == p]
    R["precincts"].append(dict(precinct=int(p), n=int(len(g)), boro=g.boro.mode().iat[0],
        black=round(100 * (g.race == "Black").mean(), 1), hisp=round(100 * (g.race == "Hispanic").mean(), 1),
        white=round(100 * (g.race == "White").mean(), 1), asian=round(100 * (g.race == "Asian / Pacific Islander").mean(), 1),
        other=round(100 * (g.race.isin(["Other", "Unknown"])).mean(), 1), arrest=round(100 * g.arrest.mean(), 1)))
R["n_precincts"] = int(pc.size)
R["top15_share"] = round(100 * pc.iloc[:15].sum() / N, 1)
R["top5_share"] = round(100 * pc.iloc[:5].sum() / N, 1)
R["precinct_gt_100"] = int((pc >= 100).sum())
# density map (NY State Plane, feet)
xy = df[["STOP_LOCATION_X", "STOP_LOCATION_Y"]].apply(pd.to_numeric, errors="coerce").dropna()
xy = xy[(xy.STOP_LOCATION_X > 900000) & (xy.STOP_LOCATION_X < 1080000) & (xy.STOP_LOCATION_Y > 110000) & (xy.STOP_LOCATION_Y < 280000)]
cell = 1500
gx = (xy.STOP_LOCATION_X // cell).astype(int); gy = (xy.STOP_LOCATION_Y // cell).astype(int)
cells = pd.DataFrame({"x": gx, "y": gy}).value_counts().reset_index()
cells.columns = ["x", "y", "n"]
R["map"] = dict(cell=cell, cells=cells.values.tolist(), n_mapped=int(len(xy)))

# ---------- Q6 weapons / contraband
R["finds"] = dict(
    weapon=int(df.weapon.sum()), firearm=int(df.firearm.sum()), knife=int(df.knife.sum()),
    otherweapon=int(df.otherweapon.sum()), contraband=int(df.contraband.sum()), anyfind=int(df.anyfind.sum()),
    frisked=int(df.frisk.sum()), searched=int(df.search.sum()),
    frisked_or_searched=int(((df.frisk == 1) | (df.search == 1)).sum()),
    weapon_given_fs=round(100 * df[(df.frisk == 1) | (df.search == 1)].weapon.mean(), 1),
    firearm_given_fs=round(100 * df[(df.frisk == 1) | (df.search == 1)].firearm.mean(), 1),
    weapon_given_nofs=round(100 * df[(df.frisk == 0) & (df.search == 0)].weapon.mean(), 1),
)
R["finds_by_race"] = {r: dict(weapon=R["by_race"][r]["weapon"], firearm=R["by_race"][r]["firearm"], contraband=R["by_race"][r]["contraband"],
    frisk=R["by_race"][r]["frisk"]) for r in race_order[:5]}
# among frisked: weapon rate by race
R["weapon_given_frisk_by_race"] = {r: round(100 * df[(df.race == r) & (df.frisk == 1)].weapon.mean(), 1) for r in race_order[:5]}
R["weapon_given_search_by_race"] = {r: round(100 * df[(df.race == r) & (df.search == 1)].anyfind.mean(), 1) for r in race_order[:5]}
cpw = df.crime == "Cpw"
R["cpw_vs_other"] = dict(
    cpw=rates(df[cpw]), other=rates(df[~cpw]),
    cpw_share_of_weapon_finds=round(100 * df[cpw].weapon.sum() / df.weapon.sum(), 1))

# ---------- Q7 repeat encounters (proxy only: dataset has no person identifier)
sig_cols = ["age", "sex", "race", "SUSPECT_HEIGHT", "SUSPECT_WEIGHT", "SUSPECT_BODY_BUILD_TYPE", "SUSPECT_HAIR_COLOR", "SUSPECT_EYE_COLOR"]
s = df.dropna(subset=["age", "SUSPECT_HEIGHT", "SUSPECT_WEIGHT", "SUSPECT_BODY_BUILD_TYPE", "SUSPECT_HAIR_COLOR", "SUSPECT_EYE_COLOR"]).copy()
s = s[(s.sex != "Unknown") & (s.race != "Unknown")]
s["sig"] = s[sig_cols].astype(str).agg("|".join, axis=1)
sz = s.groupby("sig").size()
dup_sigs = sz[sz > 1]
R["repeat"] = dict(
    n_eligible=int(len(s)), n_profiles=int(sz.size), n_profiles_multi=int(dup_sigs.size),
    stops_in_multi=int(dup_sigs.sum()), share_in_multi=round(100 * dup_sigs.sum() / len(s), 1),
    max_group=int(sz.max()),
    size_dist={int(k): int(v) for k, v in dup_sigs.value_counts().sort_index().items() if k <= 6},
)
sm = s[s.sig.isin(dup_sigs.index)]
R["repeat"]["by_race"] = {r: round(100 * (sm.race == r).sum() / max(1, (s.race == r).sum()), 1) for r in race_order[:5]}
R["repeat"]["arrest_multi"] = round(100 * sm.arrest.mean(), 1)
R["repeat"]["arrest_single"] = round(100 * s[~s.sig.isin(dup_sigs.index)].arrest.mean(), 1)
R["repeat"]["race_mix_multi"] = {r: round(100 * (sm.race == r).mean(), 1) for r in race_order[:5]}
R["repeat"]["race_mix_all"] = {r: round(100 * (s.race == r).mean(), 1) for r in race_order[:5]}
R["repeat"]["same_day_same_precinct"] = None
# Same description, same day + same precinct is more likely a duplicate record of one encounter
s2 = s.copy(); s2["key"] = s2.sig + "|" + s2.STOP_FRISK_DATE.astype(str)
R["repeat"]["same_profile_same_day"] = int((s2.groupby("key").size() > 1).sum())

# ---------- Q8 age
R["age_hist"] = {int(a): int(n) for a, n in df.age.value_counts().sort_index().items()}
R["age_missing"] = int(df.age.isna().sum())
R["age_median"] = float(df.age.median())
R["age_mean"] = round(float(df.age.mean()), 1)
R["age_groups"] = []
for g in labels:
    sub = df[df.agegrp == g]
    R["age_groups"].append(dict(group=g, **rates(sub), share=round(100 * len(sub) / df.age.notna().sum(), 1)))
R["under18_share"] = round(100 * (df.age < 18).sum() / df.age.notna().sum(), 1)
R["under25_share"] = round(100 * (df.age < 25).sum() / df.age.notna().sum(), 1)
top_reasons = ["Cpw", "Robbery", "Petit Larceny", "Assault", "Burglary", "Grand Larceny", "Grand Larceny Auto", "Menacing"]
tmp = df[df.agegrp.notna()].copy()
tmp["reason_g"] = np.where(tmp.crime.isin(top_reasons[:6]), tmp.crime, "All other")
ar = pd.crosstab(tmp.agegrp, tmp.reason_g, normalize="index") * 100
R["age_reason"] = {str(g): {k: round(float(v), 1) for k, v in ar.loc[g].items()} for g in ar.index}
R["age_reason_order"] = top_reasons[:6] + ["All other"]
R["age_race"] = {str(g): {r: round(100 * (tmp[tmp.agegrp == g].race == r).mean(), 1) for r in race_order[:5]} for g in labels}
R["age_sex_female"] = {str(g): round(100 * (tmp[tmp.agegrp == g].sex == "Female").mean(), 1) for g in labels}

# ---------- Q9 officers (no officer ID exists; command code + rank only)
R["officer_note_unique_commands"] = int(df.ISSUING_OFFICER_COMMAND_CODE.nunique())
cc = df.ISSUING_OFFICER_COMMAND_CODE.value_counts()
R["commands"] = []
for c in cc.index[:15]:
    g = df[df.ISSUING_OFFICER_COMMAND_CODE == c]
    R["commands"].append(dict(code=int(c), **rates(g)))
R["commands_top10_share"] = round(100 * cc.iloc[:10].sum() / N, 1)
R["commands_top15_share"] = round(100 * cc.iloc[:15].sum() / N, 1)
df["rank"] = df.ISSUING_OFFICER_RANK.str.strip()
rk = df["rank"].value_counts()
R["ranks"] = [dict(rank=k, **rates(df[df["rank"] == k])) for k in rk.index[:6]]
# spread of arrest rate across commands with >= 50 stops
big = df.groupby("ISSUING_OFFICER_COMMAND_CODE").filter(lambda g: len(g) >= 50)
gb = big.groupby("ISSUING_OFFICER_COMMAND_CODE")
R["command_spread"] = dict(
    n_commands=int(gb.ngroups),
    arrest_min=round(100 * gb.arrest.mean().min(), 1), arrest_max=round(100 * gb.arrest.mean().max(), 1),
    arrest_p25=round(100 * gb.arrest.mean().quantile(.25), 1), arrest_p75=round(100 * gb.arrest.mean().quantile(.75), 1),
    weapon_min=round(100 * gb.weapon.mean().min(), 1), weapon_max=round(100 * gb.weapon.mean().max(), 1),
    noaction_min=round(100 * gb.noaction.mean().min(), 1), noaction_max=round(100 * gb.noaction.mean().max(), 1),
)
# command scatter: n vs no-action rate for commands with >=30 stops
sc = df.groupby("ISSUING_OFFICER_COMMAND_CODE").agg(n=("arrest", "size"), arrest=("arrest", "mean"), weapon=("weapon", "mean"))
sc = sc[sc.n >= 30]
R["command_scatter"] = [[int(i), int(r.n), round(100 * r.arrest, 1), round(100 * r.weapon, 1)] for i, r in sc.iterrows()]

# ---------- Q10 internal 2022 benchmark numbers used against outside history
R["noaction_2022"] = round(100 * df.noaction.mean(), 1)
R["race_nyclu_style"] = {r: round(100 * (df.race == r).mean(), 1) for r in ["Black", "Hispanic", "White"]}

json.dump(R, open(out, "w"), separators=(",", ":"))
print("ok", N, len(json.dumps(R)))
