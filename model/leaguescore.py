"""A league's own linear scoring, composed from the component models' predictions.

The 14 component models (model/components.py) each predict the conditional MEAN of one stat line: targets, receptions,
receiving yards and TDs, carries, rushing yards and TDs, pass attempts / completions / yards / TDs / interceptions, fumbles
lost, and a "misc" bucket (two-point conversions at 2 and special-teams TDs at 6). Points in any LINEAR scoring are then
`sum(weight * predicted component)`, and that is the conditional mean of the league's points.

`from_league_json` turns one `leagues/<slug>/league.json` scoring dict (Sleeper keys; the ESPN league is stored in the same
vocabulary) into a `components.Scoring` plus an honest list of what it could not express:

  * `ignored_thresholds`   nonzero scoring keys that need a distribution, not a mean, or a stat no model predicts: yardage
                           bonuses (100/200-yard rushing and receiving, 300/400-yard passing), long-touchdown bonuses
                           (rush/rec/pass TD of 40+ and 50+ yards), long-play bonuses, first-down and per-distance
                           reception bins, all-purpose fumble (`fum`) and per-incompletion scoring. The repo's own
                           `ff.scoring` applies the yardage thresholds to actual stat lines; a mean cannot.
  * out of scope           kicking, team defence, special-teams units and IDP keys never enter (skill positions only, like
                           everything else in this repo); they are not listed as ignored because nothing skill-position
                           scores on them.

The misc bucket is one model of `2 * two_point + 6 * special_teams_td`, so it composes with ONE weight: the league's
two-point value over 2 when its pass / rush / rec two-point values agree, else 0. A league that scores special-teams TDs
differently (the ESPN league has no such key) still gets them at the two-point ratio: worth about 0.01 points a game, named
in PLAN_MODEL.md.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from model import components as C

REPO = Path(__file__).resolve().parents[1]

# league scoring key -> component label (weights add when two keys feed one component)
DIRECT = {"pass_yd": "y_pass_yds", "pass_td": "y_pass_td", "pass_int": "y_int", "pass_cmp": "y_cmp", "pass_att": "y_att",
          "rush_yd": "y_rush_yds", "rush_td": "y_rush_td", "rush_att": "y_car",
          "rec": "y_rec", "rec_yd": "y_rec_yds", "rec_td": "y_rec_td", "rec_tgt": "y_tgt",
          "fum_lost": "y_fum_lost"}
POSITION_BONUS = {"bonus_rec_te": ("TE", "y_rec"), "bonus_rec_rb": ("RB", "y_rec"), "bonus_rec_wr": ("WR", "y_rec")}
TWO_POINT_KEYS = ("pass_2pt", "rush_2pt", "rec_2pt")
# consumed by the misc bucket (or deliberately carried by it), so never "ignored"
MISC_KEYS = (*TWO_POINT_KEYS, "st_td")
# keys that only a defence, a kicker, a special-teams unit or an IDP scores on (or an offensive fumble recovery, which nobody models):
# out of scope, silently. Anything else nonzero that the composition cannot express is reported as ignored.
OUT_OF_SCOPE_PREFIXES = ("def_", "idp_", "st_", "pts_allow", "yds_allow", "sack", "int_ret", "fum_ret", "blk_", "tkl", "qb_hit",
                         "fg", "xp", "kr_", "pr_", "fum_rec", "bonus_def_", "bonus_sack", "bonus_tkl", "safe")
OUT_OF_SCOPE_EXACT = {"ff", "int", "kr_td", "pr_td"}


@dataclass(frozen=True)
class LeagueScoring:
    slug: str
    name: str
    platform: str
    scoring: C.Scoring
    ignored_thresholds: dict = field(default_factory=dict)     # {scoring key: points} that the composition cannot express
    misc_weight: float = 1.0

    @property
    def is_ppr(self) -> bool:
        """Identical to the PPR composition the flat model predicts (so composed points need no scoring adjustment)."""
        return self.scoring.weights == C.PPR.weights and not self.scoring.position_bonus


def _out_of_scope(key: str) -> bool:
    return key in OUT_OF_SCOPE_EXACT or key.startswith(OUT_OF_SCOPE_PREFIXES)


def compose_weights(scoring: dict) -> tuple[dict, dict, float, dict]:
    """(component weights, {(position, component): points}, misc weight, ignored {key: points}) for a Sleeper-style scoring dict."""
    weights = {c: 0.0 for c in C.TARGETS}
    used = set()
    for key, comp in DIRECT.items():
        if key in scoring:
            weights[comp] += float(scoring[key] or 0.0)
            used.add(key)
    bonus = {}
    for key, (pos, comp) in POSITION_BONUS.items():
        if scoring.get(key):
            bonus[(pos, comp)] = float(scoring[key])
        used.add(key)
    twos = {float(scoring.get(k) or 0.0) for k in TWO_POINT_KEYS}
    misc = (twos.pop() / 2.0) if len(twos) == 1 else 0.0
    weights[C.MISC] = misc
    used.update(MISC_KEYS)
    weights = {c: w for c, w in weights.items() if w != 0.0 or c in C.PPR_WEIGHTS}     # PPR's own keys always stay, even at 0
    ignored = {}
    for key, val in scoring.items():
        if key in used or not val or _out_of_scope(key):
            continue
        ignored[key] = float(val)
    # the yardage / long-TD / first-down keys all land here by construction; `pass_inc` and `fum` are nonzero only if a league scores them
    return weights, bonus, misc, dict(sorted(ignored.items()))


def from_scoring_dict(slug: str, name: str, platform: str, scoring: dict) -> LeagueScoring:
    weights, bonus, misc, ignored = compose_weights(scoring or {})
    return LeagueScoring(slug, name, platform, C.Scoring(slug, weights, bonus), ignored, misc)


def from_league_json(path: Path) -> LeagueScoring:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return from_scoring_dict(d.get("slug") or Path(path).parent.name, d.get("name") or "", d.get("platform") or "sleeper",
                             d.get("scoring") or {})


def load_leagues(root: Path = REPO) -> list[LeagueScoring]:
    """Every `leagues/*/league.json` that carries a scoring dict, in slug order."""
    out = []
    for p in sorted((root / "leagues").glob("*/league.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if d.get("scoring"):
            out.append(from_scoring_dict(d.get("slug") or p.parent.name, d.get("name") or "", d.get("platform") or "sleeper",
                                         d["scoring"]))
    return out


def describe(ls: LeagueScoring) -> str:
    """One line for the run log and the column docs."""
    w = {c: v for c, v in ls.scoring.weights.items() if v != C.PPR_WEIGHTS.get(c)}
    diff = ", ".join(f"{c}={v:g}" for c, v in w.items()) or "same as PPR"
    bon = ", ".join(f"{pos} {comp}+{v:g}" for (pos, comp), v in ls.scoring.position_bonus.items())
    ign = ", ".join(f"{k}={v:g}" for k, v in ls.ignored_thresholds.items())
    return (f"{ls.slug}: vs PPR: {diff}" + (f"; per-position: {bon}" if bon else "")
            + (f"; ignored (threshold/unmodelled): {ign}" if ign else "; nothing ignored"))
