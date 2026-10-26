"""Summarise a backtest log. Two views are reported side by side:
  STRICT    - verdicts exactly as the pilot script parsed them
  RECOVERED - for PARSE_ERROR rows, the verdict is read from the
              "thesis_status" field inside llm_reasoning (the model's own
              output, only the closing brace was missing).
Scoring rule (unchanged from the pilot): HOLDS correct if GROWTH, BROKEN
correct if DECLINE, WEAKENING not scored."""
import json
import re
import sys
from collections import Counter

PATH = sys.argv[1] if len(sys.argv) > 1 else "results/week9_backtest_scaled_log.json"
VALID = ("HOLDS", "WEAKENING", "BROKEN")


def load(path):
    data = json.load(open(path))
    if isinstance(data, dict):
        for k in ("cases", "results", "records"):
            if k in data:
                return data[k]
    return data


def recovered(row):
    v = row.get("llm_verdict")
    if v in VALID:
        return v
    m = re.search(r'"thesis_status"\s*:\s*"(HOLDS|WEAKENING|BROKEN)"',
                  row.get("llm_reasoning") or "")
    return m.group(1) if m else None


def report(name, rows, verdict_fn):
    print("\n##### %s #####" % name)
    usable = [r for r in rows if r.get("actual_outcome") in ("GROWTH", "DECLINE")]
    pairs = [(verdict_fn(r), r["actual_outcome"]) for r in usable]
    dist = Counter(v if v else "NO_VERDICT" for v, _ in pairs)
    out = Counter(o for _, o in pairs)
    print("cases with an outcome:", len(pairs))
    print("verdict distribution :", dict(dist))
    print("outcome distribution :", dict(out))
    scored = [(v, o) for v, o in pairs
              if (v == "HOLDS") or (v == "BROKEN")]
    correct = sum(1 for v, o in scored
                  if (v == "HOLDS" and o == "GROWTH") or (v == "BROKEN" and o == "DECLINE"))
    print("scored (HOLDS/BROKEN): %d, correct: %d" % (len(scored), correct))
    if scored:
        maj = Counter(o for _, o in scored).most_common(1)[0]
        print("accuracy on scored   : %.3f" % (correct / len(scored)))
        print("majority baseline    : %.3f (always predict %s on the same scored cases)"
              % (maj[1] / len(scored), maj[0]))
    if pairs:
        mo = out.most_common(1)[0]
        print("baseline on all %d   : %.3f (always %s)" % (len(pairs), mo[1] / len(pairs), mo[0]))
    for cls, hit in (("GROWTH", ("HOLDS",)), ("DECLINE", ("BROKEN",))):
        sub = [v for v, o in pairs if o == cls]
        if sub:
            h = sum(1 for v in sub if v in hit)
            print("recall %-7s (%s): %d/%d" % (cls, "/".join(hit), h, len(sub)))
    dec = [v for v, o in pairs if o == "DECLINE"]
    if dec:
        flagged = sum(1 for v in dec if v in ("WEAKENING", "BROKEN"))
        print("DECLINE cases flagged WEAKENING or BROKEN: %d/%d" % (flagged, len(dec)))
        print("DECLINE cases called HOLDS: %d/%d" % (sum(1 for v in dec if v == "HOLDS"), len(dec)))
    print("WEAKENING returned: %d, BROKEN returned: %d"
          % (dist.get("WEAKENING", 0), dist.get("BROKEN", 0)))


def main():
    rows = load(PATH)
    print("log:", PATH, "| rows:", len(rows))
    print("direction_score values:", dict(Counter(r.get("direction_score", "n/a") for r in rows)))
    print("\nFailed / skipped / unusable rows:")
    for r in rows:
        if r.get("actual_outcome") not in ("GROWTH", "DECLINE") or r.get("error"):
            print("  %s %s: %s" % (r.get("ticker"), r.get("cutoff_date"),
                                   r.get("error") or r.get("skip_reason") or r.get("direction_score")))
    report("STRICT (script-parsed verdicts)", rows,
           lambda r: r.get("llm_verdict") if r.get("llm_verdict") in VALID else None)
    report("RECOVERED (verdict read from reasoning text)", rows, recovered)
    print("\nPer-case table:")
    print("%-10s %-11s %-11s %-10s %-8s %s" % ("ticker", "cutoff", "strict", "recovered", "outcome", "fwd_ret"))
    for r in rows:
        print("%-10s %-11s %-11s %-10s %-8s %s" % (
            r.get("ticker"), r.get("cutoff_date"), r.get("llm_verdict"),
            recovered(r), r.get("actual_outcome"), r.get("actual_forward_return")))
    lb = [(r.get("ticker"), r.get("cutoff_date"), r["features"].get("lookback_days_actual"))
          for r in rows if isinstance(r.get("features"), dict)
          and (r["features"].get("lookback_days_actual") or 252) < 252]
    if lb:
        print("\nCases with lookback < 252 days:", lb)


main()
