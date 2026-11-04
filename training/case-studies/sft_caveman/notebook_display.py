"""Offline presentation helpers for the Caveman teaching notebook."""
from html import escape
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ARMS = [("base", "Base", "#888e98"), ("skill", "Skill prompt", "#327ca5"), ("adapter", "Adapter", "#e58629")]

def answers(example):
    f = example["fixture"]
    try:
        from IPython.display import HTML, display
    except ImportError:
        print(f["question"])
        print("Given:", "; ".join(f["facts"]))
        for answer in example["answers"]:
            label = next(label for arm, label, _ in ARMS if arm == answer["arm"])
            print(f"\n{label} · {answer['words']} words · {'passes' if answer['pass'] else 'fails'} recorded checks")
            print(answer["text"])
        return
    display(HTML("<h3>" + escape(f["question"]) + "</h3><p><b>Given:</b> " + escape("; ".join(f["facts"])) + "</p>"))
    for r, (_, label, color) in zip(example["answers"], ARMS):
        display(HTML(f"<div style='border-left:4px solid {color};padding:8px 16px;margin:12px 0'>"
                     f"<b>{label}</b> · {r['words']} words · {'passes' if r['pass'] else 'fails'} recorded checks"
                     f"<div style='white-space:pre-wrap;overflow-wrap:anywhere;margin-top:8px'>{escape(r['text'])}</div></div>"))

def figure(rows, fixtures, *, title="Recorded v2c", note="Exploratory result: v2b and v2c used this test set.", path=None, show=True):
    # Resample source groups, retaining every draw and pairing arms within each group.
    groups = defaultdict(list)
    for r in rows:
        groups[fixtures[r['id']]['group_id']].append(r)
    grouped = list(groups.values())
    sums = np.array([[[sum(float(r[k]) for r in g if r['arm']==a) for k in ['pass','words']]
                     for a, _, _ in ARMS] for g in grouped])
    counts = np.array([[sum(r['arm']==a for r in g) for a, _, _ in ARMS] for g in grouped])
    means = sums.sum(axis=0) / counts.sum(axis=0)[:,None]
    rng = np.random.default_rng(29)
    boots=[]
    for _ in range(2000):
        idx=rng.integers(0,len(grouped),len(grouped))
        boots.append(sums[idx].sum(axis=0)/counts[idx].sum(axis=0)[:,None])
    boots=np.asarray(boots)
    fig, axes=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
    for j,(panel_title,unit,scale) in enumerate([('Answer quality','Answers passing all checks (%)',100),('Answer length','Mean words per answer',1)]):
        ax=axes[j]; vals=means[:,j]*scale
        lo,hi=np.quantile(boots[:,:,j]*scale,[.025,.975],axis=0)
        bars=ax.bar([x[1] for x in ARMS],vals,color=[x[2] for x in ARMS],width=.6,
                    yerr=[vals-lo,hi-vals],capsize=4)
        ax.bar_label(bars,labels=[f'{v:.1f}' for v in vals],padding=12)
        ax.set(title=panel_title,ylabel=unit,ylim=(0,110 if j==0 else max(hi)*1.25))
        ax.spines[['top','right']].set_visible(False)
        ax.set_axisbelow(True);ax.grid(axis='y',alpha=.15)
    fig.suptitle(f'{title} · {len(fixtures)} fixtures · 95% group-bootstrap intervals')
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    delta=100*(boots[:,2,0]-boots[:,1,0]);lo,hi=np.quantile(delta,[.025,.975])
    print(f'Adapter minus skill prompt: {100*(means[2,0]-means[1,0]):+.1f} percentage points; 95% paired group interval [{lo:+.1f}, {hi:+.1f}].')
    print(note + ' Words measure length, not billed tokens.')
    return fig


def teaching_story(trace):
    """Show recorded selection coverage, complete candidates, and the training pair."""
    coverage = trace["coverage"]
    fig, ax = plt.subplots(figsize=(9, 3.8), layout="constrained")
    y = np.arange(len(coverage))
    ax.barh(y, [r["tasks"] for r in coverage], color="#dce2e8", label="Available training tasks")
    ax.barh(y, [r["selected"] for r in coverage], color="#327ca5", label="Tasks with a selected answer")
    ax.set_yticks(y, [r["family"].replace("_", " ") for r in coverage])
    ax.set_xlabel("Number of training tasks")
    ax.set_title("Which tasks contributed a teaching answer?")
    ax.legend(loc="lower right")
    for i, row in enumerate(coverage):
        ax.text(row["tasks"] + 8, i, f"{row['selected']}/{row['tasks']}", va="center", fontsize=9)
    ax.set_xlim(0, max(r["tasks"] for r in coverage) * 1.2)
    plt.show()
    print(" · ".join(f"{k}: {v:,}" for k, v in trace["counts"].items()))
    example = trace["example"]
    sections = [("Task and supplied facts", example["request"])]
    selected = example["training_messages"][-1]["content"]
    for candidate in example["candidates"]:
        verdict = "SELECTED" if candidate["text"] == selected else candidate["status"]
        reason = "; ".join(candidate["problems"]) or str(candidate["judge"])
        sections.append((f"Candidate {candidate['sample_index'] + 1}: {verdict} ({candidate['words']} words)",
                         candidate["text"] + "\n\nRecorded checks: " + reason))
    sections.extend(("Training " + m["role"], m["content"]) for m in example["training_messages"])
    try:
        from IPython.display import HTML, display
    except ImportError:
        for title, text in sections:
            print(f"\n{title}\n{text}")
    else:
        for title, text in sections:
            display(HTML("<div style='border-left:3px solid #327ca5;padding:8px 14px;margin:12px 0'>"
                         f"<b>{escape(title)}</b><div style='white-space:pre-wrap;overflow-wrap:anywhere'>"
                         f"{escape(text)}</div></div>"))
    return fig
