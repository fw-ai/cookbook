"""Dataset loaders and rule generators for the Muse decision-classifier case study.

Every loader returns a list of raw examples:
    {"state": str, "question": str, "classes": [(key, description), ...], "gold": key}

`classes` is the full label set for the task. The notebook decides which subset of
classes to show as lettered options (and whether to swap the gold for "none").
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from datasets import load_dataset


@dataclass(frozen=True)
class Task:
    name: str
    family: str
    heldout: bool
    load: Callable[[int, int], list[dict]]  # (n, seed) -> raw examples


def _sample(ds, n: int, seed: int):
    ds = ds.shuffle(seed=seed)
    return ds.select(range(min(n, len(ds))))


def _clip(text: str, limit: int = 1500) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + " ..."


def _hf(path, name=None, split="train"):
    return load_dataset(path, name, split=split)


def _label_names(ds, col="label"):
    return ds.features[col].names


def _simple(path, name, split, text_col, question, classes, *, label_col="label",
            label_map=None, state_prefix="Text: "):
    """Single-text classification where label ids index into `classes`."""

    def load(n, seed):
        ds = _sample(_hf(path, name, split), n, seed)
        out = []
        for r in ds:
            lab = r[label_col]
            if label_map is not None:
                lab = label_map(lab)
                if lab is None:
                    continue
            out.append({
                "state": state_prefix + _clip(r[text_col]),
                "question": question,
                "classes": classes,
                "gold": classes[lab][0],
            })
        return out

    return load


def _pair(path, name, split, a_col, b_col, a_name, b_name, question, classes,
          label_col="label"):
    def load(n, seed):
        ds = _sample(_hf(path, name, split), n, seed)
        out = []
        for r in ds:
            lab = r[label_col]
            if lab is None or lab < 0:
                continue
            out.append({
                "state": f"{a_name}: {_clip(r[a_col], 1200)}\n{b_name}: {_clip(r[b_col], 600)}",
                "question": question,
                "classes": classes,
                "gold": classes[lab][0],
            })
        return out

    return load


def _named_labels(path, name, split, text_col, question, describe, label_col="label"):
    """Classification whose label names come from the dataset features."""

    def load(n, seed):
        full = _hf(path, name, split)
        names = _label_names(full, label_col)
        classes = [(nm, describe(nm)) for nm in names]
        ds = _sample(full, n, seed)
        return [{
            "state": "Text: " + _clip(r[text_col]),
            "question": question,
            "classes": classes,
            "gold": classes[r[label_col]][0],
        } for r in ds]

    return load


# --- hand-written loaders for datasets with unusual shapes --------------------------------

def _boolq(n, seed):
    ds = _sample(_hf("google/boolq", split="train"), n, seed)
    classes = [("yes", "Yes, the passage supports answering yes."),
               ("no", "No, the passage supports answering no.")]
    return [{
        "state": f"Passage: {_clip(r['passage'], 1200)}",
        "question": f"Based only on the passage: {r['question'].rstrip('?')}?",
        "classes": classes,
        "gold": "yes" if r["answer"] else "no",
    } for r in ds]


def _wic(n, seed):
    ds = _sample(_hf("aps/super_glue", "wic", "train"), n, seed)
    classes = [("same", "The word has the same meaning in both sentences."),
               ("different", "The word has different meanings in the two sentences.")]
    return [{
        "state": f"Word: {r['word']}\nSentence 1: {r['sentence1']}\nSentence 2: {r['sentence2']}",
        "question": "Is the word used with the same meaning in both sentences?",
        "classes": classes,
        "gold": classes[0][0] if r["label"] == 1 else classes[1][0],
    } for r in ds]


def _clinc(n, seed):
    full = _hf("clinc/clinc_oos", "plus", "train")
    names = _label_names(full, "intent")
    classes = [(nm, "The user intent is: " + nm.replace("_", " ") + ".") for nm in names]
    ds = _sample(full, n, seed)
    return [{
        "state": "User message: " + r["text"],
        "question": "Which listed intent best matches the user's message?",
        "classes": classes,
        "gold": classes[r["intent"]][0],
    } for r in ds]


def _newsgroups(n, seed):
    full = _hf("SetFit/20_newsgroups", split="train")
    full = full.filter(lambda r: len(r["text"].split()) >= 20)
    names = sorted(set(full["label_text"]))
    classes = [(nm, _topic_desc(nm)) for nm in names]
    ds = _sample(full, n, seed)
    return [{
        "state": "Post: " + _clip(r["text"]),
        "question": "Which newsgroup was this post written for?",
        "classes": classes,
        "gold": r["label_text"],
    } for r in ds]


def _commonsense_qa(n, seed):
    ds = _sample(_hf("tau/commonsense_qa", split="train"), n, seed)
    out = []
    for r in ds:
        texts = r["choices"]["text"]
        labels = r["choices"]["label"]
        classes = [(f"choice_{i}", t) for i, t in enumerate(texts)]
        out.append({
            "state": "Question context: " + r["question"],
            "question": "Which answer is most sensible?",
            "classes": classes,
            "gold": classes[labels.index(r["answerKey"])][0],
            "fixed_options": True,
        })
    return out


def _arc_easy(n, seed):
    ds = _sample(_hf("allenai/ai2_arc", "ARC-Easy", "train"), n, seed)
    out = []
    for r in ds:
        texts, labels = r["choices"]["text"], r["choices"]["label"]
        if r["answerKey"] not in labels:
            continue
        classes = [(f"choice_{i}", t) for i, t in enumerate(texts)]
        out.append({
            "state": "Science question: " + r["question"],
            "question": "Which answer is correct?",
            "classes": classes,
            "gold": classes[labels.index(r["answerKey"])][0],
            "fixed_options": True,
        })
    return out


STAR_CLASSES = [(f"{i}_star", f"The reviewer would give {i} star{'s' if i > 1 else ''} out of 5.")
                for i in range(1, 6)]
SENT3_CLASSES = [("negative", "The reviewer is negative overall."),
                 ("neutral", "The reviewer is mixed or neutral."),
                 ("positive", "The reviewer is positive overall.")]
BIN_CLASSES = [("positive", "The review is positive (4 or 5 stars)."),
               ("not_positive", "The review is not positive (3 stars or fewer).")]


def _app_reviews(view):
    def load(n, seed):
        full = _hf("sealuzh/app_reviews", split="train")
        full = full.filter(lambda r: r["review"] and len(r["review"].split()) >= 4)
        ds = _sample(full, n, seed)
        out = []
        for r in ds:
            star = int(r["star"])
            if view == "stars5":
                classes, gold, q = STAR_CLASSES, STAR_CLASSES[star - 1][0], \
                    "How many stars did this app reviewer give?"
            elif view == "sent3":
                classes, q = SENT3_CLASSES, "What is the overall sentiment of this app review?"
                gold = "negative" if star <= 2 else "neutral" if star == 3 else "positive"
            else:
                classes, q = BIN_CLASSES, "Is this app review positive?"
                gold = "positive" if star >= 4 else "not_positive"
            out.append({
                "state": f"App: {r['package_name']}\nReview: {_clip(r['review'])}",
                "question": q,
                "classes": classes,
                "gold": gold,
                "fixed_options": True,
            })
        return out

    return load


# --- programmatic rule tasks (gold is computed, not labeled) -------------------------------

YES_NO = [("yes", "Yes, the rule is satisfied."), ("no", "No, the rule is not satisfied.")]


def _balanced(gen):
    """Oversample a raw generator until yes/no are 50/50, so 'always no' is not a free win."""

    def load(n, seed):
        pool = gen(n * 8, seed)
        yes = [e for e in pool if e["gold"] == "yes"][: n // 2]
        no = [e for e in pool if e["gold"] == "no"][: n - len(yes)]
        out = yes + no
        random.Random(seed).shuffle(out)
        return out

    return load


def _refund(n, seed):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        days = rng.randint(1, 60)
        opened = rng.random() < 0.5
        defective = rng.random() < 0.3
        receipt = rng.random() < 0.7
        ok = receipt and days <= 30 and (not opened or defective)
        out.append({
            "state": (f"Policy: refunds require a receipt, a purchase within 30 days, and the item "
                      f"must be unopened unless it is defective.\nCase: bought {days} days ago; "
                      f"{'opened' if opened else 'unopened'}; "
                      f"{'defective' if defective else 'works fine'}; "
                      f"{'has' if receipt else 'no'} receipt."),
            "question": "Is this customer eligible for a refund under the policy?",
            "classes": YES_NO, "gold": "yes" if ok else "no", "fixed_options": True,
        })
    return out


def _car_rental(n, seed):
    rng = random.Random(seed + 1)
    out = []
    for _ in range(n):
        age = rng.randint(17, 70)
        insured = rng.random() < 0.5
        license_years = rng.randint(0, max(0, age - 16))
        ok = license_years >= 1 and (age >= 25 or (age >= 21 and insured))
        out.append({
            "state": (f"Rule: renters need at least 1 year of license history, and must be 25+, or "
                      f"21-24 with their own insurance.\nApplicant: age {age}, licensed "
                      f"{license_years} year(s), {'has' if insured else 'no'} personal insurance."),
            "question": "May this applicant rent a car?",
            "classes": YES_NO, "gold": "yes" if ok else "no", "fixed_options": True,
        })
    return out


ROUTE_CLASSES = [("fraud", "Send to the fraud team."),
                 ("billing_escalation", "Send to billing escalations."),
                 ("billing", "Send to standard billing."),
                 ("tech", "Send to technical support.")]


def _ticket_routing(n, seed):
    rng = random.Random(seed + 2)
    phrases = {
        "unauthorized": "I see a charge I never authorized",
        "refund": "I would like a refund for my last order",
        "login": "I cannot log in to my account",
        "invoice": "my invoice total looks wrong",
    }
    out = []
    for _ in range(n):
        kind = rng.choice(list(phrases))
        amount = rng.choice([12, 45, 99, 250, 480, 520, 900, 1500])
        if kind == "unauthorized":
            gold = "fraud"
        elif kind in ("refund", "invoice"):
            gold = "billing_escalation" if amount > 500 else "billing"
        else:
            gold = "tech"
        out.append({
            "state": (f"Routing rules: unauthorized charges go to fraud; refund or invoice issues "
                      f"over $500 go to billing escalations, otherwise standard billing; account "
                      f"access problems go to tech support.\nTicket: \"Hi, {phrases[kind]}. The "
                      f"amount involved is ${amount}.\""),
            "question": "Which team should this ticket be routed to?",
            "classes": ROUTE_CLASSES, "gold": gold,
        })
    return out


def _listing_rule(n, seed):
    rng = random.Random(seed + 3)
    items = ["mountain bike", "oak desk", "used iPhone 13", "baby stroller", "guitar amp"]
    contacts = ["call me at 415-555-0199", "email me: seller88@mail.com",
                "text 212 555 0143", "reach me at jo.smith(at)gmail(dot)com"]
    benign = ["pickup only", "price is firm", "lightly used", "comes with charger",
              "message me through the app"]
    out = []
    for _ in range(n):
        violates = rng.random() < 0.5
        extra = rng.choice(contacts) if violates else rng.choice(benign)
        text = f"Selling a {rng.choice(items)} for ${rng.randint(20, 900)}, {extra}."
        out.append({
            "state": ("Marketplace rule: listings may not contain phone numbers or email "
                      f"addresses, including obfuscated ones.\nListing: {text}"),
            "question": "Does this listing violate the rule?",
            "classes": [("violates", "Yes, the listing violates the rule."),
                        ("ok", "No, the listing follows the rule.")],
            "gold": "violates" if violates else "ok", "fixed_options": True,
        })
    return out


# --- harder look-alike tasks: robust NLI, commonsense choice, hallucination, finance, sarcasm, verse ---

NLI3_KEYS = {"entailment": 0, "neutral": 1, "contradiction": 2}


def _wanli(n, seed):
    ds = _sample(_hf("alisawuffles/WANLI", split="train"), n, seed)
    return [{"state": f"Premise: {_clip(r['premise'], 1200)}\nHypothesis: {_clip(r['hypothesis'], 600)}",
             "question": "How does the hypothesis relate to the premise?", "classes": NLI3,
             "gold": r["gold"]} for r in ds]


def _snli(n, seed):
    ds = _sample(_hf("stanfordnlp/snli", split="train").filter(lambda r: r["label"] in (0, 1, 2)), n, seed)
    return [{"state": f"Premise: {r['premise']}\nHypothesis: {r['hypothesis']}",
             "question": "How does the hypothesis relate to the premise?", "classes": NLI3,
             "gold": NLI3[r["label"]][0]} for r in ds]


def _scitail(n, seed):
    ds = _sample(_hf("allenai/scitail", "tsv_format", "train"), n, seed)
    classes = [("supported", "The evidence supports the claim."),
               ("not_supported", "The evidence does not establish the claim.")]
    return [{"state": f"Evidence: {r['premise']}\nClaim: {r['hypothesis']}",
             "question": "Does the evidence support the claim?", "classes": classes,
             "gold": "supported" if r["label"] == "entails" else "not_supported"} for r in ds]


DOC_CLASSES = [("supported", "The document supports the statement."),
               ("contradicted", "The document contradicts the statement."),
               ("not_mentioned", "The document does not settle the statement either way.")]


def _doc_nli(n, seed):
    # ContractNLI shape: one long document, one statement about a single passage in it.
    pool = _sample(_hf("nyu-mll/multi_nli", split="train"), n * 5, seed + 101)
    rows = list(pool)
    rng = random.Random(seed + 101)
    out = []
    for i in range(n):
        chunk = rows[i * 5:(i + 1) * 5]
        target = rng.randrange(len(chunk))
        doc = "\n".join(f"({j + 1}) {_clip(r['premise'], 400)}" for j, r in enumerate(chunk))
        lab = chunk[target]["label"]
        out.append({"state": f"Document:\n{doc}\nStatement: {chunk[target]['hypothesis']}",
                    "question": "Does the document support, contradict, or not settle the statement?",
                    "classes": DOC_CLASSES, "gold": DOC_CLASSES[[0, 2, 1][lab]][0], "fixed_options": True})
    return out


def _siqa(n, seed):
    ds = _sample(_hf("tasksource/social_i_qa", split="train"), n, seed)
    return [{"state": f"Situation: {r['context']}\nQuestion: {r['question']}",
             "question": "Which answer fits the situation best?",
             "classes": [("choice_0", r["answerA"]), ("choice_1", r["answerB"]), ("choice_2", r["answerC"])],
             "gold": f"choice_{int(r['label'])}", "fixed_options": True} for r in ds]


def _piqa(n, seed):
    ds = _sample(_hf("baber/piqa", split="train"), n, seed)
    return [{"state": f"Goal: {r['goal']}", "question": "Which solution achieves the goal?",
             "classes": [("choice_0", r["sol1"]), ("choice_1", r["sol2"])],
             "gold": f"choice_{int(r['label'])}", "fixed_options": True} for r in ds]


def _copa(n, seed):
    ds = _sample(_hf("aps/super_glue", "copa", "train"), n, seed)
    return [{"state": f"Event: {r['premise']}",
             "question": f"Which is the more plausible {r['question']}?",
             "classes": [("choice_0", r["choice1"]), ("choice_1", r["choice2"])],
             "gold": f"choice_{int(r['label'])}", "fixed_options": True} for r in ds]


HALU_CLASSES = [("faithful", "Everything in the answer is supported by the source."),
                ("hallucinated", "The answer contains something the source does not support.")]


def _halueval(n, seed):
    rng = random.Random(seed + 7)
    per = n // 3 + 1
    out = []
    for cfg, src, ask, right, wrong in (
            ("qa", "knowledge", "question", "right_answer", "hallucinated_answer"),
            ("summarization", "document", None, "right_summary", "hallucinated_summary"),
            ("dialogue", "knowledge", "dialogue_history", "right_response", "hallucinated_response")):
        for r in _sample(_hf("pminervini/HaluEval", cfg, "data"), per, seed):
            bad = rng.random() < 0.5
            ctx = f"Source: {_clip(r[src], 1200)}" + (f"\nContext: {_clip(r[ask], 500)}" if ask else "")
            out.append({"state": f"{ctx}\nAnswer: {_clip(r[wrong if bad else right], 500)}",
                        "question": "Does the answer contain anything the source does not support?",
                        "classes": HALU_CLASSES, "gold": "hallucinated" if bad else "faithful",
                        "fixed_options": True})
    rng.shuffle(out)
    return out[:n]


FIN3 = [("negative", "Negative about the company."), ("neutral", "Neutral about the company."),
        ("positive", "Positive about the company.")]


def _fin_sentiment(n, seed):
    fpb = _sample(_hf("atrost/financial_phrasebank", split="train"), n // 2, seed)
    out = [{"state": f"Financial news: {r['sentence']}",
            "question": "Is this news positive, negative or neutral for the company it describes?",
            "classes": FIN3, "gold": FIN3[r["label"]][0], "fixed_options": True} for r in fpb]
    fiqa = _sample(_hf("TheFinAI/fiqa-sentiment-classification", split="train"), n - len(out), seed)
    for r in fiqa:
        s = r["score"]
        gold = "positive" if s > 0.1 else "negative" if s < -0.1 else "neutral"
        out.append({"state": f"Financial news: {r['sentence']}\nCompany: {r['target']}",
                    "question": f"Is this news positive, negative or neutral for {r['target']}?",
                    "classes": FIN3, "gold": gold, "fixed_options": True})
    random.Random(seed).shuffle(out)
    return out


def _sarcasm_headlines(n, seed):
    ds = _sample(_hf("raquiba/Sarcasm_News_Headline", split="train"), n, seed)
    classes = [("not_sarcastic", "The headline is sincere."), ("sarcastic", "The headline is sarcastic.")]
    return [{"state": f"Headline: {r['headline']}", "question": "Is this headline sarcastic?",
             "classes": classes, "gold": classes[int(r["is_sarcastic"])][0], "fixed_options": True} for r in ds]


LEAR_URL = "https://www.gutenberg.org/cache/epub/13650/pg13650.txt"


def _lear_limericks():
    import re
    import urllib.request
    text = urllib.request.urlopen(LEAR_URL, timeout=60).read().decode("utf-8").replace("\r", "")
    stanzas = [s.strip() for s in re.split(r"\n\s*\n", text)]
    return [[" ".join(line.split()) for line in s.splitlines()] for s in stanzas
            if s.startswith("There was") and 4 <= len(s.splitlines()) <= 6]


def _limerick_pairs(n, seed):
    # BPoMP shape: which of two versions is the original? The other has its rhyme, meter or order broken.
    rng = random.Random(seed + 13)
    poems = _lear_limericks()
    enders = [p[-1].split()[-1].strip(".,;:!?") for p in poems]
    fillers = ["very", "rather", "quite", "truly", "indeed", "really", "all"]
    out = []
    for i in range(n):
        poem = list(poems[i % len(poems)])
        kind = rng.choice(["rhyme", "meter", "order"])
        broken = list(poem)
        if kind == "rhyme":
            words = broken[-1].split()
            words[-1] = rng.choice([e for e in enders if e.lower() != words[-1].strip(".,;:!?").lower()]) + "."
            broken[-1] = " ".join(words)
        elif kind == "meter":
            j = rng.randrange(len(broken))
            words = broken[j].split()
            for _ in range(3):
                words.insert(rng.randrange(1, len(words)), rng.choice(fillers))
            broken[j] = " ".join(words)
        else:
            a, b = rng.sample(range(len(broken)), 2)
            broken[a], broken[b] = broken[b], broken[a]
        first_is_original = rng.random() < 0.5
        first, second = (poem, broken) if first_is_original else (broken, poem)
        out.append({"state": "Version 1:\n" + "\n".join(first) + "\n\nVersion 2:\n" + "\n".join(second),
                    "question": "One version is the original limerick; the other has its rhyme, meter or sense broken. "
                                "Which is the original?",
                    "classes": [("version_1", "Version 1 is the original."), ("version_2", "Version 2 is the original.")],
                    "gold": "version_1" if first_is_original else "version_2", "fixed_options": True})
    return out


def _as_yes_no(base):
    # Many real decisions arrive as a statement to judge with options Yes / No.
    def load(n, seed):
        rng = random.Random(seed + 29)
        out = []
        for ex in base(n, seed):
            key, desc = rng.choice(ex["classes"])
            out.append({"state": ex["state"], "question": f"{ex['question']} Is this true: {desc}",
                        "classes": [("yes", "Yes"), ("no", "No")], "gold": "yes" if key == ex["gold"] else "no",
                        "fixed_options": True})
        return out

    return load


EXTRA_TASKS = [
    ("wanli", "entailment", _wanli), ("snli", "entailment", _snli), ("scitail", "entailment", _scitail),
    ("doc_nli", "entailment", _doc_nli), ("social_iqa", "paraphrase_qa", _siqa), ("piqa", "paraphrase_qa", _piqa),
    ("copa", "paraphrase_qa", _copa), ("halueval", "moderation", _halueval), ("fin_sentiment", "sentiment", _fin_sentiment),
    ("sarcasm_headlines", "sentiment", _sarcasm_headlines), ("limerick_pairs", "arts", _limerick_pairs),
    ("yes_no_halueval", "yes_no", _as_yes_no(_halueval)), ("yes_no_scitail", "yes_no", _as_yes_no(_scitail)),
]


# --- registry --------------------------------------------------------------------------------

NLI3 = [("entailment", "The hypothesis follows from the premise."),
        ("neutral", "The hypothesis might or might not be true given the premise."),
        ("contradiction", "The hypothesis contradicts the premise.")]
ENT2 = [("entailment", "The second text follows from the first."),
        ("not_entailment", "The second text does not follow from the first.")]
PARA2 = [("different", "The two texts do not mean the same thing."),
         ("paraphrase", "The two texts mean the same thing.")]
SST5 = [("very_negative", "Very negative."), ("negative", "Negative."), ("neutral", "Neutral."),
        ("positive", "Positive."), ("very_positive", "Very positive.")]
POL2 = [("negative", "The sentiment is negative."), ("positive", "The sentiment is positive.")]
TWEET_SENT = [("negative", "Negative."), ("neutral", "Neutral."), ("positive", "Positive.")]
AG = [("world", "World news."), ("sports", "Sports."), ("business", "Business."),
      ("sci_tech", "Science and technology.")]


def _topic_desc(name: str) -> str:
    return "The text is about: " + name.replace("_", " ").replace(".", " / ") + "."


TASKS: list[Task] = [
    # entailment / reading
    Task("mnli", "entailment", False, _pair("nyu-mll/multi_nli", None, "train", "premise",
         "hypothesis", "Premise", "Hypothesis", "How does the hypothesis relate to the premise?", NLI3)),
    Task("boolq", "entailment", False, _boolq),
    Task("rte", "entailment", False, _pair("nyu-mll/glue", "rte", "train", "sentence1", "sentence2",
         "Text", "Claim", "Does the claim follow from the text?", ENT2)),
    Task("qnli", "entailment", False, _pair("nyu-mll/glue", "qnli", "train", "question", "sentence",
         "Question", "Sentence", "Does the sentence answer the question?",
         [("answers", "The sentence contains the answer."),
          ("does_not_answer", "The sentence does not contain the answer.")])),
    Task("wic", "entailment", False, _wic),
    # intent / routing
    Task("banking77", "intent", False, _named_labels("legacy-datasets/banking77", None, "train", "text",
         "Which listed banking intent best matches the customer's message?",
         lambda nm: "The customer intent is: " + nm.replace("_", " ") + ".")),
    Task("clinc150", "intent", False, _clinc),
    Task("ticket_routing", "rules", False, _ticket_routing),
    # topic
    Task("ag_news", "topic", False, _simple("fancyzhx/ag_news", None, "train", "text",
         "Which section does this news item belong to?", AG, state_prefix="News item: ")),
    Task("dbpedia14", "topic", False, _named_labels("fancyzhx/dbpedia_14", None, "train", "content",
         "What kind of entity does this encyclopedia entry describe?", _topic_desc)),
    Task("yahoo_topics", "topic", False, _named_labels("community-datasets/yahoo_answers_topics",
         None, "train", "question_title", "Which category does this question belong to?",
         _topic_desc, label_col="topic")),
    Task("newsgroups20", "topic", False, _newsgroups),
    # sentiment
    Task("sst5", "sentiment", False, _simple("SetFit/sst5", None, "train", "text",
         "What is the sentiment of this movie review snippet?", SST5, state_prefix="Review: ")),
    Task("imdb", "sentiment", False, _simple("stanfordnlp/imdb", None, "train", "text",
         "Is this movie review positive or negative?", POL2, state_prefix="Review: ")),
    Task("yelp_polarity", "sentiment", False, _simple("fancyzhx/yelp_polarity", None, "train",
         "text", "Is this business review positive or negative?", POL2, state_prefix="Review: ")),
    Task("tweet_sentiment", "sentiment", False, _simple("cardiffnlp/tweet_eval", "sentiment",
         "train", "text", "What is the sentiment of this tweet?", TWEET_SENT, state_prefix="Tweet: ")),
    # moderation
    Task("tweet_offensive", "moderation", False, _simple("cardiffnlp/tweet_eval", "offensive",
         "train", "text", "Is this tweet offensive?",
         [("not_offensive", "Not offensive."), ("offensive", "Offensive.")], state_prefix="Tweet: ")),
    Task("tweet_hate", "moderation", False, _simple("cardiffnlp/tweet_eval", "hate", "train", "text",
         "Does this tweet contain hate speech against immigrants or women?",
         [("not_hate", "No hate speech."), ("hate", "Contains hate speech.")], state_prefix="Tweet: ")),
    Task("listing_rule", "rules", False, _listing_rule),
    # paraphrase / QA
    Task("qqp", "paraphrase_qa", False, _pair("nyu-mll/glue", "qqp", "train", "question1", "question2",
         "Question 1", "Question 2", "Are these two questions asking the same thing?", PARA2)),
    Task("mrpc", "paraphrase_qa", False, _pair("nyu-mll/glue", "mrpc", "train", "sentence1",
         "sentence2", "Sentence 1", "Sentence 2", "Do these two sentences mean the same thing?", PARA2)),
    Task("paws", "paraphrase_qa", False, _pair("google-research-datasets/paws", "labeled_final",
         "train", "sentence1", "sentence2", "Sentence 1", "Sentence 2",
         "Do these two sentences mean the same thing?", PARA2)),
    Task("commonsense_qa", "paraphrase_qa", False, _commonsense_qa),
    Task("arc_easy", "paraphrase_qa", False, _arc_easy),
    # rules
    Task("refund_policy", "rules", False, _balanced(_refund)),
    Task("car_rental", "rules", False, _balanced(_car_rental)),
    # held out: never trained on
    Task("app_reviews_stars5", "heldout", True, _app_reviews("stars5")),
    Task("app_reviews_sent3", "heldout", True, _app_reviews("sent3")),
    Task("app_reviews_binary", "heldout", True, _app_reviews("binary")),
    Task("yelp_stars5", "heldout", True, _simple("Yelp/yelp_review_full", None, "train", "text",
         "How many stars did this business reviewer give?", STAR_CLASSES, state_prefix="Review: ")),
    Task("emotion", "heldout", True, _named_labels("dair-ai/emotion", None, "train", "text",
         "Which emotion does the writer express?", lambda nm: f"The writer feels {nm}.")),
    Task("tweet_irony", "heldout", True, _simple("cardiffnlp/tweet_eval", "irony", "train", "text",
         "Is this tweet ironic?", [("not_ironic", "Not ironic."), ("ironic", "Ironic.")],
         state_prefix="Tweet: ")),
]

# Base tasks first, then the extra tasks; order fixes the shared random stream.
# followed by the new external tasks.
ALL_TASKS: list[Task] = TASKS + [Task(name, family, False, fn) for name, family, fn in EXTRA_TASKS]
