"""PersonaPlex service-scenario training-data construction (Workstream A of the PersonaPlex
replication; see ``projects/2026-01-speech-llm/personaplex.md`` and the plan file).

PersonaPlex (arXiv 2602.06053) trains its full-duplex model partly on **service-role** dialogs:
a customer talks to a named service agent that must stay in character and grounded in a given
role context. Their Service-Duplex-Bench probes four behaviors per scenario -- *proper-noun
recall*, *context adherence*, *unfulfillable-request handling*, and *customer-rudeness
management*. The bench data itself is unreleased, and so is the training data, so we reconstruct
the **generation recipe** from the paper's description (paraphrased, not their exact prompts).

Design (mirrors ``moshirag_data.py``): this module is the data spike + the reusable building
blocks. It (1) defines a service-role **taxonomy** (domain -> agent role -> scenario kinds +
fact slots), (2) a deterministic, LLM-free **seed generator** that samples >=50 grounded
scenarios (agent name, role context, a probe question, the expected handling), (3) the
**service dialogue-instruction templates** the generator LLM uses, and (4) registers a
``"service"`` dialogue adapter so the *existing* ``HfToDialogue`` machinery
(``DATASET_ADAPTER_REGISTRY``) drives the LLM round-robin / sharding / cleaning unchanged --
only the prompt framing is new. The synthesis (Chatterbox multi-speaker), annotate, and
training paths are all reused downstream.

Run ``python -m i6_experiments.users.dorian_koch.speech_llm.personaplex_service_data`` (login
node, no GPU) to print sample scenarios + the user-messages that would be sent to the generator.
"""

from __future__ import annotations

import hashlib
import json

from sisyphus import Job, Task

from .hf_to_dialogue import _COMMON_SUFFIX, register_adapter

# ---------------------------------------------------------------------------
# Service-role taxonomy.  Each domain carries the agent's role, a pool of
# scenario kinds, and "fact slots" -- the grounded proper nouns / policy
# details the agent must recall and adhere to.  Combinatorial expansion across
# domains x scenario-kinds x probe-categories yields well over 50 scenarios.
# ---------------------------------------------------------------------------

SERVICE_DOMAINS: dict[str, dict] = {
    "health_insurance": {
        "agent_role": "a health-insurance support agent",
        "scenarios": ["a claim-status inquiry", "a coverage question", "a deductible question"],
        "facts": {
            "plan_name": ["SilverCare 200", "BlueShield Flex", "VitalPlus PPO"],
            "claim_id": ["CLM-48213", "CLM-90017", "CLM-33420"],
            "deductible": ["$1,500", "$750", "$2,000"],
        },
    },
    "airline": {
        "agent_role": "an airline customer-service agent",
        "scenarios": ["a rebooking request", "a baggage-fee question", "a refund inquiry"],
        "facts": {
            "booking_ref": ["QX7T2P", "LM9KD4", "ZB3R8N"],
            "flight_no": ["flight UA 482", "flight DL 1170", "flight AA 96"],
            "fare_class": ["a Basic Economy fare", "a Main Cabin fare", "a First Class fare"],
        },
    },
    "bank": {
        "agent_role": "a retail-bank support agent",
        "scenarios": ["a disputed charge", "a card-replacement request", "an overdraft question"],
        "facts": {
            "account_type": ["a Premier Checking account", "a Basic Savings account", "a Student account"],
            "last_four": ["ending in 4471", "ending in 8820", "ending in 0093"],
            "dispute_id": ["DSP-71109", "DSP-22845", "DSP-50031"],
        },
    },
    "telecom": {
        "agent_role": "a mobile-carrier support agent",
        "scenarios": ["a billing question", "a plan-change request", "a coverage complaint"],
        "facts": {
            "plan_name": ["the Unlimited Plus plan", "the 10GB Saver plan", "the Family Share plan"],
            "ticket_id": ["TKT-60412", "TKT-19937", "TKT-83350"],
            "monthly_fee": ["$65 a month", "$40 a month", "$120 a month"],
        },
    },
    "hotel": {
        "agent_role": "a hotel front-desk agent",
        "scenarios": ["a reservation change", "a late-checkout request", "a billing dispute"],
        "facts": {
            "confirmation": ["confirmation 5582-AQ", "confirmation 7140-BX", "confirmation 3098-CR"],
            "room_type": ["a King Deluxe room", "a Double Queen room", "a Junior Suite"],
            "nightly_rate": ["$189 a night", "$142 a night", "$310 a night"],
        },
    },
    "ecommerce": {
        "agent_role": "an online-retail support agent",
        "scenarios": ["an order-status inquiry", "a return request", "a damaged-item report"],
        "facts": {
            "order_id": ["order 100-447821", "order 100-993017", "order 100-330204"],
            "product": ["the wireless earbuds", "the standing desk", "the espresso machine"],
            "policy_window": ["a 30-day return window", "a 14-day return window", "a 90-day return window"],
        },
    },
    "utility": {
        "agent_role": "a utility-company support agent",
        "scenarios": ["a high-bill complaint", "a service-transfer request", "an outage question"],
        "facts": {
            "account_no": ["account 7782-1190", "account 4451-2208", "account 9930-7741"],
            "service": ["electric service", "natural-gas service", "water service"],
            "balance": ["a balance of $213", "a balance of $88", "a balance of $540"],
        },
    },
    "tech_support": {
        "agent_role": "a software tech-support agent",
        "scenarios": ["a login-failure ticket", "a billing-tier question", "a feature request"],
        "facts": {
            "product": ["the Acme Cloud Pro suite", "the Acme Drive app", "the Acme Mail service"],
            "case_id": ["case AX-55021", "case AX-77310", "case AX-10984"],
            "tier": ["the Pro tier", "the Team tier", "the Enterprise tier"],
        },
    },
}

# Probe categories -> (customer question template, expected-handling description).  ``{noun}`` is
# filled from the scenario's grounding fact so proper-noun recall is checkable.
PROBE_CATEGORIES: dict[str, dict] = {
    "proper_noun_recall": {
        "ask": "asks the agent to confirm a specific detail from the account on file",
        "expected": "The agent must correctly recall and state the exact grounded detail ({noun}).",
    },
    "context_adherence": {
        "ask": "asks a question that can only be answered from the provided account context",
        "expected": "The agent must answer strictly from the context ({noun}) and not invent unrelated facts.",
    },
    "unfulfillable_request": {
        "ask": "demands something outside policy that the agent cannot do",
        "expected": "The agent must politely decline, briefly explain why, and offer a legitimate alternative.",
    },
    "rudeness_management": {
        "ask": "is angry and rude while making an otherwise reasonable request",
        "expected": "The agent must stay calm, professional, and de-escalating while still helping.",
    },
}

_FIRST_NAMES = ["Brody", "Maya", "Devon", "Priya", "Liam", "Nadia", "Marcus", "Elena", "Theo", "Yuki"]
_LAST_NAMES = ["Murphy", "Okafor", "Reyes", "Sato", "Kowalski", "Hassan", "Nguyen", "Bauer", "Costa", "Ahmed"]


def _h(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def _agent_name(uid: str) -> str:
    h = _h(uid)
    return f"{_FIRST_NAMES[h % len(_FIRST_NAMES)]} {_LAST_NAMES[(h // 7) % len(_LAST_NAMES)]}"


def make_service_scenarios(n: int | None = None) -> list[dict]:
    """Deterministically enumerate grounded service scenarios (LLM-free).

    Expands domain x scenario-kind x probe-category, picks one value per fact slot by hash, and
    builds the agent name + role context + probe question + expected handling.  With ``n`` set,
    returns the first ``n`` (sorted by uid for reproducibility); else returns all.
    """
    rows: list[dict] = []
    for domain, spec in SERVICE_DOMAINS.items():
        for scenario in spec["scenarios"]:
            for probe, pc in PROBE_CATEGORIES.items():
                uid = f"{domain}|{scenario}|{probe}"
                h = _h(uid)
                # Pick one grounded value per fact slot; the first slot is the "headline" noun.
                facts = {k: vals[(h // (i + 1)) % len(vals)] for i, (k, vals) in enumerate(spec["facts"].items())}
                headline = next(iter(facts.values()))
                agent = _agent_name(uid)
                context_bits = [f"{k.replace('_', ' ')}: {v}" for k, v in facts.items()]
                context = (
                    f"You are {agent}, {spec['agent_role']}. The customer is calling about {scenario}. "
                    f"Account details on file -- {'; '.join(context_bits)}."
                )
                rows.append(
                    {
                        "uid": uid,
                        "domain": domain,
                        "agent_role": spec["agent_role"],
                        "agent_name": agent,
                        "context": context,
                        "probe_category": probe,
                        "probe_question": pc["ask"],
                        "expected": pc["expected"].format(noun=headline),
                    }
                )
    rows.sort(key=lambda r: r["uid"])
    return rows[:n] if n is not None else rows


# ---------------------------------------------------------------------------
# Service dialogue-instruction templates (analogue of DIALOGUE_INSTRUCTION_TEMPLATES in
# hf_to_dialogue.py).  Picked per-scenario by uid hash in the wiring step.  Each reuses the
# shared _COMMON_SUFFIX (TTS-friendly, no markdown, user speaks first, JSON array output).
# ---------------------------------------------------------------------------

SERVICE_DIALOGUE_TEMPLATES = [
    (
        "Write a short, natural spoken phone dialogue between a customer (user) and a service "
        "agent (assistant). The assistant stays fully in character as the named agent and answers "
        "ONLY from the provided account context. The customer opens, the agent responds helpfully "
        "and concisely. Keep to 3-5 turns.\n" + _COMMON_SUFFIX
    ),
    (
        "Write a spoken customer-service exchange where the customer (user) is brief and the agent "
        "(assistant) handles the request precisely, grounded in the account context. No corporate "
        "filler; sound like a real competent agent. 3-5 turns.\n" + _COMMON_SUFFIX
    ),
    (
        "Write a spoken service dialogue with a slightly impatient customer (user); the agent "
        "(assistant) stays calm, professional, and grounded in the context, and resolves or "
        "redirects the request. 4-6 turns.\n" + _COMMON_SUFFIX
    ),
]

SERVICE_DIALOGUE_TEMPLATE_NAMES = ["service_direct", "service_precise", "service_deescalate"]
assert len(SERVICE_DIALOGUE_TEMPLATE_NAMES) == len(SERVICE_DIALOGUE_TEMPLATES)


def pick_service_template(uid: str) -> tuple[str, str]:
    idx = _h(uid) % len(SERVICE_DIALOGUE_TEMPLATES)
    return SERVICE_DIALOGUE_TEMPLATES[idx], SERVICE_DIALOGUE_TEMPLATE_NAMES[idx]


def build_service_user_message(spec: dict, template: str) -> str:
    """Service-specific user message for the dialogue generator (analogue of _build_user_message).

    ``spec`` is the normalized dict from the ``service`` adapter below.
    """
    parts = [
        f"Role context: {spec['background']}",
        f"In this call the customer {spec['question']}.",
        f"Required agent behavior: {spec['answer']}",
        "",
        template,
    ]
    return "\n".join(parts)


@register_adapter("service")
def adapt_service(example: dict) -> dict:
    """Map a service scenario-seed row to the normalized dialogue spec.

    Reuses the generic spec shape (uid/question/answer/background) so the existing HfToDialogue
    machinery works; ``question``/``answer`` carry the probe + expected handling, ``background``
    the grounded role context.  The service-specific prompt framing is applied via
    ``build_service_user_message`` + ``pick_service_template`` in the wiring step (Workstream A4).
    """
    return {
        "uid": str(example["uid"]),
        "question": example["probe_question"],
        "answer": example["expected"],
        "aliases": [],
        "options": None,
        "background": example["context"],
        "probe_category": example.get("probe_category", ""),
    }


class MakeServiceScenarios(Job):
    """Materialize the service-role scenario seeds as an HF dataset for ``HfToDialogue``.

    Tiny, deterministic, login-node job (mirrors ``SubsampleDataset``): writes the rows from
    :func:`make_service_scenarios` so the existing dialogue-gen machinery (sharding, multi-LLM
    round-robin, cleaning) consumes them via ``adapter_name="service"``.  Bump ``version`` to
    force a re-run if the taxonomy changes.
    """

    def __init__(self, *, n: int | None = None, version: int = 1):
        self.n = n
        self.version = version
        self.out_hf = self.output_path("service_scenarios", directory=True)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        from datasets import Dataset

        rows = make_service_scenarios(self.n)
        ds = Dataset.from_list(rows)
        ds.save_to_disk(str(self.out_hf.get()))
        print(f"[MakeServiceScenarios] wrote {len(ds)} service scenarios to {self.out_hf.get()}")


def demo() -> None:
    rows = make_service_scenarios()
    print(
        f"[personaplex-service-spike] {len(rows)} grounded scenarios across {len(SERVICE_DOMAINS)} domains, "
        f"{len(PROBE_CATEGORIES)} probe categories\n"
    )
    for r in rows[:4]:
        spec = adapt_service(r)
        template, tname = pick_service_template(r["uid"])
        print(f"--- {r['uid']} (template={tname}) ---")
        print(build_service_user_message(spec, template))
        print()
    # Sanity guards (turn the construction into checks):
    uids = [r["uid"] for r in rows]
    assert len(uids) == len(set(uids)), "scenario uids must be unique"
    assert len(rows) >= 50, f"need >=50 scenarios, got {len(rows)}"
    for r in rows:
        assert r["context"] and r["probe_question"] and r["expected"], f"empty field in {r['uid']}"
        spec = adapt_service(r)
        for k in ("uid", "question", "answer", "background"):
            assert spec.get(k), f"adapter missing {k} for {r['uid']}"
    print(f"[personaplex-service-spike] OK: {len(rows)} unique grounded scenarios, all fields present")


if __name__ == "__main__":
    demo()
