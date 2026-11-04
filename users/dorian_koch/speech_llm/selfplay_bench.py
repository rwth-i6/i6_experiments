"""Self-play judge v2 (``spj2``): an LLM-graded self-play BENCHMARK on prompts and voices no model trained on.

Goal (2026-09-28): LLM grading of self-play dialogues as a benchmark, on prompts never seen in training, rating prompt
adherence, coherence and how well the two sides respond to each other; prompts written to be easy to grade; several
generations per prompt, because self-play is highly random.

Why a new set. The coherence judge v1 (``selfplay_judge``) rates self-play of 64 JRE windows whose prompts describe
windows p2-p5 trained on and whose voice clips are the TRAINING clips (one 4 s clip per episode speaker, reused for
every window of that speaker); it scores one conversation per window, and only coherence. Here:

  * PROMPTS: ``SPJ2_SCENARIOS`` -- hand-written scenario pairs, versioned and content-hashed (``scenario_digest``),
    written 2026-09-28 for this benchmark, never used by any training corpus. Each side gets a NAME, a ROLE, a GOAL /
    topic and (category ``instruction``) explicit checkable instructions, so "prompt adherence" can be checked
    against the text. Four categories x 12: ``topic`` (a concrete shared subject), ``task`` (goal-oriented service /
    role-play), ``tension`` (conflicting goals: negotiation, disagreement), ``instruction`` (explicit behaviours: say
    X, ask Y, never reveal Z, answer briefly). Neither side sees the other's prompt. The paper's Minimal line is
    prepended to each (as in training and ``CustomSelfPlayPrompts``).
  * VOICES: ``EvalVoiceCodes`` -- one 4 s clip per recording of ESB's GigaSpeech TEST split (pinned revision),
    encoded exactly like the training clips; no finetune of ours trained on GigaSpeech (JRE + one TTS voice), and
    no two clips share a recording. Each scenario gets a fixed seeded voice pair.
  * K GENERATIONS per scenario (``SelfPlayScenarioSet(replicates=K)``, default 4): the set holds each scenario K
    times as separate rows (``spj2:<category>:<name>#r<k>``, replicate-major order); ``PersonaSelfPlay`` samples each
    row independently (seeded per batch: ``conversation_seed(seed, first id of the batch)``), so the K rows differ
    only by sampling. Scores are reported with a NESTED bootstrap (prompts resampled, then replicates within each
    prompt) and the seed (within-prompt) SD next to the between-prompt SD.
  * JUDGE: ``SelfPlayJudgeV2`` -- GPT-OSS-120B (vLLM), both prompts + the two-sided ASR transcript (the v1
    builder), four anchored 1-5 scores in ONE guided-JSON reply: adherence A, adherence B, coherence, mutual
    responsiveness, each with a <= 30-word reason (schema ``maxLength``). The rubric text is content-hashed
    (``judge_rubric_digest``). ``overall`` = mean of (mean adherence, coherence, responsiveness) per conversation.
  * TABLE: ``SelfPlayJudgeV2Table`` -- models side by side, paired deltas vs a reference over the SAME scenarios
    (nested bootstrap). Registration: ``track_persona_spjudge`` (doriank).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess

from sisyphus import Job, Task, tk

from .common import json_schema_format, map_concurrent, prepend_pythonpath, vllm_gpu_mem_gb, vllm_server
from .label_free_hash import hash_bytes, label_free_hash, values_canon  # noqa: F401
from .podcast_ingest import _moshi_pythonpath
from .selfplay_judge import EMPTY_TRANSCRIPT, two_sided_transcript  # noqa: F401  -- the v1 transcript builder

#: Scenario sets, versioned. The CONTENT is hashed into every job (``scenario_digest``): editing one re-runs.
#: ``(category, name, prompt A, prompt B)``; the Minimal line is prepended by ``SelfPlayScenarioSet``.
SPJ2_SCENARIOS = {
    "spj2_v1": [
        # -- topic: a concrete shared subject, complementary roles --
        (
            "topic",
            "beekeeping",
            "Your name is Ruth. You have kept bees for twenty years. Tell the other person how you got started and what can go wrong with a hive in winter.",
            "Your name is Omar. You are thinking about keeping bees in your garden. Ask about the cost, the time it takes, and whether stings are a problem.",
        ),
        (
            "topic",
            "first_marathon",
            "Your name is Dana. You ran your first marathon last weekend. Describe the race, where it got hard, and how your legs feel now.",
            "Your name is Luis. You have never run more than five kilometers. Ask Dana how she trained and whether you could do it too.",
        ),
        (
            "topic",
            "sourdough",
            "Your name is Pierre. You are a baker. Explain how to keep a sourdough starter alive and why bread comes out flat.",
            "Your name is Kim. Your homemade sourdough bread keeps coming out flat and dense, and you want to know what you are doing wrong.",
        ),
        (
            "topic",
            "school_chess",
            "Your name is Marta. You coach a school chess club. Talk about what the children learn from chess and your club's first tournament.",
            "Your name is Ben. You are a parent wondering whether your shy eight-year-old son should join the chess club.",
        ),
        (
            "topic",
            "old_pickup",
            "Your name is Walt. You are restoring a 1968 pickup truck in your garage. Talk about the rust, finding parts, and the engine.",
            "Your name is Jess. You are a friend who knows nothing about cars but is curious why anyone would restore an old truck.",
        ),
        (
            "topic",
            "back_from_osaka",
            "Your name is Hana. You just came back from two years teaching English in Osaka. Share what surprised you about daily life in Japan.",
            "Your name is Tyler. You are thinking of teaching abroad. Ask about the food, learning the language, and whether Hana felt lonely.",
        ),
        (
            "topic",
            "rare_owl",
            "Your name is Gloria. You are a birdwatcher, and this morning you saw a rare snowy owl near the lake. You are very excited.",
            "Your name is Frank. You are Gloria's neighbor, you know nothing about birds, and you ask how she can be sure it was a snowy owl.",
        ),
        (
            "topic",
            "first_telescope",
            "Your name is Ahmed. You are an amateur astronomer. Help the other person choose a first telescope and explain what they will be able to see.",
            "Your name is Claire. Your ten-year-old daughter loves space and you want to buy her a telescope for under three hundred dollars.",
        ),
        (
            "topic",
            "community_garden",
            "Your name is Rosa. You run the neighborhood community garden. Explain how the plots work, the waiting list, and the watering schedule.",
            "Your name is Dev. You just moved into the neighborhood and you would like a plot to grow tomatoes and herbs.",
        ),
        (
            "topic",
            "jazz_vs_classical",
            "Your name is Miles. You are a jazz pianist. Explain how improvisation works and how you practice it.",
            "Your name is Ingrid. You are a classical violinist who always plays from sheet music and you find improvising scary.",
        ),
        (
            "topic",
            "volunteer_firefighter",
            "Your name is Carl. You have been a volunteer firefighter in a small town for ten years. Talk about the training and your most memorable call.",
            "Your name is Nadia. You are thinking about volunteering at the fire station and want to know how much time it takes.",
        ),
        (
            "topic",
            "board_game_design",
            "Your name is Theo. You are designing a board game about building railroads, and you want feedback on the rules.",
            "Your name is Lucy. You just played a test round of Theo's railroad game and you found the scoring confusing.",
        ),
        # -- task: goal-oriented service / role-play --
        (
            "task",
            "flight_rebooking",
            "Your name is Priya. Your flight to Denver on Friday was cancelled. You must arrive before six p.m. on Saturday for your sister's wedding.",
            "Your name is Tom. You are an agent at Skyway Air. The only Saturday seats left are on a seven a.m. flight with a stop in Chicago.",
        ),
        (
            "task",
            "restaurant_booking",
            "Your name is Greg. Call Luigi's restaurant to book a table for eight people at seven thirty on Friday. One guest uses a wheelchair.",
            "Your name is Sofia. You are the host at Luigi's. For eight people on Friday only six o'clock or nine o'clock is free. The accessible entrance is at the back.",
        ),
        (
            "task",
            "wifi_support",
            "Your name is Mr. Patel. Since a software update yesterday your laptop will not connect to your home Wi-Fi, and you need it for work.",
            "Your name is Alex. You work in technical support. Walk the caller through restarting the router, forgetting the network, and checking airplane mode.",
        ),
        (
            "task",
            "clinic_appointment",
            "Your name is Jenny. You are the receptionist at a family clinic. Today only a four fifteen appointment is free. Ask for the patient's name and date of birth.",
            "Your name is Victor. You have had a sore throat and fever for five days and you want to see a doctor today.",
        ),
        (
            "task",
            "apartment_viewing",
            "Your name is Mrs. Lee. You rent out a two-bedroom apartment for eighteen hundred dollars a month. You do not allow dogs, but cats are fine.",
            "Your name is Jordan. You want to rent the apartment and you have a cat. Ask about pets, the deposit, and parking.",
        ),
        (
            "task",
            "lost_suitcase",
            "Your name is Olga. Your blue suitcase did not arrive on flight 207 from Madrid, and it contains your work clothes for a meeting tomorrow.",
            "Your name is Sam. You work at the baggage service desk. Get a description of the suitcase, the flight number, and a delivery address.",
        ),
        (
            "task",
            "bakery_interview",
            "Your name is Helen. You are hiring a shift manager for a bakery. Ask about the candidate's experience and whether they can work weekends.",
            "Your name is Chris. You have three years of experience as a barista, you want this job, and you cannot work on Sundays.",
        ),
        (
            "task",
            "noisy_hotel_room",
            "Your name is Paula. You are a hotel guest. Your room is next to the elevator and too noisy to sleep, and you want this fixed tonight.",
            "Your name is Ivan. You are the night receptionist. You can offer a room change to the fifth floor or a twenty percent discount.",
        ),
        (
            "task",
            "stolen_card",
            "Your name is Martin. Your wallet was stolen an hour ago with your bank card in it. You want the card blocked and a new one sent.",
            "Your name is Aisha. You work at the bank's help line. Before you block a card you must verify the caller's date of birth and last purchase.",
        ),
        (
            "task",
            "antibiotics",
            "Your name is Dr. Wong. You are a pharmacist. Explain how to take the antibiotic: twice a day with food for seven days.",
            "Your name is Leo. You just picked up an antibiotic. You want to know whether you can drink alcohol and what to do if you miss a dose.",
        ),
        (
            "task",
            "wedding_catering",
            "Your name is Bianca. You run a catering company. Help the couple choose a menu for their wedding and give a price per guest.",
            "Your name is Noah. You are planning your wedding for one hundred and twenty guests, and about a third of them are vegetarian.",
        ),
        (
            "task",
            "first_driving_lesson",
            "Your name is Ray. You are a driving instructor booking a first lesson. Explain what happens in the first lesson and what to bring.",
            "Your name is Emma. You are thirty years old, you have never driven a car, and you are nervous about your first lesson.",
        ),
        # -- tension: conflicting goals --
        (
            "tension",
            "used_car_price",
            "Your name is Rick. You are selling your 2015 Honda Civic for nine thousand dollars. You will not go below eight thousand.",
            "Your name is Tanya. You want to buy Rick's Honda Civic, but your budget is seven thousand five hundred dollars and you noticed a dent in the door.",
        ),
        (
            "tension",
            "roommate_dishes",
            "Your name is Kate. You are annoyed because your roommate leaves dirty dishes in the sink for days. You want a clear rule.",
            "Your name is Max. You work night shifts, you think your roommate is too picky, and you wash your dishes when you have time.",
        ),
        (
            "tension",
            "four_day_week",
            "Your name is Julia. You believe every company should switch to a four-day work week, and you want to convince the other person.",
            "Your name is Bob. You own a small hardware store with six employees, and you think a four-day week would ruin your business.",
        ),
        (
            "tension",
            "neighbor_tree",
            "Your name is Hank. Branches from your oak tree hang over your neighbor's yard. You agree to trim them only if you split the cost.",
            "Your name is Maria. Leaves from your neighbor's oak tree block your gutters, and you think he should pay for the trimming himself.",
        ),
        (
            "tension",
            "raise_request",
            "Your name is Tina. You have worked at the company for three years and you are asking your manager for a ten percent raise.",
            "Your name is Paul. You are Tina's manager. You value her work, but your budget allows at most a three percent raise this year.",
        ),
        (
            "tension",
            "referee_call",
            "Your name is Coach Diaz. You are angry because the referee gave your team a penalty in the last minute of the soccer game.",
            "Your name is Anna. You were the referee. You are sure the penalty was correct, and you stay calm and explain the rule.",
        ),
        (
            "tension",
            "electric_cars",
            "Your name is Zoe. You are convinced electric cars are the future, and you want your uncle to buy one.",
            "Your name is Earl. You are a mechanic in a rural town and you are skeptical about electric cars because of range, charging, and cost.",
        ),
        (
            "tension",
            "school_night_concert",
            "Your name is Mia. You are sixteen and you want to go to a concert on Wednesday night. Your favorite band only plays in town once.",
            "Your name is Mr. Harris. You are Mia's father. She has a math exam on Thursday, and homework comes first.",
        ),
        (
            "tension",
            "bike_lanes",
            "Your name is Sven. You cycle to work and you want the city to build a protected bike lane on Main Street.",
            "Your name is Rita. You own a flower shop on Main Street, and you are worried the bike lane will remove the parking your customers need.",
        ),
        (
            "tension",
            "vacation_choice",
            "Your name is Owen. You want to spend your one week of vacation at the beach, doing nothing. You have to agree on a plan today.",
            "Your name is Lily. You want to spend the same vacation week visiting museums in Rome. You have to agree on a plan today.",
        ),
        (
            "tension",
            "getting_a_dog",
            "Your name is Jake. You really want to adopt a dog from the shelter this weekend, and you promise to walk it every day.",
            "Your name is Erin. You live with Jake. You are mildly allergic to dogs and you worry that Jake travels too much for work.",
        ),
        (
            "tension",
            "broken_heater",
            "Your name is Yusuf. You are a tenant. Your heater has been broken for a week in winter, and you want it repaired tomorrow.",
            "Your name is Mr. Burke. You are the landlord. The repair company cannot come until next Tuesday, and you offer a space heater.",
        ),
        # -- instruction: explicit, checkable behaviours --
        (
            "instruction",
            "peanut_party",
            "Your name is Nora. You meet the other person at a party. You must mention that you are allergic to peanuts, and ask at least two questions about their job.",
            "Your name is Felix. You are a chef who just opened a Thai restaurant. Keep inviting the other person to come and eat there.",
        ),
        (
            "instruction",
            "surprise_party",
            "Your name is Gina. You are secretly planning a surprise party on Saturday for the other person. Never reveal the party; if asked about Saturday, make up an excuse.",
            "Your name is Adam. You want to go hiking with Gina on Saturday. Keep asking what she is doing this weekend.",
        ),
        (
            "instruction",
            "busy_lawyer",
            "Your name is Ms. Grant. You are a very busy lawyer. Answer only in short sentences of a few words, and try to end the call politely.",
            "Your name is Kevin. You are a journalist who wants a long quote about a new housing law. Keep asking follow-up questions.",
        ),
        (
            "instruction",
            "museum_photo",
            "Start by introducing yourself as Doctor Lena Ortiz from the city museum. Your goal is to borrow the other person's old photo of the harbor for an exhibition.",
            "Your name is Mr. Frost. You own a photo of the harbor from 1920. You will lend it only if your grandfather's name is shown next to it.",
        ),
        (
            "instruction",
            "raffle_tickets",
            "Your name is Beth. You sell raffle tickets for a children's hospital at five dollars each. Try to get the other person to buy exactly three tickets.",
            "Your name is Dan. You want to support the charity, but you only have twelve dollars in cash with you.",
        ),
        (
            "instruction",
            "spell_the_name",
            "Your name is Carlos. You are a hotel receptionist taking a phone reservation. Ask for the guest's full name and ask them to spell the last name.",
            "Your name is Anneliese Wojcik. You want a double room for two nights starting on March third.",
        ),
        (
            "instruction",
            "one_book",
            "Your name is Fiona. You work in a bookstore. Recommend exactly one book to the customer and explain why it fits.",
            "Your name is George. You want a gift for your father, who loves sailing and history, and you do not want a book longer than three hundred pages.",
        ),
        (
            "instruction",
            "repeat_directions",
            "Your name is Pedro. You are a tourist looking for the train station. After you get directions, repeat them back to check you understood.",
            "Your name is Wendy. You live here. The station is two blocks north, then left at the bakery, next to the post office.",
        ),
        (
            "instruction",
            "soup_cookbook",
            "Your name is Stella. Whatever the other person talks about, steer the conversation toward your new cookbook about soups.",
            "Your name is Mike. You only want to talk about last night's basketball game and the last-second shot.",
        ),
        (
            "instruction",
            "forgot_birthday",
            "Your name is Liam. You forgot your best friend's birthday yesterday. Apologize sincerely and suggest a specific dinner plan to make up for it.",
            "Your name is Chloe. You are hurt that Liam forgot your birthday. You accept his apology only if he suggests a specific plan.",
        ),
        (
            "instruction",
            "capitals_quiz",
            "Your name is Quinn. You host a quiz. Ask the other person three questions about world capitals, one at a time, and say whether each answer is right.",
            "Your name is Harold. You are a quiz contestant. Try to answer every question, and ask for a hint exactly once.",
        ),
        (
            "instruction",
            "spanish_kitchen",
            "Your name is Lucia. Teach the other person three Spanish words for things in a kitchen. Say each word and its meaning, and ask them to repeat it.",
            "Your name is Dennis. You are a beginner learning Spanish. Repeat each word you are taught, and ask how to use one of them in a sentence.",
        ),
    ]
}
SPJ2_CATEGORIES = ("topic", "task", "tension", "instruction")
#: The level name the scenario rows carry (``PersonaSelfPlay(level=...)``).
SPJ2_LEVEL = "spj2"


def scenario_digest(name: str) -> str:
    """Content hash of a scenario set: hashed into every job using it, so an edited set is a different job."""
    return hashlib.sha256(json.dumps(SPJ2_SCENARIOS[name]).encode()).hexdigest()[:16]


def check_scenarios(scenarios: list, categories: tuple | None = SPJ2_CATEGORIES, max_words: int = 45) -> None:
    """Names unique, categories among ``categories`` (None: any), each side's prompt short enough for the self-play
    prompt budget (``max_words``: 45 for the spj2 formats, ``SP_IF_MAX_WORDS`` for the question lists)."""
    names = [s[1] for s in scenarios]
    assert len(set(names)) == len(names), "scenario names must be unique"
    if categories is not None:
        assert {s[0] for s in scenarios} <= set(categories), {s[0] for s in scenarios}
    for cat, name, pa, pb in scenarios:
        for p in (pa, pb):
            # the worker's KV context is conversation + 256 frames; voice 50 + 12 silence + the text tokens
            assert 8 <= len(p.split()) <= max_words, (name, len(p.split()))


def scenario_rows(scenarios: list, voices: list[dict], *, replicates: int, seed: int, minimal: str) -> list[dict]:
    """The self-play input rows: every scenario ``replicates`` times (replicate-major), each scenario with a fixed
    seeded voice PAIR (``voices``: ``{label, codes}``, two distinct clips per scenario, none reused). Row columns are
    what ``moshi_family.personaplex.selfplay`` reads plus ``scenario`` / ``category`` / ``replicate``."""
    import numpy as np

    check_scenarios(scenarios)
    if len(voices) < 2 * len(scenarios):
        raise RuntimeError(f"{len(voices)} voices for {len(scenarios)} scenarios (2 each, none reused)")
    order = np.random.default_rng(seed).permutation(len(voices))
    rows = []
    for k in range(int(replicates)):
        for i, (cat, name, pa, pb) in enumerate(scenarios):
            va, vb = voices[int(order[2 * i])], voices[int(order[2 * i + 1])]
            rows.append(
                {
                    "id": f"spj2:{cat}:{name}#r{k}",
                    "scenario": name,
                    "category": cat,
                    "replicate": k,
                    "other_ok": True,
                    "context_level": [SPJ2_LEVEL],
                    "context": [f"{minimal} {pa}"],
                    "context_other": [f"{minimal} {pb}"],
                    "voice_codes": np.asarray(va["codes"], np.int16),
                    "voice_codes_other": np.asarray(vb["codes"], np.int16),
                    "voice_label": str(va["label"]),
                    "voice_label_other": str(vb["label"]),
                }
            )
    return rows


class EvalVoiceCodes(Job):
    """``n`` voice-prompt clips from a pinned public TEST set no finetune of ours trained on (default ESB GigaSpeech
    test, one clip per recording), encoded exactly like the training clips. Worker: ``moshi_family.eval_voice_codes``
    (see its docstring for the seeded selection). CPU, ~3 min for 96 clips."""

    def __init__(
        self,
        *,
        venv_python_path: tk.Path,
        n: int,
        seed: int = 0,
        subset: str = "gigaspeech",
        repo: str = "hf-audio/esb-datasets-test-only-sorted",
        revision: str = "f04e38ac30d7f497f7cb58a29c4996e81f2c4e67",
        voice_sec: float = 4.0,
        min_active: float = 0.7,
    ):
        self.venv_python_path = venv_python_path
        self.n, self.seed = int(n), int(seed)
        self.subset, self.repo, self.revision = subset, repo, revision
        self.voice_sec, self.min_active = float(voice_sec), float(min_active)
        self.out_dir = self.output_path("dataset", directory=True)

    def tasks(self):
        yield Task("run", rqmt={"cpu": 4, "mem": 16, "time": 1})

    def run(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = prepend_pythonpath(_moshi_pythonpath(), env.get("PYTHONPATH"))
        cmd = [
            self.venv_python_path.get(),
            "-m",
            "moshi_family.eval_voice_codes",
            "--out",
            self.out_dir.get_path(),
            "--n",
            str(self.n),
            "--seed",
            str(self.seed),
            "--subset",
            self.subset,
            "--repo",
            self.repo,
            "--revision",
            self.revision,
            "--voice_sec",
            str(self.voice_sec),
            "--min_active",
            str(self.min_active),
        ]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, env=env, check=True)


class SelfPlayScenarioSet(Job):
    """The self-play INPUT dataset of a scenario set (``SPJ2_SCENARIOS[spec]``) with voices from ``voices``
    (``EvalVoiceCodes``): ``scenario_rows`` -- every scenario ``replicates`` times, level ``spj2``. Also writes
    ``scenarios.json`` (the prompts, categories and voice pair per scenario) for readers."""

    def __init__(self, *, voices: tk.Path, spec: str = "spj2_v1", replicates: int = 4, seed: int = 0):
        assert spec in SPJ2_SCENARIOS, spec
        check_scenarios(SPJ2_SCENARIOS[spec])
        self.voices = voices
        self.spec = spec
        self.scenario_digest = scenario_digest(spec)  # in the hash via hash() below
        self.replicates = int(replicates)
        self.seed = int(seed)
        self.out_dir = self.output_path("dataset", directory=True)
        self.out_scenarios = self.output_path("scenarios.json")

    @classmethod
    def hash(cls, parsed_args):
        """The scenario CONTENT is part of the hash (Sisyphus hashes only ``__init__`` arguments, so the
        ``scenario_digest`` attribute alone would not be): an edited scenario set is a different job."""
        return super(SelfPlayScenarioSet, cls).hash(
            {**parsed_args, "scenario_digest": scenario_digest(parsed_args["spec"])}
        )

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        from datasets import Dataset, load_from_disk

        from .persona_prompts import PERSONA_MINIMAL

        voices = list(load_from_disk(self.voices.get_path()).select_columns(["label", "codes"]))
        rows = scenario_rows(
            SPJ2_SCENARIOS[self.spec], voices, replicates=self.replicates, seed=self.seed, minimal=PERSONA_MINIMAL
        )
        Dataset.from_list(rows).save_to_disk(self.out_dir.get_path())
        meta = [
            {k: r[k] for k in ("scenario", "category", "context", "context_other", "voice_label", "voice_label_other")}
            for r in rows
            if r["replicate"] == 0
        ]
        json.dump(
            {"spec": self.spec, "digest": self.scenario_digest, "replicates": self.replicates, "scenarios": meta},
            open(self.out_scenarios.get_path(), "w"),
            indent=1,
        )
        print(f"[spj2] {len(meta)} scenarios x {self.replicates} = {len(rows)} rows ({self.spec})", flush=True)


# ---------------------------------------------------------------------------------------------------------------
# judge
# ---------------------------------------------------------------------------------------------------------------
SPJ2_METRICS = ("adherence_a", "adherence_b", "coherence", "responsiveness")
#: Reported per conversation besides the four: mean adherence and the headline ``overall``.
SPJ2_DERIVED = ("adherence", "overall")
#: Reason cap (characters; ~30 words) -- the schema's ``maxLength`` and the parser's truncation.
SPJ2_REASON_CHARS = 240

#: Versioned judge rubrics. The CONTENT is hashed into every judge job (``judge_rubric_digest``). Placeholders are
#: substituted with ``str.replace`` (``render_judge_message``), never ``str.format``, so braces in text are safe.
SPJ2_RUBRICS = {
    "spj2_judge_v1": """You are evaluating a spoken conversation between two AI voice agents, A and B. Each agent was given its own private instructions (below); neither saw the other's instructions. The transcript comes from automatic speech recognition of each agent's own audio channel: expect recognition errors, missing punctuation and fragments. Each line is one utterance, in time order, with its start time in seconds; the agents may overlap in time.

INSTRUCTIONS GIVEN TO A:
{prompt_a}

INSTRUCTIONS GIVEN TO B:
{prompt_b}

Duration: {duration} s. Words recognised: A {n_a}, B {n_b}.

TRANSCRIPT:
{transcript}

Rate four things, each on its own 1-5 scale. Ignore recognition errors, disfluencies, audio quality, politeness and factual accuracy. An agent that says (almost) nothing gets 1 for its own adherence.

adherence_a: how well A follows ITS instructions -- plays the stated role and name, pursues the stated goal or topic, and obeys every explicit instruction (things to mention, ask, avoid, or how to speak).
adherence_b: the same for B and B's instructions.
  5 = fully in role; pursues the goal or topic throughout; every explicit instruction followed.
  4 = clearly in role and on goal, with one minor lapse (one explicit instruction missed, or a brief drift).
  3 = partly: the role or topic is recognisable, but the goal or several instructions are ignored, or long stretches are unrelated.
  2 = barely: only a fleeting reference to the role or topic; otherwise unrelated or generic talk.
  1 = no sign of the instructions (silent, unrelated, or contradicting the role).

coherence: does the conversation make sense as a whole -- statements are intelligible, consistent with each other and with what was said before, and the content develops logically (no word salad, loops or repetition, self-contradiction, or abrupt unrelated jumps). Judge the content here, not whether the two respond to each other (that is the next item).
  5 = coherent throughout. 4 = coherent, with brief odd moments. 3 = noticeable stretches of repetition, contradiction or nonsense. 2 = mostly incoherent, with a few sensible stretches. 1 = incoherent throughout, or (almost) no speech.

responsiveness: do A and B respond TO EACH OTHER -- questions get answered, each picks up what the other just said, a change of topic by one is followed or acknowledged by the other; no talking past each other, no parallel monologues.
  5 = consistently responsive in both directions. 4 = mostly responsive; one or two missed questions or brief moments of talking past each other. 3 = some real exchange, but frequent ignored questions or parallel monologues. 2 = mostly talking past each other, with occasional responses. 1 = no exchange (one or both silent, or unrelated monologues).

Answer with JSON only. Each reason is one sentence of at most 30 words:
{"adherence_a": {"reason": "...", "score": 1-5}, "adherence_b": {"reason": "...", "score": 1-5}, "coherence": {"reason": "...", "score": 1-5}, "responsiveness": {"reason": "...", "score": 1-5}}""",
}

_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string", "minLength": 1, "maxLength": SPJ2_REASON_CHARS},
        "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
    },
    "required": ["reason", "score"],
    "additionalProperties": False,
}
SPJ2_JSON_SCHEMA = {
    "type": "object",
    "properties": {m: _ITEM_SCHEMA for m in SPJ2_METRICS},
    "required": list(SPJ2_METRICS),
    "additionalProperties": False,
}


#: Self-play GRPO REWARD rubrics (2026-10-02; ``RLFinetune(selfplay_reward={"rubric": name, ...})``): text + metric
#: names, chosen in the recipe. spj2's benchmark rubrics (``SPJ2_RUBRICS``) are valid reward rubrics too (metrics
#: ``SPJ2_METRICS``). ``sp_reward_v2`` = spj2_judge_v1 + per-side ``understanding`` (penalise a side that
#: clearly misunderstands information the other side gave), the
#: transcript's origin from ``{transcript_source}`` (v1 always says ASR and asks to ignore recognition errors, which
#: excuses a misheard name even in a monologue transcript) and a line description for the ``turns_v1`` format.
SP_REWARD_RUBRICS = {
    "sp_reward_v2": {
        "metrics": ("adherence_a", "adherence_b", "understanding_a", "understanding_b", "coherence", "responsiveness"),
        "text": """You are evaluating a spoken conversation between two AI voice agents, A and B. Each agent was given its own private instructions (below); neither saw the other's instructions. {transcript_source} Each line is a stretch of one agent's speech, in time order by start time (seconds). A line ends when the agent pauses, when the other agent starts speaking, or after the other agent has said a few words over it, so where the agents overlapped their lines alternate; a short line inside the other's speech is often a backchannel ("uh-huh", "yeah").

INSTRUCTIONS GIVEN TO A:
{prompt_a}

INSTRUCTIONS GIVEN TO B:
{prompt_b}

Duration: {duration} s. Words: A {n_a}, B {n_b}.

TRANSCRIPT:
{transcript}

Rate six things, each on its own 1-5 scale. Ignore disfluencies, audio quality, politeness and factual accuracy about the world. An agent that says (almost) nothing gets 1 for its own adherence and its own understanding.

adherence_a: how well A follows ITS instructions -- plays the stated role and name, pursues the stated goal or topic, and obeys every explicit instruction (things to mention, ask, avoid, or how to speak).
adherence_b: the same for B and B's instructions.
  5 = fully in role; pursues the goal or topic throughout; every explicit instruction followed.
  4 = clearly in role and on goal, with one minor lapse (one explicit instruction missed, or a brief drift).
  3 = partly: the role or topic is recognisable, but the goal or several instructions are ignored, or long stretches are unrelated.
  2 = barely: only a fleeting reference to the role or topic; otherwise unrelated or generic talk.
  1 = no sign of the instructions (silent, unrelated, or contradicting the role).

understanding_a: does A get WRONG what B said? Count only mistakes in what A takes up: a misheard or misremembered detail B stated (calling B by a wrong name, repeating a wrong number, date, place or item, confirming something B never said), or answering a question B did not ask. Each counts even when B corrects it afterwards. Not repeating or not using a detail is NOT a misunderstanding (ignoring the other belongs to responsiveness); if A gets nothing wrong, A gets 5.
understanding_b: the same for B: does B get wrong what A said?
  5 = no misunderstanding. 4 = one small slip that the agent corrects itself before being corrected. 3 = one clear misunderstanding (wrong name, number or item, or a question answered as if it were a different one). 2 = several misunderstandings, or one that derails the conversation. 1 = misunderstands or garbles most of what the other says, or (almost) no speech.

coherence: does the conversation make sense as a whole -- statements are intelligible, consistent with each other and with what was said before, and the content develops logically (no word salad, loops or repetition, self-contradiction, or abrupt unrelated jumps). Judge the content here, not whether the two respond to each other (that is the next item).
  5 = coherent throughout. 4 = coherent, with brief odd moments. 3 = noticeable stretches of repetition, contradiction or nonsense. 2 = mostly incoherent, with a few sensible stretches. 1 = incoherent throughout, or (almost) no speech.

responsiveness: do A and B respond TO EACH OTHER -- questions get answered, each picks up what the other just said, a change of topic by one is followed or acknowledged by the other; no talking past each other, no parallel monologues.
  5 = consistently responsive in both directions. 4 = mostly responsive; one or two missed questions or brief moments of talking past each other. 3 = some real exchange, but frequent ignored questions or parallel monologues. 2 = mostly talking past each other, with occasional responses. 1 = no exchange (one or both silent, or unrelated monologues).

Answer with JSON only. Each reason is one sentence of at most 30 words:
{"adherence_a": {"reason": "...", "score": 1-5}, "adherence_b": {"reason": "...", "score": 1-5}, "understanding_a": {"reason": "...", "score": 1-5}, "understanding_b": {"reason": "...", "score": 1-5}, "coherence": {"reason": "...", "score": 1-5}, "responsiveness": {"reason": "...", "score": 1-5}}""",
    },
}


#: ``sp_reward_v3_timing`` (2026-10-26: the grader is told about long silences, speaking rate,
#: whether the model was cut off, and the segment's total duration): ``sp_reward_v2`` plus a TIMING
#: block the reward fills per conversation (``moshi_family.rl.selfplay_reward.timing_facts``: planned length, early
#: stop, last speech, joint silences >= 3 s, words/s per side, cut off at the end, overlaps) and one paragraph on using
#: it. Why: the 1000-step multi-task run learned to say goodbye early and leave the rest of the slot silent, and the v2
#: judge, which sees only the transcript, scored those conversations higher. "About 4 words per second" is my choice,
#: near the 4.2 words/s of the ``sp_mono_v2_rate`` speech-rate penalty.
_V3_TIMING = "TIMING (measured from the recording, not said by the agents):\n{timing}"
_V3_USE = (
    "Also use TIMING. The agents are meant to keep the conversation going for its whole planned length. Wrapping up or "
    "falling silent well before the end, long silences, speech faster than about 4 words per second, and an agent cut "
    "off mid-speech are flaws: lower coherence for them, and lower responsiveness when they leave the other agent's "
    "question or turn unanswered."
)


def _v3_timing_text(v2: str) -> str:
    a, b = "TRANSCRIPT:\n{transcript}\n\n", "\n\nAnswer with JSON only."
    assert v2.count(a) == 1 and v2.count(b) == 1, "sp_reward_v2's layout changed"
    return v2.replace(a, a + _V3_TIMING + "\n\n").replace(b, "\n\n" + _V3_USE + b)


SP_REWARD_RUBRICS["sp_reward_v3_timing"] = {
    "metrics": SP_REWARD_RUBRICS["sp_reward_v2"]["metrics"],
    "text": _v3_timing_text(SP_REWARD_RUBRICS["sp_reward_v2"]["text"]),
}


def rubric_text_metrics(name: str) -> tuple[str, tuple]:
    """``(text, metrics)`` of a benchmark (``SPJ2_RUBRICS``) or reward (``SP_REWARD_RUBRICS``) rubric."""
    if name in SPJ2_RUBRICS:
        return SPJ2_RUBRICS[name], SPJ2_METRICS
    return SP_REWARD_RUBRICS[name]["text"], tuple(SP_REWARD_RUBRICS[name]["metrics"])


def rubric_json_schema(metrics) -> dict:
    """The guided-JSON schema for ``metrics`` (``SPJ2_JSON_SCHEMA`` for spj2's four)."""
    return {
        "type": "object",
        "properties": {m: _ITEM_SCHEMA for m in metrics},
        "required": list(metrics),
        "additionalProperties": False,
    }


def judge_rubric_digest(name: str) -> str:
    return hashlib.sha256(json.dumps(rubric_text_metrics(name)[0]).encode()).hexdigest()[:16]


def render_judge_message(
    rubric: str, *, prompt_a: str, prompt_b: str, duration: float, n_a: int, n_b: int, transcript: str
):
    vals = {
        "prompt_a": prompt_a,
        "prompt_b": prompt_b,
        "duration": f"{duration:.0f}",
        "n_a": str(n_a),
        "n_b": str(n_b),
        "transcript": transcript,
    }
    msg = SPJ2_RUBRICS[rubric]
    for k in ("prompt_a", "prompt_b", "duration", "n_a", "n_b"):  # transcript last: it may contain "{...}"
        msg = msg.replace("{" + k + "}", vals[k])
    return msg.replace("{transcript}", vals["transcript"])


def parse_judge_reply(raw: str) -> dict:
    """``{metric: {"score": int 1-5, "reason": str}}`` from a judge reply, or ValueError. Takes the outermost
    ``{...}``; every metric must be present with an INTEGER score in 1..5 (no bools, floats, strings) and a
    non-empty reason (whitespace collapsed, capped at ``SPJ2_REASON_CHARS``)."""
    raw = (raw or "").strip()
    i, j = raw.find("{"), raw.rfind("}")
    if i < 0 or j <= i:
        raise ValueError("no JSON object in the reply")
    d = json.loads(raw[i : j + 1])
    out = {}
    for m in SPJ2_METRICS:
        item = d.get(m) if isinstance(d, dict) else None
        if not isinstance(item, dict):
            raise ValueError(f"{m}: missing")
        s = item.get("score")
        if isinstance(s, bool) or not isinstance(s, int) or s not in (1, 2, 3, 4, 5):
            raise ValueError(f"{m}: score {s!r}")
        reason = " ".join(str(item.get("reason") or "").split())
        if not reason:
            raise ValueError(f"{m}: empty reason")
        out[m] = {"score": s, "reason": reason[:SPJ2_REASON_CHARS]}
    return out


def derived_scores(scores: dict) -> dict:
    """The four scores + ``adherence`` (mean of both sides) + ``overall`` (mean of adherence, coherence,
    responsiveness)."""
    adh = (scores["adherence_a"] + scores["adherence_b"]) / 2.0
    return {**scores, "adherence": adh, "overall": (adh + scores["coherence"] + scores["responsiveness"]) / 3.0}


def nested_bootstrap(by_prompt: dict, n_boot: int = 10000, seed: int = 0) -> tuple[float, float] | None:
    """95% interval of the grand mean under a two-stage bootstrap: resample PROMPTS with replacement, then each
    drawn prompt's replicates with replacement (``by_prompt``: ``{prompt: [scores]}``, >= 1 each)."""
    import numpy as np

    keys = sorted(by_prompt)
    if len(keys) < 2:
        return None
    rng = np.random.default_rng(seed)
    K = max(len(by_prompt[k]) for k in keys)
    mat = np.full((len(keys), K), np.nan)
    cnt = np.zeros(len(keys), int)
    for i, k in enumerate(keys):
        v = by_prompt[k]
        mat[i, : len(v)] = v
        cnt[i] = len(v)
    P = len(keys)
    means = np.empty(n_boot)
    for b0 in range(0, n_boot, 1000):
        nb = min(1000, n_boot - b0)
        ip = rng.integers(0, P, size=(nb, P))
        ik = (rng.random((nb, P, K)) * cnt[ip][:, :, None]).astype(int)
        vals = mat[ip[:, :, None], ik]
        means[b0 : b0 + nb] = vals.mean(axis=(1, 2))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_nested_bootstrap(a: dict, b: dict, n_boot: int = 10000, seed: int = 0):
    """Mean(a) - mean(b) over the prompts both have, with its 95% two-stage bootstrap interval (the same prompts
    drawn for both; replicates drawn independently per model)."""
    import numpy as np

    keys = sorted(set(a) & set(b))
    if len(keys) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    P = len(keys)

    def pack(d):
        K = max(len(d[k]) for k in keys)
        mat = np.full((P, K), np.nan)
        cnt = np.zeros(P, int)
        for i, k in enumerate(keys):
            mat[i, : len(d[k])] = d[k]
            cnt[i] = len(d[k])
        return mat, cnt, K

    (ma, ca, ka), (mb, cb, kb) = pack(a), pack(b)
    point = float(np.mean([np.mean(a[k]) for k in keys]) - np.mean([np.mean(b[k]) for k in keys]))
    deltas = np.empty(n_boot)
    for b0 in range(0, n_boot, 1000):
        nb = min(1000, n_boot - b0)
        ip = rng.integers(0, P, size=(nb, P))
        ia = (rng.random((nb, P, ka)) * ca[ip][:, :, None]).astype(int)
        ib = (rng.random((nb, P, kb)) * cb[ip][:, :, None]).astype(int)
        deltas[b0 : b0 + nb] = ma[ip[:, :, None], ia].mean(axis=(1, 2)) - mb[ip[:, :, None], ib].mean(axis=(1, 2))
    return point, (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)))


def score_stats(by_prompt: dict) -> dict:
    """Mean, nested-bootstrap CI, the between-prompt SD (of prompt means), the SEED SD (root mean within-prompt
    variance over the K generations) and the share of the variance that is between prompts."""
    import numpy as np

    keys = sorted(k for k in by_prompt if by_prompt[k])
    allv = [x for k in keys for x in by_prompt[k]]
    if not allv:
        return {"n": 0}
    pm = np.array([np.mean(by_prompt[k]) for k in keys])
    within = [np.var(by_prompt[k], ddof=1) for k in keys if len(by_prompt[k]) > 1]
    w = float(np.mean(within)) if within else float("nan")
    kbar = float(np.mean([len(by_prompt[k]) for k in keys]))
    between = float(np.var(pm, ddof=1)) if len(pm) > 1 else float("nan")
    var_prompt = max(between - w / kbar, 0.0) if within else float("nan")
    return {
        "n": len(allv),
        "n_prompts": len(keys),
        "mean": float(np.mean(allv)),
        "ci95": nested_bootstrap({k: by_prompt[k] for k in keys}),
        "prompt_sd": float(np.sqrt(between)) if len(pm) > 1 else None,
        "seed_sd": float(np.sqrt(w)) if within else None,
        "prompt_var_share": float(var_prompt / (var_prompt + w)) if within and (var_prompt + w) > 0 else None,
    }


#: spj2 judge versions (recorded as ``summary.json["judge_version"]``; a summary without the field is v1):
#:   1 -- ``extra_body={"guided_json": SPJ2_JSON_SCHEMA}``, which vLLM >= 0.19 IGNORES: the reply was unconstrained
#:        and only ``parse_judge_reply`` + 5 retries (new seed each) kept it valid. Every spj2 number up to 2026-10-12.
#:   2 -- (2026-10-12, with the vLLM fix) the schema is enforced
#:        (``common.json_schema_format``); same rubric, sampling, seeds and parser. ``judge_version`` is hashed only
#:        when != 1, so a v1 job keeps its hash (and its finished rows) and every default job is v2 -- a NEW work dir;
#:        the v1 dirs stay on disk and keep their rows (read them for any number reported under v1).
SPJ2_JUDGE_VERSION = 2


class SelfPlayJudgeV2(Job):
    """Rate self-play conversations of a scenario set (``SelfPlayScenarioSet`` rows run through ``PersonaSelfPlay``
    + ``SelfPlayTranscribe``) with ``SPJ2_RUBRICS[rubric]``: four 1-5 scores per conversation from one guided-JSON
    reply. ``runs``: ``{label: SelfPlayTranscribe dataset}`` (one label per model is the usual use: the judge is
    seeded per (prompts, transcript), so a conversation gets the same draw in any job).

    Outputs ``rows.jsonl`` (label, id, scenario, category, replicate, the four scores + reasons, word counts, the
    transcript) and ``summary.json`` (per label and metric -- the four, ``adherence``, ``overall`` --
    ``score_stats``: mean, nested-bootstrap 95% CI over scenarios then replicates, prompt SD, seed SD, prompt
    variance share; per category means; histograms; mute-conversation share). > 2% failed calls is fatal."""

    def __init__(
        self,
        *,
        runs: dict,
        rubric: str = "spj2_judge_v1",
        llm_name: str = "openai/gpt-oss-120b",
        temperature: float = 0.3,
        judge_version: int = SPJ2_JUDGE_VERSION,
    ):
        assert rubric in SPJ2_RUBRICS, rubric
        assert runs
        assert judge_version in (1, 2), judge_version
        self.runs = {str(k): v for k, v in sorted(runs.items())}
        self.judge_version = int(judge_version)
        self.rubric = rubric
        self.rubric_digest = judge_rubric_digest(rubric)  # in the hash via hash() below
        self.llm_name = llm_name
        self.temperature = float(temperature)
        self.out_rows = self.output_path("rows.jsonl")
        self.out_summary = self.output_path("summary.json")
        self.rqmt = {"gpu": 1, "cpu": 4, "mem": 32, "time": 2, "gpu_mem_gb": vllm_gpu_mem_gb(llm_name, 80)}

    __sis_hash_exclude__ = {"judge_version": 1}  # v1 jobs keep their pre-2026-10-12 hash

    @classmethod
    def hash(cls, parsed_args):
        """The rubric TEXT is part of the hash (see ``SelfPlayScenarioSet.hash``); the run labels are not, for a new
        job (``label_free_hash``)."""
        return label_free_hash(
            cls,
            parsed_args,
            canon=lambda a: {**a, "runs": values_canon(a["runs"])},
            base_hash=lambda a: super(SelfPlayJudgeV2, cls).hash(
                {**a, "rubric_digest": judge_rubric_digest(a["rubric"])}
            ),
            labels=lambda a: sorted(a["runs"]),
        )

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def completed_fraction(self):
        from .common import job_progress_fraction

        return job_progress_fraction(self)

    def items(self) -> list[dict]:
        from datasets import load_from_disk

        items = []
        for label, path in self.runs.items():
            cols = ["id", "n_frames", "frame_rate", "prompt_a", "prompt_b", "words_a", "words_b"]
            for i, r in enumerate(load_from_disk(path.get_path()).select_columns(cols)):
                rid = str(r["id"])
                if not rid.startswith("spj2:"):
                    raise RuntimeError(f"{label}: row {rid} is not a scenario-set row")
                _, cat, rest = rid.split(":", 2)
                scen, rep = rest.rsplit("#r", 1)
                dur = int(r["n_frames"]) / float(r["frame_rate"])
                wa, wb = json.loads(r["words_a"]), json.loads(r["words_b"])
                items.append(
                    {
                        "label": label,
                        "idx": i,
                        "id": rid,
                        "category": cat,
                        "scenario": scen,
                        "replicate": int(rep),
                        "dur": dur,
                        "prompt_a": r["prompt_a"],
                        "prompt_b": r["prompt_b"],
                        "n_a": sum(0 <= float(w["start"]) < dur for w in wa),
                        "n_b": sum(0 <= float(w["start"]) < dur for w in wb),
                        "transcript": two_sided_transcript(wa, wb, dur),
                    }
                )
        return items

    def run(self):
        from openai import OpenAI

        items = self.items()
        rubric, llm_name, temp = self.rubric, self.llm_name, self.temperature
        # v1 sent the ignored guided_json, i.e. nothing that constrains decoding
        constraint = json_schema_format(SPJ2_JSON_SCHEMA, "spj2") if self.judge_version >= 2 else {}
        print(
            f"[spj2-judge] {len(items)} conversations over {sorted({it['label'] for it in items})}, "
            f"judge v{self.judge_version} ({'schema enforced' if constraint else 'unconstrained'})",
            flush=True,
        )

        def judge(client, it):
            msg = render_judge_message(
                rubric,
                prompt_a=it["prompt_a"],
                prompt_b=it["prompt_b"],
                duration=it["dur"],
                n_a=it["n_a"],
                n_b=it["n_b"],
                transcript=it["transcript"],
            )
            key = it["prompt_a"] + "\n" + it["prompt_b"] + "\n" + it["transcript"]
            seed = int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**31)
            err = ""
            for attempt in range(5):
                try:
                    resp = client.chat.completions.create(
                        model=llm_name,
                        messages=[{"role": "user", "content": msg}],
                        seed=(seed + attempt) % (2**31),
                        temperature=temp,
                        max_tokens=6144,  # GPT-OSS reasons before it answers
                        **constraint,
                    )
                    return {"judged": parse_judge_reply(resp.choices[0].message.content), "error": ""}
                except Exception as e:  # API error, empty or unparsable reply: retry
                    err = f"{type(e).__name__}: {e}"[:200]
            return {"judged": None, "error": err}

        with vllm_server(llm_name) as url:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "nothing"), base_url=url)
            outs = map_concurrent(
                lambda it: judge(client, it), items, concurrency=192, progress_path=os.path.abspath("progress.json")
            )
        failed = [(it, o) for it, o in zip(items, outs) if o["judged"] is None]
        print(f"[spj2-judge] {len(failed)}/{len(items)} calls failed", flush=True)
        for it, o in failed[:5]:
            print(f"  FAILED {it['label']} {it['id']}: {o['error']}", flush=True)
        if len(failed) > 0.02 * len(items):
            raise RuntimeError(f"spj2 judging failed on {len(failed)}/{len(items)} calls (> 2%)")
        with open(self.out_rows.get_path(), "w") as f:
            for it, o in zip(items, outs):
                f.write(json.dumps({**it, **o}) + "\n")
        summary = summarize_rows([{**it, **o} for it, o in zip(items, outs)])
        summary.update(
            {
                "rubric": self.rubric,
                "rubric_digest": self.rubric_digest,
                "llm_name": llm_name,
                "judge_version": self.judge_version,
            }
        )
        json.dump(summary, open(self.out_summary.get_path(), "w"), indent=2)
        for label, s in summary["labels"].items():
            print(f"[spj2-judge] {label}: " + fmt_label(s), flush=True)


def by_prompt_scores(rows: list[dict], metric: str) -> dict:
    """``{scenario: [scores over replicates]}`` of one label's judged rows (failed rows skipped)."""
    out = {}
    for r in rows:
        if r.get("judged"):
            s = derived_scores({m: r["judged"][m]["score"] for m in SPJ2_METRICS})[metric]
            out.setdefault(r["scenario"], []).append(float(s))
    return out


def summarize_rows(rows: list[dict]) -> dict:
    import numpy as np

    summary = {"labels": {}}
    for label in sorted({r["label"] for r in rows}):
        lr = [r for r in rows if r["label"] == label]
        ok = [r for r in lr if r.get("judged")]
        s = {"n_conv": len(lr), "failed": len(lr) - len(ok), "metrics": {}, "by_category": {}, "hist": {}}
        s["mute_conv_frac"] = float(np.mean([r["n_a"] == 0 or r["n_b"] == 0 for r in lr])) if lr else None
        for m in SPJ2_METRICS + SPJ2_DERIVED:
            s["metrics"][m] = score_stats(by_prompt_scores(lr, m))
        for m in SPJ2_METRICS:
            v = [r["judged"][m]["score"] for r in ok]
            s["hist"][m] = {str(k): v.count(k) for k in (1, 2, 3, 4, 5)}
        for cat in SPJ2_CATEGORIES:
            cr = [r for r in lr if r["category"] == cat]
            s["by_category"][cat] = {
                m: score_stats(by_prompt_scores(cr, m)).get("mean") for m in SPJ2_METRICS + SPJ2_DERIVED
            }
        summary["labels"][label] = s
    return summary


def fmt_label(s: dict) -> str:
    parts = []
    for m in ("overall", "adherence", "coherence", "responsiveness"):
        x = s["metrics"][m]
        ci = x.get("ci95")
        parts.append(f"{m} {x.get('mean', float('nan')):.2f}" + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else ""))
    return "; ".join(parts) + f"; n={s['n_conv']}"


class SelfPlayJudgeV2Table(Job):
    """Models side by side from several ``SelfPlayJudgeV2`` row files (``judges``: ``{label: rows.jsonl}``; a file's
    rows of that label are used). Per model and metric: mean, nested-bootstrap CI, seed SD; and the PAIRED delta vs
    ``reference`` over the scenarios both have (nested bootstrap). Outputs ``table.json`` and ``table.md``."""

    def __init__(self, *, judges: dict, reference: str | None = None, order: list | None = None):
        assert reference is None or reference in judges, reference
        self.judges = {str(k): v for k, v in judges.items()}
        self.reference = reference
        self.order = list(order) if order else sorted(self.judges)
        assert set(self.order) == set(self.judges), "order must list every judged label once"
        self.out_json = self.output_path("table.json")
        self.out_md = self.output_path("table.md")

    @classmethod
    def hash(cls, parsed_args):
        """Labels (``judges`` keys, ``reference``, ``order``) are not hashed for a new job (``label_free_hash``): the reference is its rows file, the order only lays out the table."""
        return label_free_hash(
            cls,
            parsed_args,
            canon=lambda a: {
                "judges": values_canon(a["judges"]),
                "reference": None if a["reference"] is None else hash_bytes(a["judges"][a["reference"]]),
            },
            base_hash=lambda a: super(SelfPlayJudgeV2Table, cls).hash(a),
            labels=lambda a: (sorted(a["judges"]), a["reference"], a["order"]),
        )

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        rows, relabelled = {}, {}
        for label, path in self.judges.items():
            with open(path.get_path()) as f:
                all_rows = list(map(json.loads, f))
            rows[label] = [r for r in all_rows if r["label"] == label]
            if not rows[label]:
                # A label-free judge keeps the label of its first declaration, so after a rename its rows carry the
                # old one. A file with exactly one label is unambiguous: take it under the table's key.
                found = sorted({r["label"] for r in all_rows})
                if len(found) != 1 or found[0] in self.judges:
                    raise RuntimeError(f"{path}: no rows labelled {label} (labels: {found})")
                print(f"[spj2] {label}: rows labelled {found[0]!r} in {path}, taken under {label!r}", flush=True)
                relabelled[label] = found[0]
                rows[label] = [{**r, "label": label} for r in all_rows]
        table = {"reference": self.reference, "models": {}, "relabelled": relabelled}
        metrics = ("overall",) + tuple(m for m in SPJ2_DERIVED + SPJ2_METRICS if m != "overall")
        for label in self.order:
            s = summarize_rows(rows[label])["labels"][label]
            entry = {"n_conv": s["n_conv"], "failed": s["failed"], "mute_conv_frac": s["mute_conv_frac"], "metrics": {}}
            for m in metrics:
                e = dict(s["metrics"][m])
                if self.reference and label != self.reference:
                    d, ci = paired_nested_bootstrap(
                        by_prompt_scores(rows[label], m), by_prompt_scores(rows[self.reference], m)
                    )
                    e.update({"delta_vs_ref": d, "delta_ci95": ci})
                entry["metrics"][m] = e
            entry["by_category"] = s["by_category"]
            table["models"][label] = entry
        json.dump(table, open(self.out_json.get_path(), "w"), indent=2)
        lines = [
            "| model | n | overall | adherence | coherence | responsiveness | seed SD (overall) | mute |",
            "|---|---|---|---|---|---|---|---|",
        ]

        def cell(e):
            ci = e.get("ci95")
            t = f"{e['mean']:.2f}" + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "")
            if e.get("delta_vs_ref") is not None:
                dci = e["delta_ci95"]
                star = "*" if dci and (dci[0] > 0 or dci[1] < 0) else ""
                t += f" (Δ {e['delta_vs_ref']:+.2f}{star})"
            return t

        for label in self.order:
            e = table["models"][label]
            m = e["metrics"]
            lines.append(
                f"| {label} | {e['n_conv']} | {cell(m['overall'])} | {cell(m['adherence'])} | {cell(m['coherence'])} | "
                f"{cell(m['responsiveness'])} | {m['overall'].get('seed_sd') or float('nan'):.2f} | {e['mute_conv_frac']:.2f} |"
            )
        open(self.out_md.get_path(), "w").write("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)


# ---------------------------------------------------------------------------------------------------------------
# training scenarios for self-play GRPO
# ---------------------------------------------------------------------------------------------------------------
#: What each category means, shown to the generator LLM (the spj2 categories, module docstring).
SP_TRAIN_CATEGORY_TEXT = {
    "topic": "a concrete shared subject the two talk about; one usually knows more, the other is curious",
    "task": "a goal-oriented service or role-play exchange (ordering, booking, asking for help, an interview)",
    "tension": "conflicting goals: a negotiation, a disagreement, a complaint, persuading someone",
    "instruction": "each side gets explicit, checkable behaviours: things to say, ask, mention, avoid or never reveal, "
    "or how to speak (e.g. answer briefly)",
}
#: Levels of specificity (2026-10-01), each a word range per
#: side inside ``check_scenarios``' 8-45 bound. spj2's hand-written prompts sit between ``general`` and ``detailed``.
SP_TRAIN_LEVELS = {
    "minimal": (8, 16, "only a first name and a one-line situation or topic; no goal details, no explicit instructions"),
    "general": (15, 30, "a first name, a role and a goal or topic, in two short sentences"),
    "detailed": (28, 45, "a first name, a role, a goal, and two or three explicit, checkable instructions (things to "
                 "mention, ask, avoid or never reveal, or how to speak)"),
}  # fmt: skip
#: One example per category, written for this generator (NOT taken from ``SPJ2_SCENARIOS``).
SP_TRAIN_EXAMPLES = {
    "topic": ("sourdough", "Your name is Greta. You have baked sourdough bread every week for ten years. Explain how you keep "
              "your starter alive and what went wrong with your first loaves.",
              "Your name is Malik. You just got a sourdough starter from a friend and have no idea what to do with it. "
              "Ask how often to feed it and how long the dough should rise."),
    "task": ("bike_repair", "Your name is Ines. You run a bicycle repair shop. Find out what is wrong with the customer's bike "
             "and offer a repair date.",
             "Your name is Dev. Your bike's gears keep slipping and you need it fixed before Monday. Ask what the repair "
             "will cost."),
    "tension": ("noisy_neighbour", "Your name is Clara. Your neighbour plays drums late at night. You want them to stop "
                "after nine in the evening.",
                "Your name is Ben. You play drums and practise in the evening after work. You do not want to stop, but you "
                "could use a practice pad."),
    "instruction": ("museum_tour", "Your name is Otto. You are a museum guide. Mention the museum's oldest painting, ask the "
                    "visitor where they are from, and never say the ticket price.",
                    "Your name is Yara. You are visiting the museum. Ask how much the ticket costs, and answer every "
                    "question in one short sentence."),
}  # fmt: skip
#: Domains rotated through the generation calls (one per call, seeded) to spread the scenarios.
SP_TRAIN_DOMAINS = (
    "cooking", "travel", "the workplace", "health", "sports", "science", "history", "music", "family life", "money",
    "technology", "nature", "school", "housing", "shopping", "hobbies", "public services", "art", "cars", "pets",
    "gardening", "weather", "film and TV", "books", "games", "fitness", "neighbours", "restaurants", "holidays",
    "local events", "the outdoors", "crafts", "transport", "volunteering", "small businesses", "repairs",
)  # fmt: skip
SP_TRAIN_SYSTEM = """You write role-play scenarios for two voice agents, A and B, who will have a spoken conversation. Each agent gets ONLY its own instructions; neither sees the other's.

Rules for every scenario:
- Each side's instructions are addressed to that agent ("Your name is ... You ...") at the requested level of detail and length.
- The two sides fit together: they are in the same situation and have reasons to talk to each other.
- Everyday, safe, family-friendly content. No real people, brands or current events.
- Everything must be checkable from what the agents SAY in a one-minute conversation.
- Names of the two sides differ, and vary names across scenarios.
- "name" is a short snake_case label, unique within your answer.

Answer with JSON only: {"scenarios": [{"name": "...", "prompt_a": "...", "prompt_b": "..."}, ...]}"""
_SP_TRAIN_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "scenarios": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 3, "maxLength": 40},
                    "prompt_a": {"type": "string", "minLength": 20, "maxLength": 400},
                    "prompt_b": {"type": "string", "minLength": 20, "maxLength": 400},
                },
                "required": ["name", "prompt_a", "prompt_b"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["scenarios"],
    "additionalProperties": False,
}


_SENTENCE_COUNT = (
    r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|single)[- ](short |brief |concise |complete |full )?"
    r"sentences?\b|\bsyllables?\b|\bsentences? (?:of|no longer|up to|under|at most|with|shorter)\b"
)
#: A side that opens "You are/run/... X" and goes on with a bare order after a comma ("You are a guide, ask ...") or
#: ends on a bare verb (", ask."): note style the style rule forbids (about 1 in 10 minimal and general sides).
_FRAGMENT = (
    r"\bYou (?:are|run|own|have|work|coach|know|teach|manage|just \w+)\b[^.;,]*[,;] (?:ask|inquire|query|explain|"
    r"describe|mention|advise|request|need|must|give|tell|share|discuss)\b|, (?:ask|inquire|query|intrigued|curious)\."
)


def word_ngrams(text: str, n: int = 5) -> set:
    import re

    w = re.findall(r"[a-z0-9']+", text.lower())
    return {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}


def ngram_jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def filter_train_scenarios(
    cands: list[tuple],
    held_out: list[tuple],
    *,
    per_cell: int,
    max_jaccard: float = 0.2,
    levels: dict | None = None,
    max_words: int = 45,
    require_you: bool = False,
    reject_sentence_counts: bool = False,
    latin_only: bool = False,
):
    """Keep generated ``(category, name, prompt_a, prompt_b, level)`` in order, up to ``per_cell`` per (category,
    level): each side within the level's word range (``levels``, default ``SP_TRAIN_LEVELS``, inside
    ``check_scenarios``' 8-``max_words``), a name new
    to the set AND to ``held_out`` (spj2), and the pair's word-5-gram Jaccard with every held-out pair and every kept
    pair of its category <= ``max_jaccard``. ``require_you`` (v2 pools) also drops a scenario whose side does not go on as "You ..."/"Your ..." after its
    name line, as every spj2 prompt does (v2's first full run: first person "I"/"My" in 19% of minimal topic
    scenarios, note-style fragments in half the minimal instruction sides). ``reject_sentence_counts`` drops one
    that limits an answer in sentences (``_SENTENCE_COUNT``): speech has no full stops, the recogniser's punctuation
    decides the count the judge reads. ``latin_only`` drops one with letters outside the Latin script (a teach_check
    side quoting an Arabic phrase): the duplex models read and speak English. -> (kept, reject counts by reason)."""
    import re
    import unicodedata

    norm = lambda x: re.sub(r"[^a-z0-9]+", "_", str(x).lower()).strip("_")  # noqa: E731
    held_names = {norm(s[1]) for s in held_out}
    held_grams = [word_ngrams(s[2] + " " + s[3]) for s in held_out]
    kept, kept_grams, names, rej = [], {}, set(), {}
    count: dict = {}

    def reject(why):
        rej[why] = rej.get(why, 0) + 1

    for cat, name, pa, pb, level in cands:
        n = norm(name)
        lo, hi, _ = (levels or SP_TRAIN_LEVELS)[level]
        if count.get((cat, level), 0) >= per_cell:
            reject("cell_full")
        elif not all(max(8, lo - 3) <= len(p.split()) <= min(max_words, hi + 5) for p in (pa, pb)):  # a little slack
            reject("length")
        elif require_you and not all(re.match(r"\s*Your name is [^.]+\.\s+Your?\b", p) for p in (pa, pb)):
            reject("not_second_person")
        elif reject_sentence_counts and any(re.search(_SENTENCE_COUNT, p, re.I) for p in (pa, pb)):
            reject("sentence_count")
        elif require_you and any(re.search(_FRAGMENT, p) for p in (pa, pb)):
            reject("fragment")
        elif latin_only and any(ch.isalpha() and "LATIN" not in unicodedata.name(ch, "") for ch in pa + pb):
            reject("non_latin_script")
        elif not n or n in held_names:
            reject("name_in_spj2" if n in held_names else "no_name")
        elif n in names:
            reject("name_duplicate")
        else:
            g = word_ngrams(pa + " " + pb)
            if any(ngram_jaccard(g, h) > max_jaccard for h in held_grams):
                reject("near_spj2")
            elif any(ngram_jaccard(g, h) > max_jaccard for h in kept_grams.get(cat, [])):
                reject("near_duplicate")
            else:
                kept.append((cat, n, pa.strip(), pb.strip(), level))
                kept_grams.setdefault(cat, []).append(g)
                names.add(n)
                count[(cat, level)] = count.get((cat, level), 0) + 1
    return kept, rej


def _scenario_tuples(c: dict, d: dict) -> list:
    return [(c["category"], x["name"], x["prompt_a"], x["prompt_b"], c["level"]) for x in d["scenarios"]]


class _CallCache:
    """Finished LLM work of a job that may be interrupted: a JSONL (``{"i", "key", "out"}``, appended as each item
    finishes; ``path`` None = memory only). On load, an entry counts only if its key equals ``keys[i]`` (the request
    is unchanged); a line cut by the interruption is skipped and terminated, so the next append starts a new line."""

    def __init__(self, path: str | None, keys: list[str]):
        import threading

        self.path, self.keys, self.done, self._lock = path, list(keys), {}, threading.Lock()
        if path and os.path.exists(path):
            for line in open(path):
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                i = int(rec["i"])
                if i < len(self.keys) and rec.get("key") == self.keys[i]:
                    self.done[i] = rec["out"]
            with open(path, "rb+") as f:
                f.seek(0, 2)
                if f.tell() and (f.seek(-1, 2), f.read(1))[1] != b"\n":
                    f.write(b"\n")

    def add(self, i: int, out) -> None:
        with self._lock:
            if self.path:
                with open(self.path, "a") as f:
                    f.write(json.dumps({"i": i, "key": self.keys[i], "out": out}) + "\n")
            self.done[i] = out


def generate_scenario_batches(
    llm_name: str,
    temperature: float,
    calls: list[dict],
    tag: str = "sp-train-gen",
    *,
    system: str = SP_TRAIN_SYSTEM,
    schema: dict = _SP_TRAIN_JSON_SCHEMA,
    parse=_scenario_tuples,
    cache_path: str | None = None,
) -> list:
    """One guided-JSON request per call (``user``, ``seed`` and whatever ``parse`` reads) under ``system``, 5 attempts
    each -> per call ``parse(call, reply JSON)`` (default: ``[(category, name, prompt A, prompt B, level), ...]`` in
    answer order), or None if every attempt failed. Serves its own vLLM server for the duration.

    ``cache_path``: a JSONL of finished calls (``{"i", "out"}``, appended as each finishes). An interrupted job that
    runs again requests only the calls missing there, so a requeue does not repeat finished generation; an entry
    counts only if its request text is unchanged (``key``)."""
    from openai import OpenAI

    key = lambda c: hashlib.sha256(c["user"].encode()).hexdigest()[:16]  # noqa: E731
    cache = _CallCache(cache_path, [key(c) for c in calls])
    done = cache.done

    def gen(client, i, c):
        out, err = None, ""
        for attempt in range(5):  # 3 until 2026-11-04: a reply cut at the token cap failed ~4% of v2 calls
            try:
                resp = client.chat.completions.create(
                    model=llm_name,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": c["user"]}],
                    seed=(c["seed"] + attempt) % (2**31),
                    temperature=temperature,
                    max_tokens=8192,
                    # enforced from 2026-10-12 (run()-side; the finished v1 build was unconstrained)
                    **json_schema_format(schema, "sp_train_scenarios"),
                )
                raw = resp.choices[0].message.content or ""
                out = parse(c, json.loads(raw[raw.find("{") : raw.rfind("}") + 1]))
                break
            except Exception as e:  # retried; a final failure contributes nothing (counted)
                err = f"{type(e).__name__}: {e}"[:200]
        if out is None:
            print(f"[{tag}] call failed: {err}", flush=True)
        else:
            cache.add(i, out)
        return out

    todo = [i for i in range(len(calls)) if i not in done]
    if done:
        print(f"[{tag}] resuming: {len(done)} of {len(calls)} calls already done ({cache_path})", flush=True)
    if todo:
        with vllm_server(llm_name) as url:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "nothing"), base_url=url)
            new = map_concurrent(
                lambda i: gen(client, i, calls[i]), todo, concurrency=64, progress_path=os.path.abspath("progress.json")
            )
        done.update({i: o for i, o in zip(todo, new) if o is not None})
    # JSON round trip turns tuples into lists: hand back tuples either way
    return [
        None if done.get(i) is None else [tuple(x) if isinstance(x, list) else x for x in done[i]]
        for i in range(len(calls))
    ]


class SelfPlayScenarioGen(Job):
    """LLM-written TRAINING scenario pairs for self-play GRPO, in the spj2 format ``(category, name, prompt A,
    prompt B)``, kept disjoint from the spj2 benchmark (``filter_train_scenarios`` against ``held_out``).

    ``calls_per_cell`` calls per (category, level of specificity: ``SP_TRAIN_LEVELS``) of ``per_call`` scenarios
    each, every call with a seeded domain from ``SP_TRAIN_DOMAINS``, the category's definition, the level's
    description and word range, and the category's one example (``SP_TRAIN_EXAMPLES``, written for this job). Keeps up
    to ``per_cell`` per (category, level). Outputs ``scenarios.json`` (``[category, name, prompt A, prompt B, level]``,
    generation order) and ``stats.json`` (raw count, rejects by reason, kept per cell)."""

    def __init__(
        self,
        *,
        per_cell: int = 170,
        per_call: int = 10,
        calls_per_cell: int = 28,
        held_out: str = "spj2_v1",
        llm_name: str = "openai/gpt-oss-120b",
        temperature: float = 1.0,
        seed: int = 0,
    ):
        assert held_out in SPJ2_SCENARIOS, held_out
        self.per_cell, self.per_call, self.calls_per_cell = int(per_cell), int(per_call), int(calls_per_cell)
        self.held_out = held_out
        self.held_out_digest = scenario_digest(held_out)
        self.llm_name, self.temperature, self.seed = llm_name, float(temperature), int(seed)
        self.out_scenarios = self.output_path("scenarios.json")
        self.out_stats = self.output_path("stats.json")
        self.rqmt = {"gpu": 1, "cpu": 4, "mem": 32, "time": 2, "gpu_mem_gb": vllm_gpu_mem_gb(llm_name, 80)}

    @classmethod
    def hash(cls, parsed_args):
        """The held-out set's CONTENT and the generator prompt are part of the hash."""
        prompt = json.dumps(
            [SP_TRAIN_SYSTEM, SP_TRAIN_CATEGORY_TEXT, SP_TRAIN_EXAMPLES, SP_TRAIN_DOMAINS, SP_TRAIN_LEVELS]
        )
        return super(SelfPlayScenarioGen, cls).hash(
            {
                **parsed_args,
                "held_out_digest": scenario_digest(parsed_args["held_out"]),
                "prompt_digest": hashlib.sha256(prompt.encode()).hexdigest()[:16],
            }
        )

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def calls(self) -> list[dict]:
        import numpy as np

        rng = np.random.default_rng(self.seed)
        out = []
        for cat in SPJ2_CATEGORIES:
            for level, (lo, hi, desc) in SP_TRAIN_LEVELS.items():
                for _ in range(self.calls_per_cell):
                    dom = SP_TRAIN_DOMAINS[int(rng.integers(len(SP_TRAIN_DOMAINS)))]
                    ex = SP_TRAIN_EXAMPLES[cat]
                    example = json.dumps({"name": ex[0], "prompt_a": ex[1], "prompt_b": ex[2]})
                    user = (
                        f"Category: {cat} -- {SP_TRAIN_CATEGORY_TEXT[cat]}.\n"
                        f"Level of detail: {level} -- {desc}; {lo} to {hi} words per side.\n"
                        f"Domain for this batch: {dom}.\n\n"
                        f"Example of the category (its level of detail may differ; do not reuse its situation or "
                        f"names):\n{example}\n\n"
                        f"Write {self.per_call} new, varied scenarios of this category, at this level of detail, in "
                        f"this domain."
                    )
                    out.append(
                        {"category": cat, "level": level, "domain": dom, "user": user, "seed": int(rng.integers(2**31))}
                    )
        return out

    def run(self):
        calls = self.calls()
        print(f"[sp-train-gen] {len(calls)} calls x {self.per_call} scenarios ({self.llm_name})", flush=True)
        outs = generate_scenario_batches(self.llm_name, self.temperature, calls)
        failed = sum(o is None for o in outs)
        if failed > 0.05 * len(calls):
            raise RuntimeError(f"{failed}/{len(calls)} generation calls failed")
        cands = [s for o in outs if o for s in o]
        kept, rej = filter_train_scenarios(cands, SPJ2_SCENARIOS[self.held_out], per_cell=self.per_cell)
        check_scenarios([k[:4] for k in kept])
        per_cell = {
            f"{c}/{lv}": sum(k[0] == c and k[4] == lv for k in kept) for c in SPJ2_CATEGORIES for lv in SP_TRAIN_LEVELS
        }
        json.dump([list(k) for k in kept], open(self.out_scenarios.get_path(), "w"), indent=1)
        stats = {"calls": len(calls), "failed_calls": failed, "raw": len(cands), "kept": len(kept), "kept_per_cell": per_cell,
                 "rejected": rej}  # fmt: skip
        json.dump(stats, open(self.out_stats.get_path(), "w"), indent=1)
        print(f"[sp-train-gen] {stats}", flush=True)
        if min(per_cell.values()) < self.per_cell // 2:
            raise RuntimeError(f"too few scenarios kept per cell: {per_cell}")


class SelfPlayTrainScenarioSet(Job):
    """The self-play GRPO TRAINING input (``moshi_family.rl.ppx_selfplay.SelfPlayScenarioSource``): one row per
    ``SelfPlayScenarioGen`` scenario (ids ``sptr:<category>:<name>``), the Minimal line prepended as in spj2, and a
    seeded voice pair per scenario drawn from ``voices`` (two distinct clips; clips are reused across scenarios)."""

    def __init__(self, *, scenarios: tk.Path, voices: tk.Path, seed: int = 0):
        self.scenarios, self.voices, self.seed = scenarios, voices, int(seed)
        self.out_dir = self.output_path("dataset", directory=True)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import numpy as np
        from datasets import Dataset, load_from_disk

        from .persona_prompts import PERSONA_MINIMAL

        scen = [tuple(s) for s in json.load(open(self.scenarios.get_path()))]
        check_scenarios([s[:4] for s in scen])
        voices = list(load_from_disk(self.voices.get_path()).select_columns(["label", "codes"]))
        rng = np.random.default_rng(self.seed)
        rows = []
        for cat, name, pa, pb, level in scen:
            ia, ib = (int(i) for i in rng.choice(len(voices), size=2, replace=False))
            rows.append(
                {
                    "id": f"sptr:{cat}:{name}",
                    "scenario": name,
                    "category": cat,
                    "level": level,
                    "context": [f"{PERSONA_MINIMAL} {pa}"],
                    "context_other": [f"{PERSONA_MINIMAL} {pb}"],
                    "voice_codes": np.asarray(voices[ia]["codes"], np.int16),
                    "voice_codes_other": np.asarray(voices[ib]["codes"], np.int16),
                    "voice_label": str(voices[ia]["label"]),
                    "voice_label_other": str(voices[ib]["label"]),
                }
            )
        Dataset.from_list(rows).save_to_disk(self.out_dir.get_path())
        print(f"[sp-train-set] {len(rows)} scenarios, {len(voices)} voices", flush=True)


# ---------------------------------------------------------------------------------------------------------------
# training scenarios v2: category pools with assigned first names, mixed by weight
# ---------------------------------------------------------------------------------------------------------------
#: First names: Ninble/name-census-top-100, the top 100 male and 100 female first names of 139 countries (government
#: statistics cross-checked against public profiles; GPL-3.0), pinned by commit and content hash.
NAME_CENSUS_URL = (
    "https://raw.githubusercontent.com/Ninble/name-census-top-100/27944ac7d285291793f20859ff845aea7787e74d/"
    "first-name-database.csv"
)
NAME_CENSUS_SHA256 = "25fe1d3b8646eb50bff117a52e7ab29e103a9a31f52abcf484bcf04df759f0e9"
#: Added to the generator request when names are assigned (v1 let the LLM pick: 26% of 4,080 sides used one of
#: five names).
SP_POOL_NAMES_RULE = (
    "Use exactly these first names, one pair per scenario, in this order. Side A's instructions start with \"Your name "
    'is <A>." and side B\'s with "Your name is <B>."; use no other names for A or B.'
)

#: v2 pools: spj2's prompt style. Without it, minimal-level sides came out as "I ..." or as notes ("ask about time,
#: answer with word, hide budget").
SP_POOL_STYLE_RULE = (
    'After "Your name is ...", each side continues in full sentences addressed to the agent ("You ...", "Your ..."); '
    'never "I" or "My", never note-style fragments.'
)


def ascii_first_name(name: str) -> str | None:
    """``name`` folded to plain Latin letters (NFKD, accents dropped: Jose, Omer), or None if anything else is left
    (Cyrillic, Arabic, 'ł', digits, spaces) or it has no vowel (initials: Ch, Mj, Sk)."""
    import re
    import unicodedata

    s = unicodedata.normalize("NFKD", str(name).strip())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    ok = re.fullmatch(r"[A-Z](?:[a-z]+|(?='))(?:[-'][A-Z]?[a-z]+)?", s) and re.search(r"[aeiouyAEIOUY]", s)
    return s if ok else None


def name_pool_from_csv(text: str, min_frequency: int = 0) -> dict:
    """The census CSV (``;``-separated) -> ``{country code: sorted [[name, gender], ...]}``, one entry per name and
    country; gender ``m`` / ``f``, or ``u`` when the country lists the name for both. Rows with ``Frequency`` below
    ``min_frequency`` are skipped, and so are names equal to the name of the country listing them (Qatar in Qatar; Jordan in the US stays)."""
    import csv
    import io

    by: dict = {}
    for row in csv.DictReader(io.StringIO(text), delimiter=";"):
        n = ascii_first_name(row["Name"])
        own_country = n is not None and n.lower() == str(row.get("Country name") or "").strip().lower()
        if n is None or int(row.get("Frequency") or 0) < min_frequency or own_country:
            continue
        g = row["Gender"].strip().lower()
        d = by.setdefault(row["Country code"].strip(), {})
        d[n] = g if d.get(n, g) == g else "u"
    return {cc: sorted([n, g] for n, g in d.items()) for cc, d in sorted(by.items()) if d}


def draw_name_pairs(pool: dict, n: int, rng) -> list[tuple]:
    """``n`` pairs ``((name, gender, country), (name, gender, country))``: each side a uniform country, then a uniform
    name of that country, so every country weighs the same whatever its name count; no name twice in the ``n`` pairs."""
    ccs = sorted(pool)
    used: set = set()
    out = []
    while len(out) < n:
        pair = []
        while len(pair) < 2:
            cc = ccs[int(rng.integers(len(ccs)))]
            name, g = pool[cc][int(rng.integers(len(pool[cc])))]
            if name not in used:
                used.add(name)
                pair.append((name, g, cc))
        out.append(tuple(pair))
    return out


def follows_names(prompt_a: str, prompt_b: str, name_a: str, name_b: str) -> bool:
    """Each side's instructions open with its assigned name."""
    import re

    ok = lambda p, n: re.match(rf"\s*Your name is {re.escape(n)}\b", p) is not None  # noqa: E731
    return ok(prompt_a, name_a) and ok(prompt_b, name_b)


def prompt_name(prompt: str) -> str | None:
    """The name a side's instructions give it (``Your name is X``), if any."""
    import re

    m = re.search(r"\bYour name is ([A-Z][\w'-]*)", prompt)
    return m.group(1) if m else None


def assign_names(outs: list, calls: list[dict]) -> tuple[list, dict, int]:
    """Generated batches (``outs[i]`` answers ``calls[i]``; None = failed) -> (every scenario, ``{(prompt A, prompt
    B): (side A, side B)}``, how many used their assigned pair). Nothing is dropped: a scenario that uses its pair
    (``calls[i]["names"][k]`` for the k-th of the answer) records the pair's ``(name, gender, country)``; any other
    records the name its instructions give (``prompt_name``, gender and country unknown: ``""``), or ``None``."""
    cands, names, followed = [], {}, 0
    for out, c in zip(outs, calls):
        pairs = c.get("names") or []
        for k, s in enumerate(out or []):
            cands.append(s)
            if k < len(pairs) and follows_names(s[2], s[3], pairs[k][0][0], pairs[k][1][0]):
                followed += 1
                sides = tuple(tuple(x) for x in pairs[k])
            else:
                sides = tuple((n, "", "") if (n := prompt_name(p)) else None for p in (s[2], s[3]))
            names[(s[2].strip(), s[3].strip())] = sides
    return cands, names, followed


class FirstNamePool(Job):
    """First names for self-play scenarios (``NAME_CENSUS_URL``, checked against ``NAME_CENSUS_SHA256``). Kept: names
    that fold to plain Latin letters (``ascii_first_name``) and are counted at least ``min_frequency`` times, countries
    with at least ``min_per_country`` of them.
    Output ``names.json``: ``{"source", "sha256", "countries": {code: [[name, gender], ...]}}``. Runs on the login node
    (mini task: it downloads)."""

    def __init__(
        self,
        *,
        url: str = NAME_CENSUS_URL,
        sha256: str = NAME_CENSUS_SHA256,
        min_per_country: int = 10,
        min_frequency: int = 50,
    ):
        # min_frequency: the rare, mostly non-official entries are noise (Iceland, Haiti, Tanzania, Goddess, Green,
        # Atlanta, Muzungu all count < 35); at 50, 122 of 139 countries keep >= 10 names
        self.url, self.sha256, self.min_per_country = url, sha256, int(min_per_country)
        self.min_frequency = int(min_frequency)
        self.out_names = self.output_path("names.json")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import urllib.request

        raw = urllib.request.urlopen(self.url, timeout=120).read()
        got = hashlib.sha256(raw).hexdigest()
        if got != self.sha256:
            raise RuntimeError(f"name list changed: sha256 {got} != {self.sha256} ({self.url})")
        pool = name_pool_from_csv(raw.decode("utf-8"), self.min_frequency)
        pool = {cc: v for cc, v in pool.items() if len(v) >= self.min_per_country}
        if len(pool) < 100:
            raise RuntimeError(f"only {len(pool)} countries with >= {self.min_per_country} names")
        json.dump({"source": self.url, "sha256": got, "countries": pool}, open(self.out_names.get_path(), "w"))
        distinct = {n for v in pool.values() for n, _ in v}
        print(f"[first-names] {len(pool)} countries, {sum(map(len, pool.values()))} entries, {len(distinct)} names")


#: v2's instruction category: v1's text and example, lengths in words or seconds (a sentence count depends on the
#: recogniser's punctuation; v1's example asked for "one short sentence").
SP_V2_INSTRUCTION = (
    SP_TRAIN_CATEGORY_TEXT["instruction"] + "; any length limit is in words or seconds, never in sentences",
    (
        SP_TRAIN_EXAMPLES["instruction"][0],
        SP_TRAIN_EXAMPLES["instruction"][1],
        SP_TRAIN_EXAMPLES["instruction"][2].replace("in one short sentence", "in ten words or fewer"),
    ),
)
#: v2 levels: v1's, except that detailed instructions are things to say, mention or ask, or how to speak with a
#: number, and carry their facts. On strict audits of the first v2 pools nearly every detailed side had a "never
#: reveal" item the other side never asks about (the judge learns nothing from it) and many told a side to state a
#: price or time without giving it.
SP_V2_LEVELS = {
    **SP_TRAIN_LEVELS,
    "detailed": (28, 45, "a first name, a role, a goal, and two or three explicit, checkable instructions: things to "
                 "say, mention or ask, or how to speak, with a number of words or seconds; every fact, price, time or "
                 "name the side must say is written in its instructions with its exact value"),
}  # fmt: skip
#: Added to every v2 pool's request (hashed with the prompts): what the audits found missing across categories.
SP_V2_CONTENT_RULE = (
    "Every fact, price, time, name or number a side is told to give, state or confirm is written in that side's "
    "instructions with its exact value. A side may be told to keep something back only if the other side is told to "
    "ask about exactly that. Rules on how to answer (only yes or no, numbers only, one word) must fit the questions "
    "the other side is told to ask. Everything happens by voice: nothing to show, demonstrate, point at, write, send "
    "or open on a screen. No medicine doses or medical advice."
)
#: Per-category additions to the definition for v2 (the v1 texts stay as v1's jobs read them).
SP_V2_EXTRA = {
    "topic": "; the curious side's questions can be answered from common knowledge or from facts in the other side's "
    "instructions, and neither side is a service provider serving a customer",
    "task": "; one side needs something concrete and the other can provide it, and the prices, times and details the "
    "provider must state are in its instructions",
    "tension": "; each side holds a clear position it does not give up in its first reply, neither side's instructions "
    "contain a ready-made compromise, and any secret is the hidden reason for that side's own position",
    "teach_check": "; the teacher's instructions state exactly what is taught (the words, rule, numbers or steps) and "
    "end with correcting any mistake; the learner's instructions never contain what is taught; only things that can "
    "be said back (words, phrases, rules, numbers, steps in order), nothing done with the hands or eyes",
}
#: v2 scene rules (``v2_scene_reject``): "never reveal" items in categories that have no use for them, acts not
#: possible by voice, manner rules without a number, and (teach_check) the taught content in the learner's prompt.
_V2_SECRET = r"\b(never|not|don't) (reveal|disclose|share)\b|\bconfidential\b"
_V2_NOT_VOICE = (
    r"\b(demonstrat\w*|show (?:him|her|them|you|the other|how)|perform the|point (?:to|at|east|west|north|south)|links?|"
    r"URL|download\w*|slides?|e-?mail\w*|send (?:the|a|an|it|them|you)|chat|on (?:the|a|your) screen|bullet points?|"
    r"write (?:it|down|them))\b"
)
_V2_VAGUE = (
    r"\b(briefly|concisely|concise(?: answers?| replies)?|brief (?:answers?|replies|comments?|sentences|phrases)|"
    r"short (?:phrases|sentences|answers?|replies))\b"
    r"(?![^.]{0,40}\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty)\b)"
)


def v2_scene_reject(category: str, prompt_a: str, prompt_b: str) -> str:
    """ "" or why a v2 scenario fails a rule: ``secret`` (a "never reveal" item outside instruction and tension, where
    one is a checkable behaviour), ``not_voice``, ``vague_manner``, ``taught_in_learner`` (teach_check: the learner's
    instructions share two or more 4-word runs with the teacher's after the name)."""
    import re

    both = prompt_a + " " + prompt_b
    if category not in ("instruction", "tension") and re.search(_V2_SECRET, both, re.I):
        return "secret"
    if re.search(_V2_NOT_VOICE, both, re.I):
        return "not_voice"
    if re.search(_V2_VAGUE, both, re.I):
        return "vague_manner"
    if category == "teach_check":
        body = lambda p: p.split(".", 1)[1] if "." in p else p  # noqa: E731
        if len(word_ngrams(body(prompt_a), 4) & word_ngrams(body(prompt_b), 4)) >= 2:
            return "taught_in_learner"
    return ""


#: v2-only categories (2026-11-04), each a ``SelfPlayScenarioPool`` of its own: (definition, example). Kept apart from
#: ``SP_TRAIN_CATEGORY_TEXT``, which v1's declared jobs read.
SP_V2_CATEGORIES = {
    "exchange_details": (
        "the two must pass specific details to each other out loud: one side holds facts the other needs (a phone "
        "number, an address, a booking code, a date and time, a name to spell, an order with quantities) and gives them "
        "when asked; the other must get them right, reading them back or asking for a repeat",
        ("flower_delivery",
         "Your name is Hana. You run a flower shop and take a phone order. Get the delivery address, the recipient's "
         "name and a phone number, and read each one back to check it.",
         "Your name is Tomas. You are ordering roses for your aunt Elena Ruiz at 14 Harbour Lane, flat 3. Your phone "
         "number is 0171 555 2093. Give these details when asked."),
    ),
    "clarify": (
        "B asks A for something but leaves out one specific detail A needs to act (which item, size, date, address, "
        "account); B's instructions state that detail's exact value and tell B to give it only when asked; A's "
        "instructions say what A does and what A must find out before acting, never the value; A must notice the gap "
        "and ask a clarifying question instead of guessing. B is the customer or the one asking for help, and the "
        "detail is B's own choice or B's own number, something A could not know or look up (never a fact A's own job "
        "would give A: the platform A announces, A's clinic address). At the detailed level the extra instructions are "
        "only about how to speak or reading the detail back: no secrets to keep and no other detail to confirm",
        ("allergy_refill",
         "Your name is Priya. You work at a pharmacy counter. Before you refill anything, make sure you know exactly "
         "which medicine and which strength the customer means.",
         "Your name is Leon. You want a refill of your usual allergy pills. They are cetirizine, 10 milligrams, but "
         "only say so if you are asked."),
    ),
    "teach_check": (
        "one side teaches the other something short and concrete (a word in another language, a rule, a recipe step, "
        "a fact) and then checks it landed, by asking the learner to say it back or apply it, and corrects mistakes; "
        "the learner listens, may ask one question, and answers the check",
        ("knight_move",
         "Your name is Rosa. Teach the other person how the knight moves in chess, then ask them to explain it back to "
         "you and correct any mistake.",
         "Your name is Kwame. You have never played chess. Listen, ask one question if something is unclear, and "
         "explain the move back when asked."),
    ),
}  # fmt: skip


#: Categories whose point is a detail one side holds: the generator also names the holder and the exact values,
#: and a scenario whose values are not all in the holder's instructions, or appear on the other side, is dropped
#: (``held_values_reject``). On a strict audit of the first runs, 34 of 100 clarify and 17 of 100 exchange_details
#: scenarios told a side to give a detail "when asked" without stating it. ``(holder, example holder, example values)``.
SP_V2_VALUES = {
    "exchange_details": ("A or B", "B", ["Elena Ruiz", "14 Harbour Lane, flat 3", "0171 555 2093"]),
    "clarify": ("B", "B", ["cetirizine", "10 milligrams"]),
}
SP_V2_VALUES_RULE = (
    'Also give "holder": the side whose instructions state the details ({holder}), and "values": each exact detail '
    "the holder gives (a number, code, name, address, date, size, item), copied word for word from the holder's "
    "instructions. Every scenario states its values, at every level of detail. The other side's instructions contain "
    'none of them. Write each detail in a full sentence ("Your phone number is 0171 555 2093."), never as a list, and '
    "no years."
)
_SP_POOL_VALUES_SCHEMA = json.loads(json.dumps(_SP_TRAIN_JSON_SCHEMA))
_SP_POOL_VALUES_SCHEMA["properties"]["scenarios"]["items"]["properties"].update(
    holder={"type": "string", "enum": ["A", "B"]},
    values={
        "type": "array",
        "minItems": 1,
        "maxItems": 5,
        "items": {"type": "string", "minLength": 1, "maxLength": 60},
    },
)
_SP_POOL_VALUES_SCHEMA["properties"]["scenarios"]["items"]["required"] += ["holder", "values"]


def held_values_reject(category: str, prompt_a: str, prompt_b: str, holder: str, values: list[str]) -> str:
    """ "" or why a scenario's held details fail: ``wrong_holder`` (clarify: B holds them), ``value_missing`` (a value not
    in the holder's instructions), ``value_leaked`` (a value in the other side's), ``value_in_request`` (clarify: the
    value in the sentence where B says what B wants, so there is nothing left to ask about)."""
    import re

    def norm(t: str) -> str:
        t = re.sub(r"[\u2010-\u2015]", "-", t.lower().replace("\u2019", "'"))
        return " ".join(re.sub(r"[^\w'@:/+-]+", " ", t).split())

    if category == "clarify" and holder != "B":
        return "wrong_holder"
    own, other = (prompt_a, prompt_b) if holder == "A" else (prompt_b, prompt_a)
    vals = [norm(v) for v in values if norm(v)]
    if not vals or any(f" {v} " not in f" {norm(own)} " for v in vals):
        return "value_missing"
    if any(f" {v} " in f" {norm(other)} " for v in vals):
        return "value_leaked"
    if category == "clarify" and re.search(
        r"\b(never|not|don't) (reveal|disclose|share|mention)\b|\bconfidential\b", prompt_a + " " + prompt_b, re.I
    ):
        return "secret"
    if category == "clarify":
        for sent in re.split(r"(?<=[.!?])\s+", prompt_b):
            want = re.search(r"\b(want|wants|need|needs|would like|request|ask(?:ing)? for|order|book)\b", sent, re.I)
            if (
                want
                and not re.search(r"\b(if|when|only)\b", sent, re.I)
                and any(f" {v} " in f" {norm(sent)} " for v in vals)
            ):
                return "value_in_request"
    return ""


def _scenario_values_tuples(c: dict, d: dict) -> list:
    """Like ``_scenario_tuples`` plus a sixth field: "" or the ``held_values_reject`` reason."""
    return [(c["category"], x["name"], x["prompt_a"], x["prompt_b"], c["level"],
             held_values_reject(c["category"], x["prompt_a"], x["prompt_b"], x["holder"], x["values"]))
            for x in d["scenarios"]]  # fmt: skip


#: v2 pools: a gpt-oss check of every scenario that passed the rule filters (temperature 0, 10 per request). On the
#: first full v2 run about 1 in 6 clarify scenarios was muddled (the detail one side must find out was not the one the
#: other side holds), which no rule catches.
SP_POOL_CHECK_SYSTEM = """You check role-play scenarios for two voice agents, A and B, who will talk for about a minute. Each agent sees only its own instructions. You get the category the scenarios must fit and a numbered list.

For each scenario decide ok: true only if all of these hold:
- A and B are in the same situation and have a reason to talk to each other;
- together the two instructions fit the category;
- every fact, value or answer an agent is told to give, confirm, check or correct is stated in that agent's own instructions or is common knowledge, and the facts on both sides agree ("give the booking code when asked" with no code stated fails);
- nothing contradicts itself, and nothing needs seeing, reading, writing, sending, showing, demonstrating or doing something physical during the call (no links, cards, downloads, demonstrations);
- no rule on one side blocks what the other side is told to ask or do: "never mention allergies" against a question about allergies fails, and so does "answer only yes or no" against an open question;
- every "never reveal" or "avoid" item is something the other side might plausibly bring up, not a random secret (GPS coordinates, a salary, a password);
- every rule on how to speak can be checked from the words: a number of words or seconds, or a concrete act; "briefly", "concisely" or "melodiously" alone fails, and so does any limit counted in sentences or syllables;
- the facts given are correct and plausible, and nothing asks for medicine doses, medical triage or anything unsafe.
problem: empty if ok, else one short sentence.

Answer with JSON only: {"items": [{"i": 1, "ok": true, "problem": ""}, ...]}"""
_SP_POOL_CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"i": {"type": "integer"}, "ok": {"type": "boolean"},
                               "problem": {"type": "string", "maxLength": 300}},
                "required": ["i", "ok", "problem"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}  # fmt: skip


def pool_check_calls(category: str, definition: str, scenarios: list, per_call: int = 10) -> list[dict]:
    """``scenarios`` (``(category, name, prompt A, prompt B, level)``) -> one check request per ``per_call``."""
    out = []
    for k in range(0, len(scenarios), per_call):
        chunk = scenarios[k : k + per_call]
        items = "\n\n".join(f"{j + 1}.\nA: {s[2]}\nB: {s[3]}" for j, s in enumerate(chunk))
        out.append({"user": f"Category: {category} -- {definition}.\n\n{items}", "seed": k,
                    "scenarios": [s[1] for s in chunk]})  # fmt: skip
    return out


def parse_pool_check(call: dict, reply: dict) -> list:
    got = sorted(reply["items"], key=lambda x: x["i"])
    if [x["i"] for x in got] != list(range(1, len(call["scenarios"]) + 1)):
        raise ValueError("verdicts do not match the items")
    return [[s, bool(x["ok"]), x["problem"]] for s, x in zip(call["scenarios"], got)]


class SelfPlayScenarioPool(Job):
    """One CATEGORY of self-play training scenarios (v2), generated on its own so categories can be added or reweighted
    without regenerating the others (``SelfPlayTrainScenarioMix`` combines pools).

    The category is given by its text: ``definition`` and one ``example`` ``(name, prompt A, prompt B)``, so a new
    category is a new pool, no shared table changes. Per level of detail (``levels``, default ``SP_TRAIN_LEVELS``)
    ``calls_per_level`` gpt-oss calls of ``per_call`` scenarios, each call with a seeded domain and ``per_call``
    seeded name pairs from ``names`` (``FirstNamePool``; ``draw_name_pairs``), stated in the request
    (``SP_POOL_NAMES_RULE``); a scenario using other names is kept with the names it uses (``assign_names``). All go
    through v1's filter (``filter_train_scenarios``: word range, new snake_case name, not near spj2 or a kept
    scenario; v2: spj2's "You ..." style, no sentence counts, Latin script only), then, with ``check``, a gpt-oss
    check of each survivor (``SP_POOL_CHECK_SYSTEM``), up to ``per_level`` per level. Both LLM phases resume after an
    interruption (``generate_scenario_batches`` cache). ``rejected.jsonl`` lists what the check dropped and why.

    Output dir ``pool/``: ``scenarios.json`` (``[category, name, prompt A, prompt B, level]``), ``names.json``
    (``{scenario name: [[name, gender, country] A, [...] B]}``), ``stats.json``."""

    def __init__(
        self,
        *,
        category: str,
        definition: str,
        example: tuple,
        names: tk.Path,
        per_level: int = 200,
        per_call: int = 10,
        calls_per_level: int | None = None,
        levels: dict | None = None,
        domains: tuple = SP_TRAIN_DOMAINS,
        held_out: str = "spj2_v1",
        llm_name: str = "openai/gpt-oss-120b",
        temperature: float = 1.0,
        seed: int = 0,
        check: bool = True,
        values: bool = False,
    ):
        import re

        assert re.fullmatch(r"[a-z][a-z0-9_]*", category), category
        assert not values or category in SP_V2_VALUES, category
        assert len(example) == 3, example
        assert held_out in SPJ2_SCENARIOS, held_out
        self.category, self.definition, self.example = category, str(definition), tuple(example)
        self.names = names
        self.per_level, self.per_call = int(per_level), int(per_call)
        # ~7% of v1's scenarios failed a filter other than "cell full"; 30% headroom
        # 13/10 headroom in v1; v2's style filter (``require_you``) dropped up to a quarter of a level
        # 2026-11-04: 20/10, the scene rules and the stricter check left instruction 40 scenarios of headroom at 16/10
        self.calls_per_level = int(calls_per_level or -(-self.per_level * 20 // (10 * self.per_call)))
        self.levels = dict(levels or SP_TRAIN_LEVELS)
        self.domains, self.held_out = tuple(domains), held_out
        self.llm_name, self.temperature, self.seed = llm_name, float(temperature), int(seed)
        self.check, self.values = bool(check), bool(values)
        self.out_dir = self.output_path("pool", directory=True)
        self.rqmt = {"gpu": 1, "cpu": 4, "mem": 32, "time": 2, "gpu_mem_gb": vllm_gpu_mem_gb(llm_name, 80)}

    @classmethod
    def hash(cls, parsed_args):
        """The held-out set's content, the generator and check prompts and the resolved levels are hashed."""
        prompt = json.dumps([SP_TRAIN_SYSTEM, SP_POOL_NAMES_RULE, SP_POOL_STYLE_RULE, SP_POOL_CHECK_SYSTEM,
                             SP_V2_CONTENT_RULE, _V2_SECRET, _V2_NOT_VOICE, _V2_VAGUE])  # fmt: skip
        if parsed_args.get("values"):
            prompt += json.dumps([SP_V2_VALUES, SP_V2_VALUES_RULE, _FRAGMENT, _SENTENCE_COUNT])
        return super(SelfPlayScenarioPool, cls).hash(
            {
                **parsed_args,
                "levels": dict(parsed_args["levels"] or SP_TRAIN_LEVELS),
                "held_out_digest": scenario_digest(parsed_args["held_out"]),
                "prompt_digest": hashlib.sha256(prompt.encode()).hexdigest()[:16],
            }
        )

    def tasks(self):
        yield Task("run", rqmt=self.rqmt, resumable=True)

    def completed_fraction(self):
        from .common import job_progress_fraction

        return job_progress_fraction(self)

    def calls(self, pool: dict) -> list[dict]:
        """The generation requests; ``pool`` = ``FirstNamePool``'s ``countries``. Seeded per category, so two pools
        draw different domains and names."""
        import numpy as np

        rng = np.random.default_rng([self.seed, int(hashlib.sha256(self.category.encode()).hexdigest()[:8], 16)])
        ex = {"name": self.example[0], "prompt_a": self.example[1], "prompt_b": self.example[2]}
        vrule = ""
        if self.values:
            holder, ex_holder, ex_values = SP_V2_VALUES[self.category]
            ex.update(holder=ex_holder, values=ex_values)
            vrule = " " + SP_V2_VALUES_RULE.format(holder=holder)
        example = json.dumps(ex)
        out = []
        for level, (lo, hi, desc) in self.levels.items():
            for _ in range(self.calls_per_level):
                dom = self.domains[int(rng.integers(len(self.domains)))]
                pairs = draw_name_pairs(pool, self.per_call, rng)
                names = "\n".join(f"{k + 1}. A: {a[0]}, B: {b[0]}" for k, (a, b) in enumerate(pairs))
                user = (
                    f"Category: {self.category} -- {self.definition}.\n"
                    f"Level of detail: {level} -- {desc}; {lo} to {hi} words per side.\n"
                    f"Domain for this batch: {dom}.\n\n"
                    f"Example of the category (its level of detail may differ; do not reuse its situation or "
                    f"names):\n{example}\n\n"
                    f"Write {self.per_call} new, varied scenarios of this category, at this level of detail, in "
                    f"this domain. {SP_POOL_STYLE_RULE} {SP_V2_CONTENT_RULE}{vrule}\n\n{SP_POOL_NAMES_RULE}\n{names}"
                )
                out.append(
                    {
                        "category": self.category,
                        "level": level,
                        "domain": dom,
                        "names": [[list(a), list(b)] for a, b in pairs],
                        "user": user,
                        "seed": int(rng.integers(2**31)),
                    }
                )
        return out

    def run(self):
        import collections

        pool = json.load(open(self.names.get_path()))["countries"]
        calls = self.calls(pool)
        print(f"[sp-pool:{self.category}] {len(calls)} calls x {self.per_call} scenarios ({self.llm_name})", flush=True)
        gen = dict(schema=_SP_POOL_VALUES_SCHEMA, parse=_scenario_values_tuples) if self.values else {}
        outs = generate_scenario_batches(self.llm_name, self.temperature, calls, tag=f"sp-pool:{self.category}",
                                         cache_path=os.path.abspath("calls.jsonl"), **gen)  # fmt: skip
        failed = sum(o is None for o in outs)
        if failed > 0.05 * len(calls):
            raise RuntimeError(f"{failed}/{len(calls)} generation calls failed")
        cands, names, followed = assign_names(outs, calls)
        bad_values = collections.Counter(s[5] for s in cands if len(s) > 5 and s[5])
        dropped = [(s[:5], f"values: {s[5]}") for s in cands if len(s) > 5 and s[5]]
        cands = [tuple(s[:5]) for s in cands if not (len(s) > 5 and s[5])]
        scene_why = {s[1]: v2_scene_reject(self.category, s[2], s[3]) for s in cands}
        bad_scene = collections.Counter(w for w in scene_why.values() if w)
        dropped += [(s, f"rule: {scene_why[s[1]]}") for s in cands if scene_why[s[1]]]
        cands = [s for s in cands if not scene_why[s[1]]]
        flags = dict(require_you=True, reject_sentence_counts=True, latin_only=True)
        held = SPJ2_SCENARIOS[self.held_out]
        passed, rej = filter_train_scenarios(cands, held, per_cell=10**9, **flags)
        rej.update(bad_values)
        rej.update(bad_scene)
        if self.check:
            vcalls = pool_check_calls(self.category, self.definition, passed)
            print(f"[sp-pool:{self.category}] checking {len(passed)} scenarios in {len(vcalls)} calls", flush=True)
            vouts = generate_scenario_batches(self.llm_name, 0.0, vcalls, tag=f"sp-pool:{self.category}:check",
                                              system=SP_POOL_CHECK_SYSTEM, schema=_SP_POOL_CHECK_SCHEMA,
                                              parse=parse_pool_check, cache_path=os.path.abspath("check.jsonl"))  # fmt: skip
            if sum(o is None for o in vouts) > 0.05 * len(vcalls):
                raise RuntimeError(f"{sum(o is None for o in vouts)}/{len(vcalls)} check calls failed")
            verdict = {v[0]: v for o in vouts if o for v in o}
            failed_check = [(s, verdict[s[1]][2] if s[1] in verdict else "check failed") for s in passed
                            if not verdict.get(s[1], [None, False])[1]]  # fmt: skip
            gone = {s[1] for s, _ in failed_check}
            passed = [s for s in passed if s[1] not in gone]
            rej["check_failed"] = len(failed_check)
            dropped += failed_check
        # names are already new and unique; this applies the per-level cap in generation order
        kept, cap = filter_train_scenarios(passed, held, per_cell=self.per_level, **flags)
        rej["cell_full"] = cap.get("cell_full", 0)
        check_scenarios([k[:4] for k in kept], categories=None)
        sides = {k[1]: names[(k[2], k[3])] for k in kept}
        first = collections.Counter(s[0] for pair in sides.values() for s in pair if s)
        per_level = {lv: sum(k[4] == lv for k in kept) for lv in self.levels}
        stats = {
            "category": self.category,
            "calls": len(calls),
            "failed_calls": failed,
            "raw": len(cands),
            "used_assigned_names": followed,
            "kept": len(kept),
            "kept_per_level": per_level,
            "rejected": rej,
            "distinct_first_names": len(first),
            "top_first_names": first.most_common(10),
            "countries": len({s[2] for pair in sides.values() for s in pair if s and s[2]}),
        }
        d = self.out_dir.get_path()
        json.dump([list(k) for k in kept], open(os.path.join(d, "scenarios.json"), "w"), indent=1)
        json.dump(
            {k: [list(s) if s else None for s in v] for k, v in sides.items()},
            open(os.path.join(d, "names.json"), "w"),
            indent=1,
        )
        json.dump(stats, open(os.path.join(d, "stats.json"), "w"), indent=1)
        with open(os.path.join(d, "rejected.jsonl"), "w") as f:
            for s, why in dropped:
                f.write(
                    json.dumps({"scenario": s[1], "level": s[4], "prompt_a": s[2], "prompt_b": s[3], "problem": why})
                    + "\n"
                )
        print(f"[sp-pool:{self.category}] {stats}", flush=True)
        if min(per_level.values()) < self.per_level // 2:
            raise RuntimeError(f"too few scenarios kept per level: {per_level}")


# ---- instruction following: one side makes a Tulu 3 persona IF request, the other just answers ----------------------
# Source: Tulu 3's persona instruction-following prompts. Every row carries the IFEval-taxonomy labels of its
# constraints, so the rows to keep are chosen by label, with no LLM in the data path: a row is kept only if every
# constraint can be checked by ear (``SP_IF_SPOKEN_LABELS``). An earlier IF-Chat v3 rewrite (three gpt-oss calls and
# hand-tuned patterns, 2026-11-04) was replaced by this: too many unablated custom steps for a paper.
TULU_PERSONA_IF_REPO = "allenai/tulu-3-sft-personas-instruction-following"  # ODC-BY-1.0
TULU_PERSONA_IF_REVISION = "fe0c7d350c9b4542b8d829a6f1daa1c259f0ba0e"
#: The 8 of the dataset's 25 constraint labels a listener can check (the IFEval types for words said, avoided or
#: repeated, length in words, the ending, a given option, the language, repeating the request). The other 17 are about
#: how text looks (case, punctuation, JSON, bullets, sections, paragraphs, titles, placeholders, highlighting,
#: quotation marks, a postscript, letter counts, two responses split by "******") or, for the number of sentences,
#: decided by the transcriber: speech has no full stops, the ASR's punctuation sets the count the judge reads.
SP_IF_SPOKEN_LABELS = (
    "include keywords",
    "keywords:exclude words",
    "keywords:frequency",
    "length constraints:number of words",
    "specific ending",
    "format:choose one from options",
    "repeat the prompt",
    "response language",
)
#: How ``grader_info`` names each label. The labels are hints, not the checks: on a 25-row sample some rows labelled
#: "repeat the prompt" ask for no repetition, so the judge checks the requirements the request itself states.
SP_IF_LABEL_CHECKS = {
    "include keywords": "words to include",
    "keywords:exclude words": "words to avoid",
    "keywords:frequency": "how often a word is said",
    "length constraints:number of words": "length in words",
    "specific ending": "the ending",
    "format:choose one from options": "one of given options",
    "repeat the prompt": "repeating the request",
    "response language": "the language",
}
#: Longest request kept, in words: B's prompt (name + opening + request) stays within ``SP_IF_MAX_WORDS``.
SP_IF_PROMPT_WORDS = 50
#: Word cap of B's prompt: 70 words + the Minimal line is ~110 text tokens, inside the self-play prefix budget (KV
#: context = conversation + 256 frames; voice 50 + 12 silence + text).
SP_IF_MAX_WORDS = 70
#: A request whose text still asks for something visual or written, an unlabelled constraint the labels miss (on a
#: sample of 30 rows: "Format: Bullet points", "2 paragraphs separated by ^^^^"; on 40 written scenarios: a UI
#: mockup, "changing . to !"), or an answer far longer than a one-minute call ("exactly 150 words", "a 300-word
#: article", "detailed"): the row is dropped, never edited.
SP_IF_REJECT = (
    r"\b(capital(ized)? letters?|all caps|upper ?case|lower ?case|commas?|punctuation|quotation marks?|"
    r"double quotes|brackets?|asterisks?|bold|italics?|bullets?|bullet points?|markdown|headings?|titled?|sections?|"
    r"paragraphs?|blank lines?|line breaks?|json|tables?|columns?|code|format(ted|ting)?|placeholders?|postscript|"
    r"highlight(ed)?|numbered|font|layout|pdf|emoji|separated by|wrap(ped)?|graphic|logo|poster|diagram|chart|"
    r"infographic|slides?|sketch|pages?|documents?|detailed|comprehensive|in-depth|mock-?ups?|murals?|"
    r"user interface|web ?site design|(?:this|the) persona|screenshots?|images?|photos?|photographs?|captions?|pictures?|illustrations?|"
    r"instrumental|sheet music|chord progressions?|datasets?|spreadsheets?|in the local area|near me|kubernetes|docker|"
    r"terraform|yaml|sql|command line|terminal commands|ipa|IPA|phonetic)\b|"
    r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|single)[- ]sentences?\b|"
    r"\b(develop|build|implement|create|code) (?:an? )?(?:\w+ ){0,3}(feature|app|application|program|website|function|"
    r"algorithm|class|script for an? \w+ app)\b|"
    r"\b([1-9]\d{2,}|\d{1,3},\d{3})[- ]words?\b|\^\^|\*\*|<<|>>|\[|\]|\bp\.\s?s\.?(?=\W)|\? to !|"
    r"\. to [!?]|\b(period|full stop)s? (to|with) (an? )?(question|exclamation)"
)
#: An answer asked for in another language: the duplex models speak English only.
SP_IF_OTHER_LANGUAGE = (
    r"\b(?i:in|into|using) (Spanish|French|German|Italian|Portuguese|Dutch|Swedish|Norwegian|Danish|Finnish|Polish|Czech|"
    r"Russian|Ukrainian|Greek|Turkish|Arabic|Hebrew|Persian|Hindi|Urdu|Bengali|Japanese|Chinese|Mandarin|Cantonese|"
    r"Korean|Vietnamese|Thai|Indonesian|Malay|Tagalog|Swahili|Latin|Hungarian|Romanian|Catalan|Basque|Welsh|Irish)\b"
)
#: Side A of a question scenario: its name and one of these, nothing about the request.
SP_IF_ANSWERER = (
    "Answer the other person's questions.",
    "Help the other person with whatever they ask.",
    "The other person will ask you something; answer it.",
    "Answer whatever the caller asks you.",
    "Someone will ask you for help; do your best to answer.",
)
#: Side B's opening before the request, seeded per scenario.
SP_IF_INTROS = (
    "Ask the other person for this, in your own words, keeping every requirement:",
    "You need help with this. Ask for it in your own words and keep every requirement:",
)
_IF_TEXT_ONLY = (
    r"```",  # code
    r"^\s*\|.*\|\s*$",  # table rows
    r"\{\s*\"",  # JSON
    r"\bNAME_\d",  # anonymised names
    r"https?://",  # links
)
#: "What would X say ...? Imagine you are X and answer": role-play rows, mostly labelled "repeat the prompt" though
#: they ask for no repetition; on 60 written scenarios, 4 of the 5 muddled ones (B took the role it should ask A to
#: play).
_IF_ROLE_PLAY = r"\b(imagine|pretend) (that )?you are\b|\bas if you were\b"
_IF_TIME_SENSITIVE = (
    r"\b(today|tonight|this (?:week|month|year)|latest|current(?:ly)?|right now|news|stock price|20[2-3]\d|upcoming|"
    r"next \d+ (?:days|weeks|months)|next (?:week|month)|"
    r"(?:last|recent|past) (?:\d+|two|three|four|five|few|several) (?:matches|games|races|seasons|episodes))\b"
)


def persona_if_candidate(row: dict, max_words: int = SP_IF_PROMPT_WORDS) -> tuple[str | None, str]:
    """A Tulu 3 persona IF row -> (its prompt, whitespace-collapsed, "") or (None, reason). Reasons, in order:
    ``written_label`` (a constraint label outside ``SP_IF_SPOKEN_LABELS``), ``long`` (over ``max_words``),
    ``text_only`` (code, tables, JSON, placeholders, links), ``written_text`` (``SP_IF_REJECT``), ``other_language``
    (``SP_IF_OTHER_LANGUAGE``), ``time_sensitive``, ``role_play`` (``_IF_ROLE_PLAY``), ``non_english`` (under 97% ASCII)."""
    import re

    labels = {str(c).strip() for c in row.get("constraints") or []}
    prompt = " ".join(str(row.get("prompt") or "").split())
    if not labels or not labels <= set(SP_IF_SPOKEN_LABELS):
        return None, "written_label"
    if not prompt or len(prompt.split()) > max_words:
        return None, "long"
    if any(re.search(p, row.get("prompt") or "", re.M) for p in _IF_TEXT_ONLY):
        return None, "text_only"
    if re.search(SP_IF_REJECT, prompt, re.I):
        return None, "written_text"
    if re.search(SP_IF_OTHER_LANGUAGE, prompt):
        return None, "other_language"
    if re.search(_IF_TIME_SENSITIVE, prompt, re.I):
        return None, "time_sensitive"
    if re.search(_IF_ROLE_PLAY, prompt, re.I):
        return None, "role_play"
    if sum(ch.isascii() for ch in prompt) < 0.97 * len(prompt):
        return None, "non_english"
    return prompt, ""


def persona_if_grader_info(prompt: str, labels: list[str]) -> str:
    """The judge's notes for one request: its exact wording (B says it in its own words) and the source's constraint
    labels as hints (``SP_IF_LABEL_CHECKS``)."""
    hints = ", ".join(SP_IF_LABEL_CHECKS[lb] for lb in sorted(labels))
    return (
        f'B was asked to request: "{prompt}" Check that A meets every requirement this request states '
        f"(the source data labels them as: {hints})."
    )


class SelfPlayPersonaIfPool(Job):
    """A self-play pool of INSTRUCTION-FOLLOWING requests (v2): side B makes one Tulu 3 persona IF request
    (``TULU_PERSONA_IF_REPO`` at a pinned revision), side A gets only its name and a generic answering line
    (``SP_IF_ANSWERER``). No LLM: rows are kept by their IFEval constraint labels (``persona_if_candidate``: every
    label audible, at most ``SP_IF_PROMPT_WORDS`` words, no written-only wording), in a seeded order, up to
    ``n_keep``, none near spj2 or near a kept one (word-5-gram Jaccard of the requests > ``max_jaccard``). B's prompt
    is its name, a seeded opening (``SP_IF_INTROS``) and the request verbatim; ``grader_info`` gives the judge the
    exact request and what each label requires.

    Output dir ``pool/`` as ``SelfPlayScenarioPool``'s (level ``persona_if``) plus ``grader_info.json`` and
    ``provenance.jsonl`` (per scenario: Tulu row id, revision, prompt, constraint labels). A mini task: the login node
    downloads the 30k-row parquet."""

    LEVEL = "persona_if"

    def __init__(
        self,
        *,
        names: tk.Path,
        n_keep: int = 800,
        category: str = "ask_and_answer",
        held_out: str = "spj2_v1",
        max_jaccard: float = 0.4,
        revision: str = TULU_PERSONA_IF_REVISION,
        seed: int = 0,
    ):
        assert held_out in SPJ2_SCENARIOS, held_out
        self.names, self.n_keep, self.category, self.held_out = names, int(n_keep), category, held_out
        self.max_jaccard, self.revision, self.seed = float(max_jaccard), revision, int(seed)
        self.out_dir = self.output_path("pool", directory=True)

    @classmethod
    def hash(cls, parsed_args):
        """The held-out set's content and the selection rules (labels, patterns, caps, prompt lines) are hashed."""
        rules = [SP_IF_SPOKEN_LABELS, SP_IF_LABEL_CHECKS, SP_IF_PROMPT_WORDS, SP_IF_MAX_WORDS, SP_IF_REJECT,
                 _IF_TEXT_ONLY, _IF_TIME_SENSITIVE, SP_IF_ANSWERER, SP_IF_INTROS, SP_IF_OTHER_LANGUAGE, _IF_ROLE_PLAY]  # fmt: skip
        return super(SelfPlayPersonaIfPool, cls).hash(
            {
                **parsed_args,
                "held_out_digest": scenario_digest(parsed_args["held_out"]),
                "rules_digest": hashlib.sha256(json.dumps(rules).encode()).hexdigest()[:16],
            }
        )

    def tasks(self):
        yield Task("run", mini_task=True)

    def select(self, rows: list[dict], pool: dict) -> tuple[list, dict]:
        """``rows`` (``id``, ``prompt``, ``constraints``) and ``FirstNamePool`` countries -> (scenarios
        ``[(category, name, prompt A, prompt B, level, grader_info, row)]``, stats)."""
        import collections

        import numpy as np

        rng = np.random.default_rng([self.seed, int(hashlib.sha256(self.category.encode()).hexdigest()[:8], 16)])
        why = collections.Counter()
        cands = []
        for r in rows:
            p, reason = persona_if_candidate(r)
            why[reason or "candidate"] += 1
            if p:
                cands.append((p, r))
        held_grams = [word_ngrams(s[2] + " " + s[3]) for s in SPJ2_SCENARIOS[self.held_out]]
        kept, kept_grams = [], []
        for j in rng.permutation(len(cands)):
            if len(kept) == self.n_keep:
                break
            p, r = cands[int(j)]
            g = word_ngrams(p)
            if any(ngram_jaccard(g, h) > 0.2 for h in held_grams):
                why["near_spj2"] += 1
                continue
            if any(ngram_jaccard(g, h) > self.max_jaccard for h in kept_grams):
                why["near_duplicate"] += 1
                continue
            ((a, b),) = draw_name_pairs(pool, 1, rng)
            pa = f"Your name is {a[0]}. {SP_IF_ANSWERER[int(rng.integers(len(SP_IF_ANSWERER)))]}"
            pb = f"Your name is {b[0]}. {SP_IF_INTROS[int(rng.integers(len(SP_IF_INTROS)))]} {p}"
            labels = sorted({str(c).strip() for c in r["constraints"]})
            name = "if_" + hashlib.sha256(str(r["id"]).encode()).hexdigest()[:12]
            kept.append((self.category, name, pa, pb, self.LEVEL, persona_if_grader_info(p, labels),
                         {"names": [list(a), list(b)], "id": r["id"], "prompt": r["prompt"], "constraints": labels}))  # fmt: skip
            kept_grams.append(g)
        stats = {"rows": len(rows), "candidates": len(cands), "kept": len(kept), "reasons": dict(why),
                 "labels": dict(collections.Counter(lb for k in kept for lb in k[6]["constraints"])),
                 "labels_per_request": dict(collections.Counter(len(k[6]["constraints"]) for k in kept))}  # fmt: skip
        return kept, stats

    def run(self):
        import io

        import pyarrow.parquet as pq
        import requests
        from huggingface_hub import HfApi, hf_hub_url
        from huggingface_hub.utils import build_hf_headers

        files = [f for f in HfApi().list_repo_files(TULU_PERSONA_IF_REPO, repo_type="dataset", revision=self.revision)
                 if f.endswith(".parquet")]  # fmt: skip
        rows = []
        for f in sorted(files):
            url = hf_hub_url(TULU_PERSONA_IF_REPO, f, repo_type="dataset", revision=self.revision)
            resp = requests.get(url, headers=build_hf_headers(), timeout=300)
            resp.raise_for_status()
            rows += pq.read_table(io.BytesIO(resp.content), columns=["id", "prompt", "constraints"]).to_pylist()
        pool = json.load(open(self.names.get_path()))["countries"]
        kept, stats = self.select(rows, pool)
        stats.update({"files": sorted(files), "revision": self.revision})
        check_scenarios([k[:4] for k in kept], categories=None, max_words=SP_IF_MAX_WORDS)
        d = self.out_dir.get_path()
        json.dump([list(k[:5]) for k in kept], open(os.path.join(d, "scenarios.json"), "w"), indent=1)
        json.dump({k[1]: k[6]["names"] for k in kept}, open(os.path.join(d, "names.json"), "w"), indent=1)
        json.dump({k[1]: k[5] for k in kept}, open(os.path.join(d, "grader_info.json"), "w"), indent=1)
        with open(os.path.join(d, "provenance.jsonl"), "w") as f:
            for k in kept:
                prov = {"scenario": k[1], "tulu_id": k[6]["id"], "revision": self.revision, "prompt": k[6]["prompt"],
                        "constraints": k[6]["constraints"]}  # fmt: skip
                f.write(json.dumps(prov) + "\n")
        json.dump(stats, open(os.path.join(d, "stats.json"), "w"), indent=1)
        print(f"[sp-pool:{self.category}] {stats}", flush=True)
        if len(kept) < self.n_keep:
            raise RuntimeError(f"only {len(kept)} requests for n_keep={self.n_keep}")


# ---- ask_and_answer as role-play: the persona IF requests written into scenarios by the v2 generator ---------------
# ``SelfPlayPersonaIfPool`` picks the requests (by label, no LLM); this job has the same generator as the other v2
# pools (``SP_TRAIN_SYSTEM``, assigned names, ``generate_scenario_batches``) write a scenario around each one: B gets a
# reason to ask and the request worded as spoken, A a role suited to answering. A rewrite that loses a requirement is
# dropped, never repaired: a quoted phrase or number missing from B (``persona_if_scene_reject``), then a gpt-oss check
# of every requirement (``SP_IF_VERIFY_SYSTEM``). The verbatim pool stays declared, so the rewrite can be ablated.
SP_IF_SCENE_DEFINITION = (
    "B asks A out loud for what a written request asks for; A answers. Give B's instructions in three parts. b_who: "
    'one short sentence on who B is and why B needs it ("You ..."). b_ask: one sentence starting with "Ask" with the '
    'task and its facts but no requirements ("Ask for a short toast for your sister\'s wedding", "Ask whether ..."). '
    'b_requirements: every requirement of the request, each completing "Ask that ..." and starting with "it" ("it '
    'never uses the word \u201cx\u201d", "it ends with the phrase \u201cy\u201d", "it picks one of \u201cX\u201d, \u201cY\u201d or '
    '\u201cZ\u201d and says it word for word"); quoted words exactly and in curly quotes, every number kept. If the request '
    'wants itself repeated, b_ask quotes the question and a requirement is "it first repeats that question word for '
    "word, then answers it\". If the request casts the answerer as someone, B asks A to answer as that person. A's "
    "instructions say who A is, someone suited to the topic, and nothing about the requirements. B's parts never use "
    "A's name; no brackets, dashes or lists"
)
#: ``(request, name, prompt A, b_who, b_ask, b_requirements)``, written for this job (not a Tulu row).
SP_IF_SCENE_EXAMPLE = (
    'Write a short toast for my sister\'s wedding that does not use the word "marriage" and ends with the phrase '
    '"to love and laughter".',
    "wedding_toast",
    "Your name is Ines. You write speeches and toasts for people who are nervous about speaking in public.",
    "Your name is Malik. Your sister marries on Saturday and you have to give a toast.",
    "Ask for a short toast for your sister's wedding",
    ["it never uses the word \u201cmarriage\u201d", "it ends with the phrase \u201cto love and laughter\u201d"],
)
#: An assembled requirement must open as a clause about the answer and never address B as the answerer.
_IF_REQ_START = (
    r"(it|the answer|the (?:\w+ )?(?:reply|response|story|poem|list|toast|speech|letter|song|description))\b"
)
_IF_REQ_BAD = r"\byour (response|answer|reply)\b|\byou (repeat|use|include|end|say|choose|avoid|write)\b"


def build_if_prompt_b(who: str, ask: str, reqs: list[str]) -> tuple[str, bool]:
    """B's instructions from the generator's parts -> (text, every requirement well-formed). Each requirement must
    complete "Ask that ..." as a clause about the answer (``_IF_REQ_START``), never an order to B or "your answer"."""
    import re

    def clip(t: str) -> str:
        return " ".join(t.split()).strip(" .;,")

    ask = clip(ask)
    reqs = [re.sub(r"^(?:ask )?that\s+", "", clip(r), flags=re.I) for r in reqs if clip(r)]
    reqs = [r[0].lower() + r[1:] if re.match(r"(It|The)\b", r) else r for r in reqs]
    ok = all(re.match(_IF_REQ_START, r) and not re.search(_IF_REQ_BAD, r, re.I) for r in reqs)
    # the ask: a sentence opening "Ask" (not "Ask for whether ...", not the raw request in quotes), holding no
    # requirement (a quoted phrase in both parts was stated twice), never B's own words or an order to B
    req_quotes = {q for r in reqs for q in re.findall(r"\u201c([^\u201d]+)\u201d", r)}
    ok = ok and bool(
        re.match(
            r"Ask\b(?! that\b)(?! for (?:whether|if|how|what|why|list|write|create|provide)\b)(?! (?:for )?\u201c)", ask
        )
        and not re.search(r"\byour (response|answer|reply)\b|\b(my|do not|don't)\b|\bAsk that\b", ask, re.I)
        and not req_quotes & set(re.findall(r"\u201c([^\u201d]+)\u201d", ask))
    )
    text = f"{clip(who)}. {ask}."
    if reqs:
        text += " Ask that " + (", and that ".join(reqs) if len(reqs) < 3 else
                                ", that ".join(reqs[:-1]) + ", and that " + reqs[-1]) + "."  # fmt: skip
    return text, ok


#: Word range of A's instructions (B's cap is ``SP_IF_MAX_WORDS``).
SP_IF_SCENE_A_WORDS = (8, 30)
SP_IF_VERIFY_SYSTEM = """You check rewritten requests. Each item has an ORIGINAL written request, the instructions B gets for asking for it in a spoken conversation, and the instructions A (who answers) gets.

For each item decide:
- keeps_all: true only if B, asking as instructed, states the original's question or task itself, with every fact or claim it gives, and every requirement of the original: each word or phrase to include, avoid or repeat, each count (sentences, words, how often), each option to choose from, the exact ending, repeating the request, the language. Different wording is fine; a requirement that is lost, weakened, loosened or changed is not, and neither is an added one.
- b_asks: true if B's instructions clearly tell B to ask A for it, with every requirement stated as part of what B asks of the answer. False if they tell B to write or answer it, speak to the answerer directly ("Provide ...", "Could you ...") instead of telling B what to ask, give a requirement as an order to B itself ("Ask for X, and do not mention the word Y" tells B to avoid Y instead of asking A to), are muddled about who asks what, or make B the person the answerer is asked to imagine being.
- a_neutral: false only if A's instructions state or hint at a requirement on the answer's form: words to use or avoid, a count, the ending, options, repeating, the language. A role that fits the topic is fine.
- problem: empty if all three are true, else one short sentence.

Answer with JSON only: {"items": [{"i": 1, "keeps_all": true, "b_asks": true, "a_neutral": true, "problem": ""}, ...]}"""
_SP_IF_VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "i": {"type": "integer"},
                    "keeps_all": {"type": "boolean"},
                    "b_asks": {"type": "boolean"},
                    "a_neutral": {"type": "boolean"},
                    "problem": {"type": "string", "maxLength": 300},
                },
                "required": ["i", "keeps_all", "b_asks", "a_neutral", "problem"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}
#: The generator's reply: the request's number (alignment), A's instructions and B's in parts (``build_if_prompt_b``):
#: free-written B text gave requirements as orders to B in about 40% of a strict audit's sample.
_SP_IF_SCENE_SCHEMA = {
    "type": "object",
    "properties": {
        "scenarios": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "request": {"type": "integer"},
                    "name": {"type": "string", "minLength": 3, "maxLength": 40},
                    "prompt_a": {"type": "string", "minLength": 20, "maxLength": 400},
                    "b_who": {"type": "string", "minLength": 15, "maxLength": 400},
                    "b_ask": {"type": "string", "minLength": 3, "maxLength": 400},
                    "b_requirements": {
                        "type": "array",
                        "maxItems": 8,
                        "items": {"type": "string", "minLength": 3, "maxLength": 240},
                    },
                },
                "required": ["request", "name", "prompt_a", "b_who", "b_ask", "b_requirements"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["scenarios"],
    "additionalProperties": False,
}
_NUMBER_WORDS = {
    **{w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                  "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())},
    **{w: 10 * i for i, w in enumerate("thirty forty fifty sixty seventy eighty ninety".split(), 3)},
    "once": 1, "twice": 2, "thrice": 3, "hundred": 100,
}  # fmt: skip


def spoken_numbers(text: str, min_word: int = 0) -> set:
    """The numbers ``text`` states, as digits or words (``twenty-five`` -> 25; a word below ``min_word`` is ignored,
    so the original's "one of" is not a count)."""
    import re

    t = re.sub(r"[\u2010-\u2015]", "-", text.lower().replace(",", ""))
    out = {int(x) for x in re.findall(r"(?<![\d.])\d+(?![\d.]\d)", t)}
    words = "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True))
    for tens, unit in re.findall(rf"\b({words})(?:[- ]({words}))?\b", t):
        a, b = _NUMBER_WORDS[tens], _NUMBER_WORDS.get(unit)
        n = a + b if b is not None and a >= 20 and a % 10 == 0 and 0 < b < 10 else a
        out |= {n} if n >= min_word else set()
        if b is not None and n == a and b >= min_word:  # two separate number words
            out.add(b)
    return out


def quoted_phrases(text: str) -> list[str]:
    """Phrases the request puts in double quotes (straight or curly) or in single quotes that are not apostrophes."""
    import re

    found = re.findall(r'["“”]([^"“”]{1,80})["“”]', text) + re.findall(r"(?<![\w])'([^']{1,60})'(?![\w])", text)
    return [q.strip(" .,!?;:").lower() for q in found if q.strip(" .,!?;:")]


def persona_if_scene_grader_info(prompt_b: str, labels: list[str]) -> str:
    """The judge's notes for a written scenario: B's instructions (what B was told to ask; a rewrite that slipped
    past the checks then still scores A on what B asked) and the source's labels as hints."""
    hints = ", ".join(SP_IF_LABEL_CHECKS[lb] for lb in sorted(labels))
    return (
        f'B was told: "{prompt_b}" Check that A meets every requirement B asks for in the conversation '
        f"(the source data labels them as: {hints})."
    )


def persona_if_scene_reject(request: str, prompt_a: str, prompt_b: str, names: list) -> str:
    """A written scenario's first, rule-based check -> "" or the reason: ``names`` (not opening with the assigned
    pair), ``length`` (A outside ``SP_IF_SCENE_A_WORDS`` with a little slack, B outside 8-``SP_IF_MAX_WORDS``),
    ``b_not_told_to_ask`` (no "ask" in B, or B opens as the request itself: "Could you ...", "I need ...", "Please
    ..."), ``b_names_a`` (B's instructions use A's name, which B cannot know), ``truncated`` (B not ending a sentence), ``written_format`` (brackets, parentheses or "--" left in B), ``copied_example`` (B near the example's B: word-5-gram Jaccard > 0.2),
    ``lost_quote`` (a quoted phrase of the request missing from B), ``lost_number`` (a count of the request missing
    from B, digits or words)."""
    import re

    lo, hi = SP_IF_SCENE_A_WORDS
    if not follows_names(prompt_a, prompt_b, names[0][0], names[1][0]):
        return "names"
    rest_b = prompt_b.split(".", 1)[1] if "." in prompt_b else prompt_b
    if not re.search(r"\bask\b", rest_b, re.I) or re.match(r"\s*(Could|Can|Would|Will) you\b|\s*(Please|I)\b", rest_b):
        return "b_not_told_to_ask"
    if re.search(rf"\b{re.escape(names[0][0])}\b", prompt_b):
        return "b_names_a"
    if not lo <= len(prompt_a.split()) <= hi + 5 or not 8 <= len(prompt_b.split()) <= SP_IF_MAX_WORDS:
        return "length"
    if not re.search(r"[.?!][\"”'’)]*\s*$", prompt_b):
        return "truncated"
    if re.search(r"[()\[\]\\]|--", prompt_b):
        return "written_format"
    if ngram_jaccard(word_ngrams(prompt_b), word_ngrams(build_if_prompt_b(*SP_IF_SCENE_EXAMPLE[3:])[0])) > 0.2:
        return "copied_example"
    b = " ".join(prompt_b.lower().replace("’", "'").split())
    if any(q.replace("’", "'") not in b for q in quoted_phrases(request)):
        return "lost_quote"
    if not spoken_numbers(request, min_word=2) <= spoken_numbers(prompt_b):
        return "lost_number"
    return ""


class SelfPlayPersonaIfScenes(Job):
    """``ask_and_answer`` scenarios (v2) written by the v2 generator around the persona IF requests of
    ``SelfPlayPersonaIfPool`` (``requests`` = its ``out_dir``; its names, grader notes and order are kept).

    ``per_call`` requests per gpt-oss call under ``SP_TRAIN_SYSTEM``, with the category ``SP_IF_SCENE_DEFINITION``, one
    example (``SP_IF_SCENE_EXAMPLE``) and the requests' assigned names (``SP_POOL_NAMES_RULE``); one scenario per
    request, in order. Each scenario must then pass ``persona_if_scene_reject``, not be near spj2 (pair word-5-gram
    Jaccard > 0.2), and pass a gpt-oss check that B keeps every requirement, is told to ask, and A is told none
    (``SP_IF_VERIFY_SYSTEM``, temperature 0). Kept in the source's order, up to ``n_keep``; the judge's notes are
    built from B's instructions (``persona_if_scene_grader_info``), not the original: A is scored on what B asked. Both LLM phases resume after
    an interruption.

    Output dir ``pool/`` as ``SelfPlayPersonaIfPool``'s (``provenance.jsonl`` adds both prompts) plus
    ``rejected.jsonl`` (scenario, reason, the check's note, the two prompts)."""

    LEVEL = "persona_if"

    def __init__(
        self,
        *,
        requests: tk.Path,
        n_keep: int = 800,
        per_call: int = 5,  # 10 per call hit the token cap on 8 of 110 calls
        category: str = "ask_and_answer",
        held_out: str = "spj2_v1",
        llm_name: str = "openai/gpt-oss-120b",
        temperature: float = 1.0,
        seed: int = 0,
    ):
        assert held_out in SPJ2_SCENARIOS, held_out
        self.requests, self.n_keep, self.per_call, self.category = requests, int(n_keep), int(per_call), category
        self.held_out, self.llm_name, self.temperature, self.seed = held_out, llm_name, float(temperature), int(seed)
        self.out_dir = self.output_path("pool", directory=True)
        self.rqmt = {"gpu": 1, "cpu": 4, "mem": 32, "time": 2, "gpu_mem_gb": vllm_gpu_mem_gb(llm_name, 80)}

    @classmethod
    def hash(cls, parsed_args):
        """The held-out set's content, the prompts and the check's caps are hashed."""
        rules = [SP_TRAIN_SYSTEM, SP_POOL_NAMES_RULE, SP_IF_SCENE_DEFINITION, SP_IF_SCENE_EXAMPLE, SP_IF_SCENE_A_WORDS,
                 SP_IF_MAX_WORDS, SP_IF_VERIFY_SYSTEM, sorted(_NUMBER_WORDS.items()), _SP_IF_SCENE_SCHEMA]  # fmt: skip
        return super(SelfPlayPersonaIfScenes, cls).hash(
            {
                **parsed_args,
                "held_out_digest": scenario_digest(parsed_args["held_out"]),
                "rules_digest": hashlib.sha256(json.dumps(rules).encode()).hexdigest()[:16],
            }
        )

    def tasks(self):
        yield Task("run", rqmt=self.rqmt, resumable=True)

    def completed_fraction(self):
        from .common import job_progress_fraction

        return job_progress_fraction(self)

    @staticmethod
    def read_requests(d: str) -> list[dict]:
        """A ``SelfPlayPersonaIfPool`` dir -> per scenario ``{scenario, request, names, grader_info, provenance}``."""
        names = json.load(open(os.path.join(d, "names.json")))
        notes = json.load(open(os.path.join(d, "grader_info.json")))
        out = []
        for line in open(os.path.join(d, "provenance.jsonl")):
            p = json.loads(line)
            s = p["scenario"]
            out.append({"scenario": s, "request": " ".join(p["prompt"].split()), "names": names[s],
                        "grader_info": notes[s], "provenance": p})  # fmt: skip
        return out

    def calls(self, reqs: list[dict]) -> list[dict]:
        import numpy as np

        rng = np.random.default_rng([self.seed, int(hashlib.sha256(self.category.encode()).hexdigest()[:8], 16)])
        req_ex, *example = SP_IF_SCENE_EXAMPLE
        example = json.dumps({"name": example[0], "prompt_a": example[1], "b_who": example[2], "b_ask": example[3],
                              "b_requirements": example[4]})  # fmt: skip
        lo, hi = SP_IF_SCENE_A_WORDS
        out = []
        for k in range(0, len(reqs), self.per_call):
            chunk = reqs[k : k + self.per_call]
            listed = "\n".join(f"{j + 1}. {r['request']}" for j, r in enumerate(chunk))
            names = "\n".join(f"{j + 1}. A: {r['names'][0][0]}, B: {r['names'][1][0]}" for j, r in enumerate(chunk))
            user = (
                f"Category: {self.category} -- {SP_IF_SCENE_DEFINITION}.\n"
                f"Length: A {lo} to {hi} words; B at most {SP_IF_MAX_WORDS} words.\n\n"
                f"Example, for the request: {req_ex}\n{example}\n\n"
                f"Write {len(chunk)} scenarios, one per request below; give each its request number.\n{listed}\n\n"
                f"{SP_POOL_NAMES_RULE}\n{names}"
            )
            out.append({"user": user, "seed": int(rng.integers(2**31)), "scenarios": [r["scenario"] for r in chunk]})
        return out

    def parse_scenes(self, call: dict, reply: dict) -> list:
        """One scenario per request, matched by its ``request`` number (each number once, else the attempt fails and
        is retried); named after the request; HTML entities decoded."""
        import html

        got = {int(x["request"]): x for x in reply["scenarios"]}
        if sorted(got) != list(range(1, len(call["scenarios"]) + 1)) or len(reply["scenarios"]) != len(got):
            raise ValueError(f"request numbers {sorted(got)} for {len(call['scenarios'])} requests")
        clean = lambda t: " ".join(html.unescape(t).split())  # noqa: E731
        out = []
        for j, s in enumerate(call["scenarios"]):
            g = got[j + 1]
            b, ok = build_if_prompt_b(clean(g["b_who"]), clean(g["b_ask"]), [clean(r) for r in g["b_requirements"]])
            if not ok:
                self._bad_requirement = getattr(self, "_bad_requirement", set()) | {s}
            out.append((self.category, s, clean(g["prompt_a"]), b, self.LEVEL))
        return out

    def verify_calls(self, items: list[tuple]) -> list[dict]:
        """``items`` = ``(request, scenario tuple)``; ``per_call`` per check request."""
        out = []
        for k in range(0, len(items), self.per_call):
            chunk = items[k : k + self.per_call]
            user = "\n\n".join(
                f"Item {j + 1}\nORIGINAL: {req}\nB: {s[3]}\nA: {s[2]}" for j, (req, s) in enumerate(chunk)
            )
            out.append({"user": user, "seed": k, "scenarios": [s[1] for _, s in chunk]})
        return out

    @staticmethod
    def parse_verdicts(call: dict, reply: dict) -> list:
        got = sorted(reply["items"], key=lambda x: x["i"])
        if [x["i"] for x in got] != list(range(1, len(call["scenarios"]) + 1)):
            raise ValueError("verdicts do not match the items")
        return [[s, bool(x["keeps_all"]), bool(x["b_asks"]), bool(x["a_neutral"]), x["problem"]]
                for s, x in zip(call["scenarios"], got)]  # fmt: skip

    def assemble(self, reqs: list[dict], scenes: dict, verdicts: dict) -> tuple[list, list, dict]:
        """``scenes`` / ``verdicts`` by scenario name (verdict ``[name, keeps_all, b_asks, a_neutral, problem]``; a missing
        one = its call failed) -> (kept ``(request dict, scenario)``, rejected rows, stats)."""
        import collections

        why = collections.Counter()
        kept, rejected = [], []
        for r in reqs:
            s, v = scenes.get(r["scenario"]), verdicts.get(r["scenario"])
            if len(kept) == self.n_keep:
                break
            if s is None:
                reason, note = "generation_failed", ""
            else:
                reason, note = self.rule_reject(r, s), ""
                if not reason:
                    if v is None:
                        reason = "verify_failed"
                    elif not all(v[1:4]):
                        reason = ("lost_requirement", "b_not_asking", "a_told_requirement")[[*v[1:4]].index(False)]
                        note = v[4]
            why[reason or "kept"] += 1
            if reason:
                rejected.append({"scenario": r["scenario"], "reason": reason, "note": note, "request": r["request"],
                                 "prompt_a": s[2] if s else "", "prompt_b": s[3] if s else ""})  # fmt: skip
            else:
                kept.append((r, s))
        return kept, rejected, {"requests": len(reqs), "kept": len(kept), "reasons": dict(why)}

    def rule_reject(self, r: dict, s: tuple) -> str:
        import re

        reason = persona_if_scene_reject(r["request"], s[2], s[3], r["names"])
        if not reason and s[1] in getattr(self, "_bad_requirement", ()):
            reason = "bad_requirement"
        if (
            not reason
            and re.search(r"word for word, then answers", s[3])
            and not re.search(r"\u201c[^\u201d]{15,}\u201d", s[3])
        ):
            reason = "repeat_unquoted"  # "repeat the question" needs the exact question in B's words
        if not reason and any(ngram_jaccard(word_ngrams(s[2] + " " + s[3]), h) > 0.2 for h in self._held_grams()):
            reason = "near_spj2"
        return reason

    def _held_grams(self):
        if not hasattr(self, "_held"):
            self._held = [word_ngrams(s[2] + " " + s[3]) for s in SPJ2_SCENARIOS[self.held_out]]
        return self._held

    def run(self):
        reqs = self.read_requests(self.requests.get_path())
        calls = self.calls(reqs)
        tag = f"sp-pool:{self.category}"
        print(f"[{tag}] {len(reqs)} requests, {len(calls)} calls x {self.per_call} ({self.llm_name})", flush=True)
        outs = generate_scenario_batches(self.llm_name, self.temperature, calls, tag=tag, parse=self.parse_scenes,
                                         schema=_SP_IF_SCENE_SCHEMA, cache_path=os.path.abspath("calls.jsonl"))  # fmt: skip
        failed = sum(o is None for o in outs)
        if failed > 0.05 * len(calls):
            raise RuntimeError(f"{failed}/{len(calls)} generation calls failed")
        scenes = {s[1]: s for o in outs if o for s in o}
        to_check = [(r["request"], scenes[r["scenario"]]) for r in reqs
                    if r["scenario"] in scenes and not self.rule_reject(r, scenes[r["scenario"]])]  # fmt: skip
        vcalls = self.verify_calls(to_check)
        print(f"[{tag}] checking {len(to_check)} scenarios in {len(vcalls)} calls", flush=True)
        vouts = generate_scenario_batches(self.llm_name, 0.0, vcalls, tag=f"{tag}:verify", system=SP_IF_VERIFY_SYSTEM,
                                          schema=_SP_IF_VERIFY_SCHEMA, parse=self.parse_verdicts,
                                          cache_path=os.path.abspath("verify.jsonl"))  # fmt: skip
        verdicts = {v[0]: v for o in vouts if o for v in o}
        kept, rejected, stats = self.assemble(reqs, scenes, verdicts)
        stats.update({"calls": len(calls), "failed_calls": failed, "verify_calls": len(vcalls),
                      "failed_verify_calls": sum(o is None for o in vouts)})  # fmt: skip
        check_scenarios([s[:4] for _, s in kept], categories=None, max_words=SP_IF_MAX_WORDS)
        d = self.out_dir.get_path()
        json.dump([list(s) for _, s in kept], open(os.path.join(d, "scenarios.json"), "w"), indent=1)
        json.dump({s[1]: r["names"] for r, s in kept}, open(os.path.join(d, "names.json"), "w"), indent=1)
        notes = {s[1]: persona_if_scene_grader_info(s[3], r["provenance"]["constraints"]) for r, s in kept}
        json.dump(notes, open(os.path.join(d, "grader_info.json"), "w"), indent=1)
        with open(os.path.join(d, "provenance.jsonl"), "w") as f:
            for r, s in kept:
                f.write(json.dumps({**r["provenance"], "prompt_a": s[2], "prompt_b": s[3]}) + "\n")
        with open(os.path.join(d, "rejected.jsonl"), "w") as f:
            for x in rejected:
                f.write(json.dumps(x) + "\n")
        json.dump(stats, open(os.path.join(d, "stats.json"), "w"), indent=1)
        print(f"[{tag}] {stats}", flush=True)
        if len(kept) < self.n_keep:
            raise RuntimeError(f"only {len(kept)} scenarios for n_keep={self.n_keep}")


def mix_counts(sizes: dict, weights: dict, n_rows: int) -> dict:
    """Rows per pool: ``n_rows`` split by ``weights`` (largest remainder, ties by name); raises if a pool is too small
    or a weight has no pool."""
    assert set(weights) <= set(sizes), set(weights) - set(sizes)
    w = {k: float(v) for k, v in weights.items() if float(v) > 0}
    assert w, "all weights are zero"
    total = sum(w.values())
    exact = {k: n_rows * v / total for k, v in w.items()}
    counts = {k: int(x) for k, x in exact.items()}
    for k in sorted(w, key=lambda k: (-(exact[k] - counts[k]), k))[: n_rows - sum(counts.values())]:
        counts[k] += 1
    short = {k: (c, sizes[k]) for k, c in counts.items() if c > sizes[k]}
    if short:
        raise RuntimeError(f"pools too small for the mix (wanted, available): {short}")
    return counts


class SelfPlayTrainScenarioMix(Job):
    """The self-play GRPO training input (v2): ``n_rows`` scenarios from category pools (``{category:
    SelfPlayScenarioPool.out_dir}``) in the proportions ``weights`` (``mix_counts``), drawn without replacement
    (seeded). The RL source shuffles the rows once and draws them in order, so the row shares are the training shares;
    reweighting re-runs only this job. Rows as ``SelfPlayTrainScenarioSet``'s (ids ``sptr2:<category>:<name>``, Minimal
    line prepended, a seeded voice pair per row) plus each side's name, gender and country (``""`` when unknown) and
    ``grader_info`` (the pool's ``grader_info.json`` entry, else ``""``): facts the RL judge uses to check answers.

    ``n_rows`` is fixed at declaration, so ``rl_spec`` checks a run's self-play draws against it before the run
    exists (``rules.check_scenario_budget``)."""

    def __init__(self, *, pools: dict, weights: dict, voices: tk.Path, n_rows: int, seed: int = 0):
        assert set(weights) <= set(pools), set(weights) - set(pools)
        self.pools, self.weights, self.voices = dict(pools), dict(weights), voices
        self.n_rows, self.seed = int(n_rows), int(seed)
        self.out_dir = self.output_path("dataset", directory=True)
        self.out_stats = self.output_path("stats.json")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import numpy as np
        from datasets import Dataset, load_from_disk

        from .persona_prompts import PERSONA_MINIMAL

        scen, sides, grader = {}, {}, {}
        for cat in sorted(self.pools):
            d = self.pools[cat].get_path()
            scen[cat] = [tuple(s) for s in json.load(open(os.path.join(d, "scenarios.json")))]
            sides[cat] = json.load(open(os.path.join(d, "names.json")))
            g = os.path.join(d, "grader_info.json")
            grader[cat] = json.load(open(g)) if os.path.exists(g) else {}
            assert all(s[0] == cat for s in scen[cat]), f"pool {cat} holds another category"
        counts = mix_counts({k: len(v) for k, v in scen.items()}, self.weights, self.n_rows)
        voices = list(load_from_disk(self.voices.get_path()).select_columns(["label", "codes"]))
        rng = np.random.default_rng(self.seed)
        rows = []
        for cat in sorted(counts):
            for i in sorted(int(i) for i in rng.choice(len(scen[cat]), size=counts[cat], replace=False)):
                _, name, pa, pb, level = scen[cat][i]
                (na, ga, ca), (nb, gb, cb) = (s or ("", "", "") for s in sides[cat][name])
                ia, ib = (int(j) for j in rng.choice(len(voices), size=2, replace=False))
                rows.append(
                    {
                        "id": f"sptr2:{cat}:{name}",
                        "scenario": name,
                        "category": cat,
                        "level": level,
                        "context": [f"{PERSONA_MINIMAL} {pa}"],
                        "context_other": [f"{PERSONA_MINIMAL} {pb}"],
                        "voice_codes": np.asarray(voices[ia]["codes"], np.int16),
                        "voice_codes_other": np.asarray(voices[ib]["codes"], np.int16),
                        "voice_label": str(voices[ia]["label"]),
                        "voice_label_other": str(voices[ib]["label"]),
                        "name": na, "gender": ga, "country": ca,
                        "name_other": nb, "gender_other": gb, "country_other": cb,
                        "grader_info": str(grader[cat].get(name, "")),
                    }
                )  # fmt: skip
        assert len({r["id"] for r in rows}) == len(rows), "duplicate scenario ids"
        Dataset.from_list(rows).save_to_disk(self.out_dir.get_path())
        stats = {"rows": len(rows), "per_category": counts, "weights": self.weights, "voices": len(voices)}
        json.dump(stats, open(self.out_stats.get_path(), "w"), indent=1)
        print(f"[sp-train-mix] {stats}", flush=True)
