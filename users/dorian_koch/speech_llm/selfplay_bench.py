"""Self-play judge v2 (``spj2``): an LLM-graded self-play BENCHMARK on prompts and voices no model trained on.

User, 2026-09-28: "Can we do some llm grading on the self play dialogues and also use that as benchmark? Ofc only use
prompts that were not seen in training. And have the llm grade prompt adherence, coherence, and how well the llms
respond to each other. Maybe also adjust our prompts to make it easier to grade for this benchmark. [...] maybe it
would be good to have multiple generations for the same prompt tho, because i think its really random."

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

from .common import map_concurrent, vllm_gpu_mem_gb, vllm_server
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


def check_scenarios(scenarios: list) -> None:
    """Names unique, every category used, each side's prompt short enough for the self-play prompt budget."""
    names = [s[1] for s in scenarios]
    assert len(set(names)) == len(names), "scenario names must be unique"
    assert {s[0] for s in scenarios} <= set(SPJ2_CATEGORIES), {s[0] for s in scenarios}
    for cat, name, pa, pb in scenarios:
        for p in (pa, pb):
            # the worker's KV context is conversation + 256 frames; voice 50 + 12 silence + the text tokens
            assert 8 <= len(p.split()) <= 45, (name, len(p.split()))


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
        env["PYTHONPATH"] = _moshi_pythonpath() + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
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


def judge_rubric_digest(name: str) -> str:
    return hashlib.sha256(json.dumps(SPJ2_RUBRICS[name]).encode()).hexdigest()[:16]


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
    ):
        assert rubric in SPJ2_RUBRICS, rubric
        assert runs
        self.runs = {str(k): v for k, v in sorted(runs.items())}
        self.rubric = rubric
        self.rubric_digest = judge_rubric_digest(rubric)  # in the hash via hash() below
        self.llm_name = llm_name
        self.temperature = float(temperature)
        self.out_rows = self.output_path("rows.jsonl")
        self.out_summary = self.output_path("summary.json")
        self.rqmt = {"gpu": 1, "cpu": 4, "mem": 32, "time": 2, "gpu_mem_gb": vllm_gpu_mem_gb(llm_name, 80)}

    @classmethod
    def hash(cls, parsed_args):
        """The rubric TEXT is part of the hash (see ``SelfPlayScenarioSet.hash``)."""
        return super(SelfPlayJudgeV2, cls).hash(
            {**parsed_args, "rubric_digest": judge_rubric_digest(parsed_args["rubric"])}
        )

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def completed_fraction(self):
        from sisyphus import global_settings as gs

        try:
            d = json.load(open(os.path.join(self._sis_path(gs.JOB_WORK_DIR), "progress.json")))
            return min(d["done"] / d["total"], 1.0)
        except (OSError, ValueError, KeyError, ZeroDivisionError):
            return None

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
        print(f"[spj2-judge] {len(items)} conversations over {sorted({it['label'] for it in items})}", flush=True)

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
                        extra_body={"guided_json": SPJ2_JSON_SCHEMA},
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
        summary.update({"rubric": self.rubric, "rubric_digest": self.rubric_digest, "llm_name": llm_name})
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

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        rows = {}
        for label, path in self.judges.items():
            with open(path.get_path()) as f:
                rows[label] = [r for r in map(json.loads, f) if r["label"] == label]
            if not rows[label]:
                raise RuntimeError(f"{path}: no rows labelled {label}")
        table = {"reference": self.reference, "models": {}}
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
