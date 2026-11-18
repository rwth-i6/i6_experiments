"""Name pools for the SFT dialogue writer: an LLM asked to invent a name picks the same few (pilot 2026-11-17, Qwen3.8:
"Sarah" in 8 of 30 English dialogues, "Lena" in 6 of 32 German, "Анна" in 8 of 33 Russian, "王强" in 5 of 32
Chinese). So the seed draws every person's name, by gender, and the voice is drawn from the same gender.

Given names are common names of adults of mixed ages in each language area (lists ordered roughly by frequency; the
draw weight falls off slowly with rank so the head does not dominate); family names likewise. Russian family names
take the feminine form for women (``_ru_feminine``); Chinese names are family name + one or two given characters.
"""

from __future__ import annotations

import random

GIVEN = {
    "en": {
        "female": """Mary Jennifer Linda Patricia Elizabeth Susan Jessica Karen Lisa Nancy Emily Michelle Amanda Melissa
        Stephanie Rebecca Laura Rachel Hannah Megan Olivia Emma Sophie Charlotte Grace Chloe Lucy Amelia Ava Isabella
        Mia Abigail Madison Natalie Samantha Victoria Katherine Danielle Heather Nicole Christina Kelly Angela Brenda
        Joanne Siobhan Aoife Niamh Fiona Claire Helen Margaret Janet Diane Ruth Joyce Gloria Teresa Priya Aisha Mei
        Maria Carmen Rosa Fatima Keisha Tamika Jasmine Zoe Harper Ellie Freya Imogen Poppy Bethany Kirsty Leanne
        Shauna Tara Erin Molly Ruby Isla Maya Layla Nora Hazel Stella Paige Tiffany Courtney""".split(),
        "male": """James John Robert Michael William David Richard Joseph Thomas Charles Christopher Daniel Matthew
        Anthony Mark Steven Paul Andrew Joshua Kevin Brian George Edward Ryan Jacob Nicholas Jonathan Justin Brandon
        Samuel Benjamin Nathan Patrick Sean Liam Noah Oliver Jack Harry Oscar Ethan Lucas Mason Logan Owen Connor
        Callum Declan Ciaran Eoin Rhys Gareth Ian Graham Neil Stuart Colin Alan Gary Wayne Darren Craig Raj Arjun Omar
        Ahmed Wei Kenji Carlos Luis Jose Miguel Marcus Darnell Andre Tyrone Malik Isaiah Elijah Caleb Hunter Wyatt
        Dylan Tyler Aaron Adam Scott Jeffrey Gregory Dennis Frank Raymond Peter Henry Arthur Alfie Theo Finn""".split(),
    },
    "de": {
        "female": """Sabine Petra Claudia Andrea Susanne Nicole Stefanie Julia Katharina Anna Sarah Laura Lisa Lea Lena
        Hannah Marie Sophie Johanna Emma Mia Clara Charlotte Paula Greta Ida Frieda Mila Leonie Lara Jana Nina Franziska
        Christina Melanie Sandra Tanja Anja Birgit Monika Ursula Renate Gabriele Ingrid Karin Heike Silke Kerstin
        Martina Doris Elke Brigitte Daniela Simone Jasmin Vanessa Jessica Annika Miriam Carina Theresa Magdalena
        Veronika Elisabeth Barbara Bettina Antje Ulrike Dagmar Ines Svenja Merle Wiebke Maren Tabea Helena Luisa
        Emilia Amelie Marlene Selin Aylin Elif Natascha Irina Olga Katarzyna Agnieszka Vesna Dragana Corinna
        Regina Angelika Rosemarie Hildegard""".split(),
        "male": """Michael Thomas Andreas Stefan Christian Markus Martin Frank Peter Klaus Jürgen Wolfgang Uwe Bernd
        Dieter Ralf Jörg Matthias Alexander Sebastian Daniel Tobias Florian Philipp Jan Tim Lukas Felix Jonas Leon
        Paul Noah Elias Finn Ben Luis Maximilian Moritz Niklas Julian Simon David Fabian Dominik Patrick Marcel Sven
        Torsten Carsten Holger Dirk Olaf Rainer Heinz Günter Manfred Horst Helmut Gerhard Werner Hans Karl Josef
        Johannes Benedikt Korbinian Quirin Lorenz Valentin Konstantin Vincent Anton Emil Theo Oskar Henri Mats
        Malte Hauke Jannik Kevin Dennis Mehmet Emre Can Murat Ali Dimitri Sergej Piotr Goran Mario Marco Luca
        Fabio Reto Urs Beat""".split(),
    },
    "ru": {
        "female": """Елена Ольга Наталья Татьяна Ирина Светлана Анна Мария Екатерина Юлия Анастасия Марина Людмила
        Галина Валентина Надежда Любовь Вера Нина Лариса Оксана Виктория Ксения Дарья Алина Полина Софья
        Вероника Елизавета Варвара Василиса Ульяна Арина Таисия Милана Ева Злата Кристина Алёна Яна Евгения Диана
        Маргарита Валерия Александра Антонина Зинаида Раиса Тамара Лидия Клавдия Жанна Инна Алла Римма Регина Эльвира
        Карина Камила Дина Лилия Гульнара Айгуль Олеся Ангелина Таня Маша Катя Настя Лена Оля Света Ира""".split(),
        "male": """Александр Сергей Дмитрий Андрей Алексей Иван Михаил Евгений Николай Владимир Максим Артём Павел
        Роман Денис Игорь Олег Виктор Юрий Константин Антон Илья Кирилл Никита Егор Тимофей Матвей Даниил Георгий
        Фёдор Степан Глеб Арсений Лев Марк Семён Ярослав Вадим Виталий Валерий Геннадий Анатолий Борис Василий
        Григорий Леонид Пётр Станислав Руслан Тимур Ринат Эдуард Аркадий Вячеслав Ефим Захар Савелий Богдан
        Всеволод Святослав Тарас Ростислав Елисей Платон Мирон Давид Эмиль Артур Марат Ильдар Азат Саша Миша Дима
        Серёжа Лёша Ваня Коля""".split(),
    },
    "zh": {
        "female": list("秀芳敏静丽燕艳娟霞玲婷雪琳慧颖洁晶倩璐瑶欣怡佳悦萌晨蕾薇琪妍菲莉岚"),
        "male": list("伟强磊军勇杰涛斌超明刚平辉鹏华飞鑫波宇浩凯健峰亮建林晨阳旭东博俊"),
    },
}
#: the second character of a two-character Chinese given name (gender-neutral)
ZH_SECOND = list("文佳子雨思嘉梓一欣晓天浩宇婷涵琪晨明华志海春秋红丽云玉")

FAMILY = {
    "en": """Smith Johnson Williams Brown Jones Garcia Miller Davis Rodriguez Martinez Wilson Anderson Taylor Thomas
    Moore Jackson Martin Lee Thompson White Harris Clark Lewis Robinson Walker Young Allen King Wright Scott Hill Green
    Adams Baker Nelson Carter Mitchell Roberts Turner Phillips Campbell Parker Evans Edwards Collins Stewart Morris
    Murphy Cook Rogers Morgan Cooper Peterson Reed Bailey Bell Kelly Howard Ward Cox Richardson Wood Watson Brooks
    Bennett Gray Hughes Price Sanders Patel Nguyen Kim Chen Singh Khan Ali Shah O'Brien O'Connor Byrne Ryan Walsh
    Doyle McCarthy Gallagher Murray Fraser Macdonald Robertson Hughes Davies Jenkins Lloyd Price Owen Palmer Fletcher
    Hart Webb Marshall Holmes Chapman Ellis""".split(),
    "de": """Müller Schmidt Schneider Fischer Weber Meyer Wagner Becker Schulz Hoffmann Schäfer Koch Bauer Richter
    Klein Wolf Schröder Neumann Schwarz Zimmermann Braun Krüger Hofmann Hartmann Lange Schmitt Werner Schmitz Krause
    Meier Lehmann Schmid Schulze Maier Köhler Herrmann König Walter Mayer Huber Kaiser Fuchs Peters Lang Scholz Möller
    Weiß Jung Hahn Schubert Vogel Friedrich Keller Günther Frank Berger Winkler Roth Beck Lorenz Baumann Franke Albrecht
    Schuster Simon Ludwig Böhm Winter Kraus Martin Schumacher Krämer Vogt Stein Jäger Otto Sommer Groß Seidel Heinrich
    Brandt Haas Schreiber Graf Dietrich Ziegler Kuhn Kühn Pohl Engel Horn Busch Bergmann Thomas Voigt Sauer Arnold
    Wolff Pfeiffer Yılmaz Kaya Demir Nowak Kowalski Gruber Steiner Moser Hofer Brunner Meili Frei""".split(),
    "ru": """Иванов Смирнов Кузнецов Попов Васильев Петров Соколов Михайлов Новиков Фёдоров Морозов Волков Алексеев
    Лебедев Семёнов Егоров Павлов Козлов Степанов Николаев Орлов Андреев Макаров Никитин Захаров Зайцев Соловьёв
    Борисов Яковлев Григорьев Романов Воробьёв Сергеев Кузьмин Фролов Александров Дмитриев Королёв Гусев Киселёв
    Ильин Максимов Поляков Сорокин Виноградов Ковалёв Белов Медведев Антонов Тарасов Жуков Баранов Филиппов Комаров
    Давыдов Беляев Герасимов Богданов Осипов Сидоров Матвеев Титов Марков Миронов Крылов Куликов Карпов Власов
    Мельников Денисов Гаврилов Тихонов Казаков Афанасьев Данилов Савельев Тимофеев Фомин Чернов Абрамов Мартынов
    Ефимов Федотов Щербаков Назаров Калинин Исаев Чернышёв Быков Маслов Родионов Коновалов Лазарев Воронин Климов
    Филатов Пономарёв Голубев Кудрявцев Прохоров Наумов Потапов Журавлёв Овчинников Трофимов Леонов Соболев Ермаков
    Колесников Гончаров Емельянов Никифоров Грачёв Котов Гришин Ефремов Архипов Громов Кириллов Малышев Панов
    Моисеев Румянцев Акимов Кондратьев Бирюков Горбунов Анисимов Ерёмин Тихомиров Галкин Лукьянов Михеев Скворцов
    Юдин Белоусов Нестеров Симонов Прокофьев Харитонов Князев Цветков Левин Митрофанов Воронов Аксёнов Софронов
    Мальцев Логинов Горшков Савин Краснов Майоров Демидов Елисеев Рыбаков Сафонов Плотников Демин Хохлов Жданов
    Островский Вишневский Ковальчук Шевченко Бондаренко Ткаченко Мельник Гарипов Хабибуллин Сафин""".split(),
    "zh": list("王李张刘陈杨黄赵吴周徐孙马朱胡郭何高林罗郑梁谢宋唐许韩冯邓曹彭曾肖田董袁潘于蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏韦付方白邹孟熊秦邱江尹薛闫段雷侯龙史陶黎贺顾毛郝龚邵万钱严覃武戴莫孔向汤"),
}


def _ranked(rng: random.Random, pool: list[str]) -> str:
    """A draw weighted ~ 1 / sqrt(rank): frequent names more often, the tail still common."""
    return rng.choices(pool, weights=[1.0 / (k + 1) ** 0.5 for k in range(len(pool))])[0]


def _ru_feminine(family: str) -> str:
    for a, b in (("ский", "ская"), ("цкий", "цкая"), ("ов", "ова"), ("ев", "ева"), ("ёв", "ёва"), ("ин", "ина")):
        if family.endswith(a):
            return family[: -len(a)] + b
    return family  # Шевченко, Мельник, ...


def draw_person(lang: str, gender: str, rng: random.Random, avoid: set[str] = frozenset()) -> dict:
    """``{"gender", "given", "family", "full"}`` for a person of ``gender`` in the ``lang`` area, ``given`` not in
    ``avoid``."""
    assert gender in ("female", "male"), gender
    for _ in range(50):
        if lang == "zh":
            given = rng.choice(GIVEN["zh"][gender]) + (rng.choice(ZH_SECOND) if rng.random() < 0.7 else "")
            if rng.random() < 0.5 and len(given) == 2:
                given = given[::-1]
        else:
            given = _ranked(rng, GIVEN[lang][gender])
        if given not in avoid:
            break
    family = _ranked(rng, FAMILY[lang])
    if lang == "ru" and gender == "female":
        family = _ru_feminine(family)
    full = family + given if lang == "zh" else f"{given} {family}"
    return {"gender": gender, "given": given, "family": family, "full": full}


def draw_cast(lang: str, rng: random.Random, n_extra: int = 2) -> dict:
    """The people of one dialogue: ``user`` and ``assistant`` (each with a gender drawn 50/50; the voice draw uses
    it) and ``n_extra`` others for anyone the conversation mentions, all with different given names."""
    used: set[str] = set()
    cast = {}
    for role in ["user", "assistant"] + [f"other{k}" for k in range(n_extra)]:
        p = draw_person(lang, rng.choice(["female", "male"]), rng, used)
        used.add(p["given"])
        cast[role] = p
    return cast
