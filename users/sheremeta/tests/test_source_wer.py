from i6_experiments.users.sheremeta.recognition.source_wer import assign_source, parse_dtl, split_by_source
from i6_experiments.users.sheremeta.returnn.metrics import format_metrics_markdown

# the seq tag patterns of the corpora Loquacious dev is assembled from
_SOURCES = {
    "voxpopuli": r"(?i)plenary",
    "yodas": r"\.wav",
    "common_voice": r"^common_voice",
    "librispeech": r"^\d+-\d+-\d+",
}


_DTL = (
    "Percent Total Error       =    7.5%   (150)\n"
    "Percent Correct           =   93.0%   (1860)\n"
    "Percent Substitution      =    4.0%   (80)\n"
    "Percent Deletions         =    3.0%   (60)\n"
    "Percent Insertions        =    0.5%   (10)\n"
    "Percent Word Accuracy     =   92.5%\n"
    "Ref. words                =           (2000)\n"
    "Hyp. words                =           (1950)\n"
    "Aligned words             =           (2010)\n"
)


def test_assign_source_first_match_wins_and_other():
    cases = {
        "20090204-0900-PLENARY-3-en_20090204-09:07:32_17": "voxpopuli",
        "xXADsgcd-2c-00223-00176452-00178443.wav": "yodas",
        "common_voice_en_49282": "common_voice",
        "7601-291468-0006": "librispeech",
        "peoples_speech_abc": "other",
    }
    for tag, source in cases.items():
        assert assign_source(tag, _SOURCES) == source, tag


def test_split_by_source_keeps_comments_out_and_groups_lines():
    lines = [
        ";; header\n",
        "common_voice_en_1 1 0.00 1.00 HELLO\n",
        "a.wav 1 0.00 1.00 WORLD\n",
        "common_voice_en_2 1 0.00 1.00 AGAIN\n",
    ]
    split = split_by_source(lines, _SOURCES)
    assert set(split) == {"common_voice", "yodas"}, split
    assert split["common_voice"] == [lines[1], lines[3]], split
    assert split["yodas"] == [lines[2]], split


def test_parse_dtl_recomputes_percentages_from_counts():
    stats = parse_dtl(_DTL, precision_ndigit=2)
    assert stats == {
        "wer": 7.5,
        "sub": 4.0,
        "del": 3.0,
        "ins": 0.5,
        "num_errors": 150,
        "ref_words": 2000,
    }, stats


def test_format_markdown_wer_by_source_table():
    report = format_metrics_markdown(
        {
            "name": "m",
            "wer": 7.14,
            "wer_label": "ep100",
            "wer_all": {"ep100": 7.14, "ep100_beam4": 7.2},
            "wer_by_source": {"ep100_beam4": {"yodas": 20.1, "common_voice": 5.5}, "ep100": {"yodas": 19.0}},
        }
    )
    assert "## WER by source" in report, report
    assert "| Label | common_voice | yodas |" in report, report
    assert "| ep100 |  | 19.0 |" in report, report
    assert report.index("| ep100 |") < report.index("| ep100_beam4 |"), report
