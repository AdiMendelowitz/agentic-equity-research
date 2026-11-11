from aer.utils.python_replacements import (
    classify_sector,
    extract_company_from_title,
    extract_worker_count,
    has_minimum_signal,
    normalize_company_name,
)


def test_normalize_strips_inc_suffix():
    assert normalize_company_name("Alphabet Inc.") == "alphabet"


def test_normalize_handles_punctuation():
    assert normalize_company_name("AT&T") == "att"


def test_normalize_strips_group_suffix():
    assert normalize_company_name("Goldman Sachs Group") == "goldman sachs"


def test_normalize_strips_whitespace():
    assert normalize_company_name("  Apple  ") == "apple"


def test_normalize_is_idempotent():
    first = normalize_company_name("Microsoft Corporation")
    assert normalize_company_name(first) == first


def test_classify_sector_tech_keyword():
    assert classify_sector("CloudTech Inc", None) == "tech"


def test_classify_sector_finance_keyword():
    assert classify_sector("Global Capital Bank", None) == "finance"


def test_classify_sector_whole_word_matching_ai_not_in_said():
    assert classify_sector("Company said it was fine", None) != "ai"


def test_classify_sector_default_when_no_match():
    assert classify_sector("Unknown Corp XYZ", None) == "tech"


def test_classify_sector_uses_raw_title():
    assert classify_sector("Acme Inc", "Acme Inc lays off 300 biotech researchers") == "healthcare"


def test_extract_company_simple():
    assert extract_company_from_title("Google lays off 1,000 employees") == "Google"


def test_extract_company_two_words():
    assert extract_company_from_title("Goldman Sachs cuts 3,000 jobs") == "Goldman Sachs"


def test_extract_company_with_breaking_prefix():
    assert extract_company_from_title("Breaking: Apple reduces workforce by 10%") == "Apple"


def test_extract_company_fallback_on_lowercase():
    result = extract_company_from_title("some lowercase title with no match")
    assert result == "some lowercase title with no"


def test_extract_worker_count_simple():
    assert extract_worker_count("laying off 500 employees") == 500


def test_extract_worker_count_comma_formatted():
    assert extract_worker_count("cutting 1,200 workers") == 1200


def test_extract_worker_count_returns_none_on_no_match():
    assert extract_worker_count("no numbers here") is None


def test_extract_worker_count_positions_keyword():
    assert extract_worker_count("eliminating 800 positions") == 800


class _Stub:
    def __init__(self, workers=None, reason=None):
        self.workers_affected = workers
        self.reason = reason


def test_has_minimum_signal_true_when_workers_present():
    assert has_minimum_signal([_Stub(workers=500)]) is True


def test_has_minimum_signal_true_when_reason_present():
    assert has_minimum_signal([_Stub(reason="cost reduction")]) is True


def test_has_minimum_signal_false_when_all_none():
    assert has_minimum_signal([_Stub(), _Stub()]) is False


def test_has_minimum_signal_false_for_empty_list():
    assert has_minimum_signal([]) is False
