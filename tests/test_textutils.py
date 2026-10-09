"""Text utilities: tokenisation, concept lexicon, IDF-weighted lexical scoring."""

from __future__ import annotations

from pipeline.textutils import (
    concepts_in,
    dedupe_fingerprint,
    lexical_score,
    normalize_for_match,
    normalize_title,
    search_units,
    split_sentences,
    tokenize,
)


class TestTokenize:
    def test_latin_words_pass_through_whole(self) -> None:
        assert tokenize("copper inventories fall") == ["copper", "inventories", "fall"]

    def test_cjk_expands_to_unigrams_and_bigrams(self) -> None:
        tokens = tokenize("碳酸锂")
        # Unigrams so a one-character query (锂) can match, bigrams so 碳酸 is a unit.
        assert {"碳", "酸", "锂", "碳酸", "酸锂"} <= set(tokens)

    def test_single_cjk_character_survives_the_length_filter(self) -> None:
        assert "锂" in tokenize("锂价")

    def test_latin_stopwords_are_dropped_but_cjk_is_not(self) -> None:
        assert "the" not in tokenize("the price of copper")
        assert "的" not in tokenize("铜的价格") or "价" in tokenize("铜的价格")


class TestSearchUnits:
    def test_maximal_cjk_run_is_one_unit_not_n_grams(self) -> None:
        # The whole point: a CJK run stays whole, not split into overlapping n-grams,
        # because scoring on n-grams lets a document that shares three of nine
        # fragments look a third relevant. The concept marker is appended separately.
        units = search_units("印度尼西亚")
        assert "印度尼西亚" in units
        assert "印" not in units
        assert "印尼" not in units

    def test_latin_words_are_units(self) -> None:
        units = search_units("copper inventories")
        assert "copper" in units
        assert "inventories" in units

    def test_concept_markers_are_appended(self) -> None:
        units = search_units("印尼镍矿配额")
        assert "§indonesia" in units
        assert "§nickel" in units
        assert "§quota" in units

    def test_mixed_script_splits_on_the_script_boundary(self) -> None:
        units = search_units("2027 年镍产量")
        assert "2027" in units
        assert "年镍产量" in units


class TestConceptLexicon:
    def test_chinese_and_english_forms_map_to_one_concept(self) -> None:
        assert "rare_earth" in concepts_in("稀土")
        assert "rare_earth" in concepts_in("rare earth elements")

    def test_ascii_forms_match_on_word_boundaries(self) -> None:
        # "list" must not fire inside "listing", or every catalogue document matches.
        assert "catalogue" in concepts_in("the critical minerals list")
        assert "catalogue" not in concepts_in("listing prices rose")

    def test_cjk_forms_match_as_substrings(self) -> None:
        assert "quota" in concepts_in("开采总量控制指标")

    def test_direction_words_are_not_mistaken_for_genre(self) -> None:
        # 跌 is a direction, not a request for the price series.
        assert "fall" in concepts_in("锌加工费跌至多年低点")

    def test_order_is_lexicon_order_not_occurrence_order(self) -> None:
        # Stable order keeps a document's embedding independent of word order.
        assert concepts_in("镍和铜") == concepts_in("铜和镍")

    def test_alumina_is_alumina_not_aluminium(self) -> None:
        concepts = concepts_in("氧化铝价格")
        assert "alumina" in concepts
        assert "aluminium" not in concepts


class TestLexicalScore:
    def test_identical_text_outscores_a_partial_overlap(self) -> None:
        text = "Indonesia trims nickel ore quota for 2027"
        partial = "Indonesia trims nickel"
        assert lexical_score(text, text) > lexical_score(text, partial)
        # Not 1.0: the density term saturates at ``full_tf`` occurrences, so a term
        # mentioned once contributes a third of the density half. That is intended --
        # the score is a blend, not a probability.
        assert lexical_score(text, text) >= 0.8

    def test_unrelated_text_scores_zero(self) -> None:
        assert lexical_score("nickel quota", "zinc treatment charges") == 0.0

    def test_concepts_bridge_languages(self) -> None:
        # A Chinese question against the English answer must out-score a document
        # that merely shares the mineral.
        question = "印度尼西亚 2027 年镍矿配额调整了多少？"
        anchor = "Indonesia trims nickel ore quota for 2027"
        other = "关于调整镍管理政策的公告（2026-262）"
        assert lexical_score(question, anchor) > lexical_score(question, other)

    def test_idf_makes_a_rare_term_decisive(self) -> None:
        query = "印尼 镍矿"
        df = {"§indonesia": 1, "§nickel": 300}
        n_docs = 764
        rare = lexical_score(query, "Indonesia nickel", df=df, n_docs=n_docs)
        common = lexical_score(query, "nickel only", df=df, n_docs=n_docs)
        assert rare > common

    def test_cross_lingual_match_goes_through_the_concept_lexicon(self) -> None:
        # 镍 does not literally occur in "nickel" -- the match exists only because
        # both sides carry the §nickel concept marker. That is the entire mechanism
        # by which a Chinese question retrieves an English document.
        assert lexical_score("镍", "nickel") > 0.0
        assert lexical_score("镍", "copper") == 0.0

    def test_same_language_literal_match_outweighs_a_bare_concept(self) -> None:
        assert lexical_score("镍", "镍矿") > lexical_score("镍", "nickel")


class TestDedupeHelpers:
    def test_fingerprint_ignores_formatting_not_content(self) -> None:
        a = dedupe_fingerprint("Copper  prices   ROSE!")
        b = dedupe_fingerprint("copper prices rose")
        assert a == b

    def test_fingerprint_distinguishes_a_rewritten_sentence(self) -> None:
        a = dedupe_fingerprint("Lithium destocking nears its end as cathode rates rise")
        b = dedupe_fingerprint("Lithium restocking begins as cathode rates fall")
        assert a != b

    def test_normalize_title_strips_publisher_suffix(self) -> None:
        assert normalize_title("Zinc TC collapse - mining.com") == "zinc tc collapse"

    def test_normalize_for_match_keeps_cjk(self) -> None:
        # The whole reason normalisation is Unicode-native: an ASCII-strip would
        # delete the Chinese half of the corpus.
        assert normalize_for_match("稀土出口管制") == "稀土出口管制"


class TestSplitSentences:
    def test_splits_on_cjk_terminators_without_trailing_space(self) -> None:
        assert split_sentences("第一句。第二句。") == ["第一句。", "第二句。"]

    def test_does_not_split_on_a_decimal_point(self) -> None:
        # "0.4%" has no space after the dot, so it stays inside its sentence.
        sentences = split_sentences("Grade rose to 0.4% this week. Demand held.")
        assert sentences == ["Grade rose to 0.4% this week.", "Demand held."]

    def test_abbreviations_do_split_and_that_is_accepted(self) -> None:
        # Documented limitation: "Fig." ends a segment because the rule is
        # whitespace-based and there is no abbreviation list. The four-character
        # fragment is dropped by the answerer's minimum-sentence length, so it never
        # reaches a user.
        sentences = split_sentences("Grade rose. Fig. 3 shows it.")
        assert "Fig." in sentences

    def test_blank_line_is_a_boundary(self) -> None:
        assert len(split_sentences("heading\n\nbody text")) == 2
