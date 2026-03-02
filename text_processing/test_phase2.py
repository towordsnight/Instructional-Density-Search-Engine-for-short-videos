#!/usr/bin/env python3
"""
Phase 2 Test Suite: Transcript cleaning pipeline.

Run: python -m pytest text_processing/test_phase2.py -v
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from text_processing.clean_transcript import (
    STEPS,
    DEFAULT_STEP_ORDER,
    clean_transcript,
    clean_dataset,
    _normalize_unicode,
    _remove_caption_artifacts,
    _remove_timestamps,
    _remove_urls,
    _remove_mentions_hashtags,
    _remove_fillers,
    _collapse_repeated_words,
    _normalize_whitespace,
    _segment_sentences,
)


# ===================================================================
# TestCleanTranscript — 12 tests
# ===================================================================

class TestCleanTranscript(unittest.TestCase):
    """Tests for the main clean_transcript() public API."""

    def test_none_input(self):
        self.assertEqual(clean_transcript(None), "")

    def test_empty_string(self):
        self.assertEqual(clean_transcript(""), "")

    def test_whitespace_only(self):
        self.assertEqual(clean_transcript("   \n\t  "), "")

    def test_clean_text_passthrough(self):
        """Already clean text should survive the pipeline mostly intact."""
        text = "Learn how to design jewelry. First, choose your materials."
        result = clean_transcript(text)
        self.assertIn("Learn", result)
        self.assertIn("design", result)
        self.assertIn("jewelry", result)

    def test_preserves_instructional_keywords(self):
        """Instructional keywords (from instructional_score.py) must survive cleaning."""
        keywords = [
            "tutorial", "guide", "learn", "design", "meaning",
            "significance", "history", "heritage", "step", "create",
        ]
        text = " ".join(keywords)
        result = clean_transcript(text)
        for kw in keywords:
            self.assertIn(kw, result, f"Keyword '{kw}' was removed by cleaning")

    def test_removes_music_tag(self):
        result = clean_transcript("Hello [Music] world")
        self.assertNotIn("[Music]", result)
        self.assertIn("Hello", result)
        self.assertIn("world", result)

    def test_removes_fillers(self):
        result = clean_transcript("So uh you need um to add the ingredients")
        self.assertNotIn(" uh ", result)
        self.assertNotIn(" um ", result)
        self.assertIn("add", result)
        self.assertIn("ingredients", result)

    def test_removes_timestamps(self):
        result = clean_transcript("At 0:15 we start the tutorial")
        self.assertNotIn("0:15", result)
        self.assertIn("tutorial", result)

    def test_removes_urls(self):
        result = clean_transcript("Check out https://example.com for more info")
        self.assertNotIn("https://", result)
        self.assertIn("Check out", result)

    def test_removes_repeated_words(self):
        result = clean_transcript("the the recipe is is great")
        self.assertNotIn("the the", result)
        self.assertNotIn("is is", result)

    def test_custom_steps(self):
        """Only the specified steps should run."""
        text = "[Music] uh hello https://x.com"
        # Only remove URLs — other artifacts should remain
        result = clean_transcript(text, steps=["remove_urls"])
        self.assertIn("[Music]", result)
        self.assertIn("uh", result)
        self.assertNotIn("https://", result)

    def test_invalid_step_raises(self):
        with self.assertRaises(ValueError):
            clean_transcript("hello", steps=["nonexistent_step"])


# ===================================================================
# TestUnicodeNormalization — 4 tests
# ===================================================================

class TestUnicodeNormalization(unittest.TestCase):
    """Tests for _normalize_unicode step."""

    def test_nfc_normalization(self):
        # e + combining acute (NFD) -> e-acute (NFC)
        nfd = "caf\u0065\u0301"  # "café" in NFD
        result = _normalize_unicode(nfd)
        self.assertIn("\u00e9", result)  # NFC form

    def test_removes_zero_width_chars(self):
        text = "hello\u200bworld\u200c!"
        result = _normalize_unicode(text)
        self.assertEqual(result, "helloworld!")

    def test_removes_bom(self):
        text = "\ufeffHello world"
        result = _normalize_unicode(text)
        self.assertEqual(result, "Hello world")

    def test_replaces_nonbreaking_spaces(self):
        text = "hello\u00a0world"
        result = _normalize_unicode(text)
        self.assertEqual(result, "hello world")


# ===================================================================
# TestCaptionArtifacts — 4 tests
# ===================================================================

class TestCaptionArtifacts(unittest.TestCase):
    """Tests for _remove_caption_artifacts step."""

    def test_removes_all_known_tags(self):
        tags = ["[Music]", "[Applause]", "[Laughter]", "[Inaudible]", "[Foreign]"]
        for tag in tags:
            result = _remove_caption_artifacts(f"before {tag} after")
            self.assertNotIn(tag, result, f"Tag {tag} was not removed")

    def test_case_insensitive(self):
        for variant in ["[MUSIC]", "[music]", "[Music]"]:
            result = _remove_caption_artifacts(variant)
            self.assertEqual(result.strip(), "")

    def test_preserves_non_tag_brackets(self):
        text = "This is [important] information"
        result = _remove_caption_artifacts(text)
        self.assertIn("[important]", result)

    def test_multiple_tags_in_sequence(self):
        text = "[Music] Welcome [Applause] to the show [Laughter]"
        result = _remove_caption_artifacts(text)
        self.assertNotIn("[Music]", result)
        self.assertNotIn("[Applause]", result)
        self.assertNotIn("[Laughter]", result)
        self.assertIn("Welcome", result)
        self.assertIn("show", result)


# ===================================================================
# TestFillerRemoval — 4 tests
# ===================================================================

class TestFillerRemoval(unittest.TestCase):
    """Tests for _remove_fillers step."""

    def test_removes_single_fillers(self):
        text = "So uh we need um to er add ah the hmm stuff"
        result = _remove_fillers(text)
        for filler in ["uh", "um", "er", "ah", "hmm"]:
            self.assertNotRegex(result, rf"\b{filler}\b")

    def test_preserves_like_as_verb(self):
        """'like' should NOT be removed — it's too ambiguous."""
        text = "I really like this design"
        result = _remove_fillers(text)
        self.assertIn("like", result)

    def test_removes_multi_word_fillers(self):
        text = "The meaning you know is really i mean significant"
        result = _remove_fillers(text)
        self.assertNotIn("you know", result.lower())
        self.assertNotIn("i mean", result.lower())
        self.assertIn("meaning", result)
        self.assertIn("significant", result)

    def test_case_insensitive_fillers(self):
        text = "UH okay UM sure"
        result = _remove_fillers(text)
        self.assertNotRegex(result, r"\bUH\b")
        self.assertNotRegex(result, r"\bUM\b")


# ===================================================================
# TestRepeatedWords — 3 tests
# ===================================================================

class TestRepeatedWords(unittest.TestCase):
    """Tests for _collapse_repeated_words step."""

    def test_double_word_collapse(self):
        result = _collapse_repeated_words("the the cat")
        self.assertEqual(result, "the cat")

    def test_triple_word_collapse(self):
        result = _collapse_repeated_words("really really really good")
        self.assertEqual(result, "really good")

    def test_no_false_positives(self):
        """Different words that look similar should not be collapsed."""
        text = "that this then there"
        result = _collapse_repeated_words(text)
        self.assertEqual(result, text)


# ===================================================================
# TestSentenceSegmentation — 3 tests
# ===================================================================

class TestSentenceSegmentation(unittest.TestCase):
    """Tests for _segment_sentences step."""

    def test_preserves_existing_punctuation(self):
        text = "First, add water. Then, stir well."
        result = _segment_sentences(text)
        # Already has periods — no extra periods should be added
        self.assertEqual(text, result)

    def test_segments_run_on_text(self):
        text = "we add the water Then we stir it"
        result = _segment_sentences(text)
        self.assertIn(". Then", result)

    def test_multiple_markers(self):
        text = "we start here Next we add salt Finally we serve"
        result = _segment_sentences(text)
        self.assertIn(". Next", result)
        self.assertIn(". Finally", result)


# ===================================================================
# TestCleanDataset — 5 tests
# ===================================================================

class TestCleanDataset(unittest.TestCase):
    """Tests for clean_dataset() batch processing."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.input_path = os.path.join(self.tmpdir, "input.json")
        self.output_path = os.path.join(self.tmpdir, "output.json")

    def _write_input(self, data):
        with open(self.input_path, "w") as f:
            json.dump(data, f)

    def test_batch_cleans_transcripts(self):
        data = [
            {"id": "1", "title": "T1", "transcript": "[Music] uh hello world"},
            {"id": "2", "title": "T2", "transcript": "clean text already"},
        ]
        self._write_input(data)
        stats = clean_dataset(self.input_path, self.output_path)
        self.assertEqual(stats["cleaned"], 2)

        with open(self.output_path) as f:
            result = json.load(f)
        self.assertNotIn("[Music]", result[0]["transcript"])

    def test_schema_preservation(self):
        """All non-transcript fields should survive unchanged."""
        data = [
            {"id": "x", "platform": "youtube", "title": "T", "transcript": "[Music] hi",
             "views": 100, "url": "https://yt.com/x"},
        ]
        self._write_input(data)
        clean_dataset(self.input_path, self.output_path)

        with open(self.output_path) as f:
            result = json.load(f)
        self.assertEqual(result[0]["id"], "x")
        self.assertEqual(result[0]["platform"], "youtube")
        self.assertEqual(result[0]["views"], 100)
        self.assertEqual(result[0]["url"], "https://yt.com/x")

    def test_null_transcript_handling(self):
        data = [
            {"id": "1", "title": "T1", "transcript": None},
            {"id": "2", "title": "T2"},  # missing transcript key
        ]
        self._write_input(data)
        stats = clean_dataset(self.input_path, self.output_path)
        self.assertEqual(stats["skipped"], 2)
        self.assertEqual(stats["cleaned"], 0)

    def test_empty_dataset(self):
        self._write_input([])
        stats = clean_dataset(self.input_path, self.output_path)
        self.assertEqual(stats["total"], 0)

    def test_summary_stats(self):
        data = [
            {"id": "1", "transcript": "hello [Music] world"},
            {"id": "2", "transcript": None},
            {"id": "3", "transcript": "  "},
        ]
        self._write_input(data)
        stats = clean_dataset(self.input_path, self.output_path)
        self.assertEqual(stats["total"], 3)
        self.assertEqual(stats["cleaned"], 1)
        self.assertEqual(stats["skipped"], 2)


# ===================================================================
# TestRankingPreservation — 1 integration test
# ===================================================================

class TestRankingPreservation(unittest.TestCase):
    """Verify that cleaning preserves instructional density ordering."""

    def test_instructional_density_preserved_after_cleaning(self):
        """After cleaning TEST_DATA transcripts, instructional density must
        still be higher for instructional videos than for entertainment vlogs."""
        from test_ranking import TEST_DATA
        from instructional_score import compute_instructional_score

        inst_ids = {"inst_t", "inst_knot", "inst_lock"}
        vlog_ids = {"vlog_1", "vlog_2"}

        inst_scores = []
        vlog_scores = []

        for item in TEST_DATA:
            cleaned = clean_transcript(item["transcript"])
            combined = f"{item['title']} [SEP] {cleaned}"
            score = compute_instructional_score(combined)

            if item["id"] in inst_ids:
                inst_scores.append(score)
            elif item["id"] in vlog_ids:
                vlog_scores.append(score)

        min_inst = min(inst_scores)
        max_vlog = max(vlog_scores)
        self.assertGreater(
            min_inst, max_vlog,
            f"Instructional min ({min_inst}) should exceed vlog max ({max_vlog}) after cleaning"
        )


# ===================================================================
# Pipeline completeness checks
# ===================================================================

class TestPipelineCompleteness(unittest.TestCase):
    """Verify pipeline registry and step ordering are consistent."""

    def test_all_default_steps_exist_in_registry(self):
        for step in DEFAULT_STEP_ORDER:
            self.assertIn(step, STEPS, f"Step '{step}' in DEFAULT_STEP_ORDER but not in STEPS")

    def test_registry_matches_default_order(self):
        self.assertEqual(set(DEFAULT_STEP_ORDER), set(STEPS.keys()))

    def test_step_count(self):
        self.assertEqual(len(DEFAULT_STEP_ORDER), 9)


if __name__ == "__main__":
    unittest.main()
