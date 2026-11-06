"""Document split and token-prefix identities without network or tokenizer deps."""

import tempfile
import unittest
from pathlib import Path

from cst_research.experiments.ffn_data import split_stories, story_digest, write_prefix


class FakeEncoding:
    eot_token = 255

    @staticmethod
    def encode_ordinary(text):
        return list(text.encode())


class FFNDataTests(unittest.TestCase):
    def test_story_split_precedes_packing_and_handles_line_boundaries(self):
        self.assertEqual(
            list(
                split_stories(["one\n", "two<|endoftext|>three<|endoftext|>\n", "four"])
            ),
            ["one\ntwo", "three", "four"],
        )

    def test_prefix_is_exact_and_rejects_cross_split_and_internal_duplicates(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "tokens.bin"
            result, ids = write_prefix(
                iter(["leak", "ab", "ab", "cd"]),
                p,
                5,
                FakeEncoding(),
                forbidden={story_digest("leak")},
            )
            self.assertEqual(result["duplicates_excluded"], 2)
            self.assertEqual(result["tokens"], 5)
            self.assertEqual(len(p.read_bytes()), 20)
            self.assertEqual(ids, {story_digest("ab"), story_digest("cd")})
            self.assertNotIn(story_digest("leak"), ids)
