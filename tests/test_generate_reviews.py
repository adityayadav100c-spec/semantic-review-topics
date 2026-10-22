import unittest

from generate_reviews import THEMES, generate_rows


class SyntheticReviewTests(unittest.TestCase):
    def test_generates_balanced_deterministic_examples(self):
        first = generate_rows(3, seed=21)
        second = generate_rows(3, seed=21)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(THEMES) * 3)
        self.assertEqual({row["theme"] for row in first}, set(THEMES))
        self.assertTrue(all(row["is_synthetic"] == "true" for row in first))
        self.assertTrue(all(row["review"].strip() for row in first))

    def test_rejects_empty_topic_count(self):
        with self.assertRaises(ValueError):
            generate_rows(0, seed=42)


if __name__ == "__main__":
    unittest.main()
