import unittest
from word_to_caption import map_words_to_captions

class TestMapWordsToCaptions(unittest.TestCase):
    def test_basic_functionality(self):
        captions = ["This is a test caption", "Another example of a caption", "The quick brown fox jumps over the lazy dog on a sunny day."]
        expected_output = {'test': [0], 'caption': [0, 1], 'example': [1], 'quick': [2], 'brown': [2], 'fox': [2], 'jumps': [2], 'lazy': [2], 'dog': [2], 'sunny': [2], 'day': [2]}
        self.assertEqual(map_words_to_captions(captions), expected_output)

    def test_empty_captions(self):
        captions = []
        expected_output = {}
        self.assertEqual(map_words_to_captions(captions), expected_output)

    def test_no_meaningful_words(self):
        captions = ["a an the", "is are was were"]
        expected_output = {}
        self.assertEqual(map_words_to_captions(captions), expected_output)

    def test_mixed_case_words(self):
        captions = ["The Quick Brown Fox", "the quick brown fox"]
        expected_output = {
            'quick': [0, 1],
            'brown': [0, 1],
            'fox': [0, 1]
        }
        self.assertEqual(map_words_to_captions(captions), expected_output)

if __name__ == "__main__":
    unittest.main()