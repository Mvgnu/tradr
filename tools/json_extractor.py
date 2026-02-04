import json
from typing import List, Any


class JsonExtractor:
    """
    A robust parser to find and extract all valid JSON objects and arrays
    embedded within a larger string of free-form text.
    """

    def extract(self, text: str) -> List[Any]:
        """
        Finds and parses all top-level JSON objects and arrays in a string.

        Args:
            text: The string to search through.

        Returns:
            A list of parsed JSON objects/arrays found in the text.
        """
        json_objects = []
        search_start_index = 0
        while search_start_index < len(text):
            first_brace = text.find('{', search_start_index)
            first_bracket = text.find('[', search_start_index)
            if first_brace == -1 and first_bracket == -1:
                break
            if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
                start_index = first_brace
            else:
                start_index = first_bracket
            end_index = self._find_matching_brace(text, start_index)
            if end_index != -1:
                potential_json = text[start_index:end_index + 1]
                try:
                    parsed = json.loads(potential_json)
                    json_objects.append(parsed)
                    search_start_index = end_index + 1
                except json.JSONDecodeError:
                    search_start_index = start_index + 1
            else:
                search_start_index = start_index + 1
        return json_objects

    def _find_matching_brace(self, text: str, start_index: int) -> int:
        """
        Given a string and the starting index of an opening brace '{' or
        bracket '[', finds the index of its corresponding closing brace/bracket.
        Correctly handles nested structures and quoted strings.
        """
        brace_stack = []
        is_in_string = False
        is_escaped = False

        opening_char = text[start_index]
        closing_char = '}' if opening_char == '{' else ']'

        for i in range(start_index, len(text)):
            char = text[i]

            if is_in_string:
                if is_escaped:
                    is_escaped = False
                elif char == '\\':
                    is_escaped = True
                elif char == '"':
                    is_in_string = False
            else:
                if char == '"':
                    is_in_string = True
                elif char == opening_char:
                    brace_stack.append(char)
                elif char == closing_char:
                    if not brace_stack:
                        return -1
                    brace_stack.pop()
                    if not brace_stack:
                        return i

        return -1  # No matching brace found 