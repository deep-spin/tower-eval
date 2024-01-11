import re

DET_REGEX = r"\"(?P<text>.+?)\" - (?P<severity>minor|major?)"
TAG_REGEX = r"(?P<severity><minor>|<major>)(?P<text>.+?)(</minor>|</major>)"


def tag_to_annotation(
    generation: str,
    mt: str,
) -> dict[str, str | list[dict[str, str | int]]]:
    """Converts text like: "This is an <minor>annotation<\\minor>" to a dictionary like:
    {"mt": "This is an annotation", "annotations": [{"start": 11, "end": 21, "severity": "minor"}]}
    """
    seen_tags = 0
    matches = list(re.finditer(TAG_REGEX, generation))
    annotations = []
    for m in matches:
        text = m.group("text")
        # if text is not in mt, reject annotation
        if text in mt:
            severity = m.group("severity")[1:-1]  # remove < and >
            # for every tag we have seen, there are 15 characters that we need to subtract from the start index
            start = m.start() - seen_tags * 15
            end = start + len(text)
            annotations.append(
                {"start": start, "end": end, "severity": severity, "text": text}
            )
            seen_tags += 1

    return annotations


def det_to_annotation(
    generation: str, mt: str
) -> dict[str, str | list[dict[str, str | int]]]:
    """Converts text like: "This is an <minor>annotation<\\minor>" to a dictionary like:
    {"mt": "This is an annotation", "annotations": [{"start": 11, "end": 21, "severity": "minor"}]}
    """
    annotations = []
    matches = list(re.finditer(DET_REGEX, generation))
    for m in matches:
        text = m.group("text")
        # if text is not in mt, reject annotation
        if text in mt:
            severity = m.group("severity")
            # check if flagged text is already in annotations list
            # if not, add the first match; if present n times, add the nth + 1 match
            equal_matches = list(re.finditer(re.escape(text), mt))
            i_to_append = 0
            for a in annotations:
                if a["text"] == text:
                    i_to_append += 1
            # for some reason the model flagged the same thing twice; do not append annotation
            if i_to_append >= len(equal_matches):
                continue
            match_to_append = equal_matches[i_to_append]
            start, end = match_to_append.span()
            annotations.append(
                {"start": start, "end": end, "severity": severity, "text": text}
            )

    return annotations
