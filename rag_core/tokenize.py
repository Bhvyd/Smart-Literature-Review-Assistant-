
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")

_STOPWORD_TEXT = """
a about above after again against all also am an and any are as at be because been before being below
between both but by can cannot could did do does doing down during each either few for from further
had has have having he her here hers herself him himself his how however i if in into is it its itself
just me might more most must my myself no nor not now of off on once only or other our ours ourselves
out over own per same shall she should since so some such than that the their theirs them themselves
then there these they this those through thus to too under until up upon very via was we were what when
where whether which while who whom why will with within without would you your yours yourself yourselves
dont doesnt didnt cant couldnt wouldnt shouldnt wont isnt arent wasnt werent havent hasnt hadnt
youre youve youll youd theyre theyve theyll theyd weve ive im thats whats theres heres
"""

STOPWORDS: frozenset[str] = frozenset(_STOPWORD_TEXT.split())


def normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.casefold().replace("'", "").replace("\u2019", "")


def tokenize(
    text: str,
    *,
    drop_stopwords: bool = True,
    min_length: int = 2,
) -> list[str]:
    if not text:
        return []

    tokens: Iterable[str] = _TOKEN_RE.findall(normalize(text))

    if min_length > 1:
        tokens = [token for token in tokens if len(token) >= min_length]

    if drop_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]

    return list(tokens)
