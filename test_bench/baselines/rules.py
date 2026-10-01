"""Context rules of the prototype, now part of the engine (re-exported for older imports).

See ``anonymizer.engine.context`` (labels, form cells, table columns), ``anonymizer.engine.names``
(given-name dictionary, list-name expansion) and ``anonymizer.engine.patterns`` (URLs, amounts).
"""

from anonymizer.engine.context import (  # noqa: F401
    LABEL_ONLY,
    LABEL_VALUE,
    Line,
    context_rules,
)
from anonymizer.engine.names import GIVEN_NAMES, expand_name, names_by_dictionary  # noqa: F401
from anonymizer.engine.patterns import PERSONAL_URL, complete_url, is_amount  # noqa: F401
from anonymizer.engine.patterns import norm as _norm  # noqa: F401
