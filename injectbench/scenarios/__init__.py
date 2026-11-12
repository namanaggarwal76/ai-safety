"""Built-in scenario registry."""
from __future__ import annotations

from ..scenario import Scenario
from . import calendar_assistant, code_review, document_qa, email_triage, support_ticket, web_research

SCENARIOS: dict[str, Scenario] = {
    m.SCENARIO.id: m.SCENARIO
    for m in (email_triage, document_qa, support_ticket, web_research, calendar_assistant, code_review)
}
