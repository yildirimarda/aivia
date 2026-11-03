"""The verification stage: a topic needs two independent source domains.

Selection must not promote a topic that only one publisher reported. This
stage inspects the source URLs already attached to each selected topic and
groups them by registrable domain, so two pages on the same site - including
two language editions of one wiki - count as a single source. A topic with
sources from at least two independent domains is marked ``verified``; a topic
with zero or one domain is marked ``unverified`` and the reason is recorded.

The stage never fetches a page and never calls a provider: it verifies the
provenance a topic already carries. It is deterministic and offline, so the
same candidates always produce the same decisions.
"""

from __future__ import annotations

import ipaddress
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from urllib.parse import urlsplit

from aivia.core.errors import ConfigError
from aivia.core.models.topic import Topic, TopicCandidate
from aivia.providers.protocols.source import SourceResponse

__all__ = [
    "DEFAULT_MIN_DOMAINS",
    "IndependentDomain",
    "TopicVerifier",
    "VerificationDecision",
    "VerificationResult",
    "VerificationStage",
    "VerificationStatus",
    "VerifiedTopic",
    "Verifier",
    "VerifyResult",
    "VerifyStage",
    "build_verification_stage",
    "domain_of",
    "independent_domains",
    "is_verified",
    "mark_verified",
    "registrable_domain",
    "verification",
    "verification_stage",
    "verify",
    "verify_candidate",
    "verify_candidates",
    "verify_topics",
]

DEFAULT_MIN_DOMAINS = 2
"""A topic needs sources from at least this many independent domains."""

VerifiableTopic = Topic | TopicCandidate
"""A topic record carrying source URLs and a verification flag."""

VerifiedTopic = VerifiableTopic
"""Alias for :data:`VerifiableTopic`."""

IndependentDomain = str
"""One registrable domain that independently reported a topic."""

_SECOND_LEVEL_SUFFIXES = frozenset(
    {
        "ac.in",
        "ac.jp",
        "ac.nz",
        "ac.uk",
        "asn.au",
        "co.id",
        "co.il",
        "co.in",
        "co.jp",
        "co.ke",
        "co.kr",
        "co.mz",
        "co.nz",
        "co.th",
        "co.uk",
        "co.za",
        "com.ar",
        "com.au",
        "com.bh",
        "com.bo",
        "com.br",
        "com.cn",
        "com.co",
        "com.do",
        "com.ec",
        "com.eg",
        "com.gt",
        "com.hk",
        "com.hn",
        "com.kw",
        "com.mx",
        "com.ng",
        "com.ni",
        "com.om",
        "com.pa",
        "com.pe",
        "com.pr",
        "com.py",
        "com.qa",
        "com.sa",
        "com.sg",
        "com.sv",
        "com.tr",
        "com.tw",
        "com.uy",
        "com.ve",
        "edu.au",
        "edu.br",
        "edu.mx",
        "firm.in",
        "gen.in",
        "go.jp",
        "gob.mx",
        "gov.au",
        "gov.br",
        "gov.in",
        "gov.uk",
        "govt.nz",
        "gr.jp",
        "id.au",
        "ind.in",
        "lg.jp",
        "ltd.uk",
        "me.uk",
        "ne.jp",
        "net.au",
        "net.br",
        "net.in",
        "net.mx",
        "net.uk",
        "nhs.uk",
        "or.jp",
        "org.au",
        "org.br",
        "org.in",
        "org.mx",
        "org.nz",
        "org.uk",
        "plc.uk",
        "police.uk",
        "sch.uk",
        "school.nz",
    }
)


class VerificationStatus(StrEnum):
    """The verification state of one topic."""

    VERIFIED = "verified"
    UNVERIFIED = "unverified"

    @property
    def verified(self) -> bool:
        """Whether this status marks the topic as verified."""
        return self is VerificationStatus.VERIFIED


def _validate_min_domains(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError("minimum domains must be an integer")
    if value < DEFAULT_MIN_DOMAINS:
        raise ConfigError(f"minimum domains must be at least {DEFAULT_MIN_DOMAINS}")
    return value


def registrable_domain(host: str) -> str:
    """Return the registrable domain for *host*.

    ``en.wikipedia.org`` and ``de.wikipedia.org`` both become
    ``wikipedia.org``; ``news.example.co.uk`` becomes ``example.co.uk``. IP
    addresses are returned unchanged. The helper is deliberately small and
    offline: it recognises the common multi-label public suffixes the research
    sources use and otherwise keeps the last two labels.
    """
    if not isinstance(host, str):
        raise ConfigError("host must be a string")
    value = host.strip().lower().rstrip(".")
    if not value:
        return ""
    if value.startswith("www."):
        value = value[4:]
    try:
        ipaddress.ip_address(value)
    except ValueError:
        pass
    else:
        return value
    labels = [label for label in value.split(".") if label]
    if len(labels) <= 2:
        return ".".join(labels)
    suffix = ".".join(labels[-2:])
    if suffix in _SECOND_LEVEL_SUFFIXES:
        return ".".join(labels[-3:])
    return suffix


domain_of = registrable_domain
"""Alias for :func:`registrable_domain`."""


def _host_of(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def independent_domains(candidate: VerifiableTopic) -> list[str]:
    """Return the distinct registrable domains behind *candidate*'s sources."""
    domains: list[str] = []
    for url in candidate.source_urls:
        domain = registrable_domain(_host_of(url))
        if domain and domain not in domains:
            domains.append(domain)
    return domains


def _coerce_candidate(value: Any) -> VerifiableTopic:
    if isinstance(value, (Topic, TopicCandidate)):
        return value
    if isinstance(value, str):
        return TopicCandidate(title=value)
    if isinstance(value, Mapping):
        return TopicCandidate.model_validate(dict(value))
    raise ConfigError("candidate must be a topic, a candidate or a mapping")


def _coerce_candidates(value: Any) -> list[VerifiableTopic]:
    if value is None:
        return []
    if isinstance(value, (Topic, TopicCandidate, str)):
        return [_coerce_candidate(value)]
    if isinstance(value, SourceResponse):
        return list(value.candidates)
    if isinstance(value, Mapping):
        nested = value.get("candidates", value.get("topics"))
        if nested is not None and not any(
            key in value for key in ("title", "name", "topic", "subject")
        ):
            return _coerce_candidates(nested)
        return [_coerce_candidate(value)]
    if hasattr(value, "candidates") and not isinstance(value, (bytes, bytearray)):
        return _coerce_candidates(value.candidates)
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        candidates: list[VerifiableTopic] = []
        for item in value:
            candidates.extend(_coerce_candidates(item))
        return candidates
    raise ConfigError("candidates must be a topic or an iterable of topics")


def _as_candidate(value: VerifiableTopic) -> TopicCandidate:
    if isinstance(value, TopicCandidate):
        return value
    return TopicCandidate.model_validate(
        value.model_dump(exclude={"claims", "candidate"})
    )


@dataclass(frozen=True, slots=True)
class VerificationDecision:
    """One topic, its independent domains and its verification status."""

    candidate: VerifiableTopic
    domains: tuple[str, ...] = ()
    source_count: int = 0
    min_domains: int = DEFAULT_MIN_DOMAINS
    reason: str = ""

    @property
    def verified(self) -> bool:
        """Whether the topic cleared the independent-domain requirement."""
        return len(self.domains) >= self.min_domains

    @property
    def unverified(self) -> bool:
        """Whether the topic failed the independent-domain requirement."""
        return not self.verified

    @property
    def single_source(self) -> bool:
        """Whether the topic has exactly one independent domain."""
        return len(self.domains) == 1

    @property
    def domain_count(self) -> int:
        """The number of independent domains behind the topic."""
        return len(self.domains)

    @property
    def status(self) -> VerificationStatus:
        """The verification state as an enum."""
        if self.verified:
            return VerificationStatus.VERIFIED
        return VerificationStatus.UNVERIFIED

    @property
    def title(self) -> str:
        """The topic title, for convenient reporting."""
        return self.candidate.title

    @property
    def urls(self) -> list[str]:
        """The source URLs inspected for this decision."""
        return list(self.candidate.source_urls)

    @property
    def independent_domains(self) -> list[str]:
        """The independent domains as a list."""
        return list(self.domains)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the decision."""
        return {
            "title": self.candidate.title,
            "verified": self.verified,
            "unverified": self.unverified,
            "single_source": self.single_source,
            "status": self.status.value,
            "domains": list(self.domains),
            "domain_count": self.domain_count,
            "source_count": self.source_count,
            "min_domains": self.min_domains,
            "reason": self.reason,
            "urls": self.urls,
            "candidate": self.candidate.model_dump(mode="json"),
        }


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Every decision plus the verified and unverified topic subsets."""

    decisions: tuple[VerificationDecision, ...] = ()
    min_domains: int = DEFAULT_MIN_DOMAINS
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def candidates(self) -> list[VerifiableTopic]:
        """Every inspected topic, in input order, with its flag updated."""
        return [decision.candidate for decision in self.decisions]

    @property
    def topics(self) -> list[VerifiableTopic]:
        """Alias for :attr:`candidates`."""
        return self.candidates

    @property
    def all_candidates(self) -> list[VerifiableTopic]:
        """Alias for :attr:`candidates`."""
        return self.candidates

    @property
    def verified_decisions(self) -> list[VerificationDecision]:
        """The decisions that cleared the domain requirement."""
        return [decision for decision in self.decisions if decision.verified]

    @property
    def unverified_decisions(self) -> list[VerificationDecision]:
        """The decisions that failed the domain requirement."""
        return [decision for decision in self.decisions if not decision.verified]

    @property
    def verified(self) -> list[VerifiableTopic]:
        """The topics marked verified."""
        return [decision.candidate for decision in self.verified_decisions]

    @property
    def unverified(self) -> list[VerifiableTopic]:
        """The topics marked unverified."""
        return [decision.candidate for decision in self.unverified_decisions]

    @property
    def verified_candidates(self) -> list[VerifiableTopic]:
        """Alias for :attr:`verified`."""
        return self.verified

    @property
    def unverified_candidates(self) -> list[VerifiableTopic]:
        """Alias for :attr:`unverified`."""
        return self.unverified

    @property
    def single_source(self) -> list[VerifiableTopic]:
        """The topics with exactly one independent domain."""
        return [
            decision.candidate for decision in self.decisions if decision.single_source
        ]

    @property
    def single_source_candidates(self) -> list[VerifiableTopic]:
        """Alias for :attr:`single_source`."""
        return self.single_source

    @property
    def titles(self) -> list[str]:
        """The titles of every inspected topic."""
        return [decision.title for decision in self.decisions]

    @property
    def verified_titles(self) -> list[str]:
        """The titles of the verified topics."""
        return [decision.title for decision in self.verified_decisions]

    @property
    def unverified_titles(self) -> list[str]:
        """The titles of the unverified topics."""
        return [decision.title for decision in self.unverified_decisions]

    @property
    def count(self) -> int:
        """The number of inspected topics."""
        return len(self.decisions)

    @property
    def input_count(self) -> int:
        """Alias for :attr:`count`."""
        return self.count

    @property
    def verified_count(self) -> int:
        """The number of verified topics."""
        return len(self.verified_decisions)

    @property
    def unverified_count(self) -> int:
        """The number of unverified topics."""
        return len(self.unverified_decisions)

    @property
    def single_source_count(self) -> int:
        """The number of topics with exactly one independent domain."""
        return sum(1 for decision in self.decisions if decision.single_source)

    def filter(self, *, verified_only: bool = True) -> list[VerifiableTopic]:
        """Return the verified topics, or every topic when not filtering."""
        return self.verified if verified_only else self.candidates

    def to_response(
        self,
        *,
        provider: str = "verify",
        query: str = "",
        verified_only: bool = False,
    ) -> SourceResponse:
        """Return the inspected topics as a :class:`SourceResponse`."""
        return SourceResponse(
            provider=provider,
            query=query,
            candidates=[
                _as_candidate(candidate)
                for candidate in self.filter(verified_only=verified_only)
            ],
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the stage."""
        return {
            "decisions": [decision.to_dict() for decision in self.decisions],
            "verified": [decision.title for decision in self.verified_decisions],
            "unverified": [decision.title for decision in self.unverified_decisions],
            "metadata": dict(self.metadata),
        }

    def __iter__(self) -> Iterator[VerifiableTopic]:
        return iter(self.candidates)

    def __len__(self) -> int:
        return len(self.decisions)


VerifyResult = VerificationResult
"""Alias for :class:`VerificationResult`."""


def verify_candidate(
    candidate: Any,
    *,
    min_domains: int = DEFAULT_MIN_DOMAINS,
) -> VerificationDecision:
    """Verify one topic against independent source domains and mark it."""
    floor = _validate_min_domains(min_domains)
    topic = _coerce_candidate(candidate)
    domains = tuple(independent_domains(topic))
    if not topic.source_urls:
        reason = "no source urls to verify"
    elif len(domains) >= floor:
        reason = f"verified across {len(domains)} independent domains"
    elif len(domains) == 1:
        reason = f"single source domain: {domains[0]}"
    else:
        reason = f"only {len(domains)} independent domains; need {floor}"
    topic.verified = len(domains) >= floor
    return VerificationDecision(
        candidate=topic,
        domains=domains,
        source_count=len(topic.source_urls),
        min_domains=floor,
        reason=reason,
    )


def verify_candidates(
    candidates: Any,
    *,
    min_domains: int = DEFAULT_MIN_DOMAINS,
    metadata: Mapping[str, Any] | None = None,
) -> VerificationResult:
    """Verify every candidate and mark single source topics unverified."""
    floor = _validate_min_domains(min_domains)
    values = _coerce_candidates(candidates)
    decisions = tuple(
        verify_candidate(candidate, min_domains=floor) for candidate in values
    )
    result_metadata = {
        "min_domains": floor,
        "input_count": len(decisions),
        "verified": sum(1 for decision in decisions if decision.verified),
        "unverified": sum(1 for decision in decisions if not decision.verified),
        "single_source": sum(1 for decision in decisions if decision.single_source),
    }
    if metadata:
        result_metadata.update(metadata)
    return VerificationResult(
        decisions=decisions,
        min_domains=floor,
        metadata=result_metadata,
    )


verify_topics = verify_candidates
"""Alias for :func:`verify_candidates`."""

verify = verify_candidates
"""Alias for :func:`verify_candidates`."""

verification = verify_candidates
"""Alias for :func:`verify_candidates`."""


def is_verified(
    candidate: Any,
    *,
    min_domains: int = DEFAULT_MIN_DOMAINS,
) -> bool:
    """Whether *candidate* has sources from enough independent domains."""
    floor = _validate_min_domains(min_domains)
    return len(independent_domains(_coerce_candidate(candidate))) >= floor


def mark_verified(
    candidate: Any,
    *,
    min_domains: int = DEFAULT_MIN_DOMAINS,
) -> VerifiableTopic:
    """Verify *candidate* and return it with its ``verified`` flag updated."""
    return verify_candidate(candidate, min_domains=min_domains).candidate


class VerificationStage:
    """A reusable verification stage bound to one minimum domain count."""

    def __init__(
        self,
        min_domains: int = DEFAULT_MIN_DOMAINS,
        *,
        name: str = "verify",
    ) -> None:
        self._min_domains = _validate_min_domains(min_domains)
        self._name = name.strip() or "verify"

    @property
    def name(self) -> str:
        """The provider name reported by :meth:`to_response`."""
        return self._name

    @property
    def min_domains(self) -> int:
        """The number of independent domains a topic must have."""
        return self._min_domains

    def verify(
        self,
        candidates: Any,
        *,
        min_domains: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> VerificationResult:
        """Verify *candidates* using this stage's configuration."""
        floor = self._min_domains if min_domains is None else min_domains
        return verify_candidates(
            candidates,
            min_domains=floor,
            metadata=metadata,
        )

    run = verify
    __call__ = verify

    async def run_async(
        self,
        candidates: Any,
        *,
        min_domains: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> VerificationResult:
        """Async variant; verification itself never awaits anything."""
        return self.verify(
            candidates,
            min_domains=min_domains,
            metadata=metadata,
        )

    verify_async = run_async

    def filter(self, candidates: Any) -> list[VerifiableTopic]:
        """Return only the candidates that clear the domain requirement."""
        return self.verify(candidates).verified

    def to_response(
        self,
        candidates: Any,
        *,
        query: str = "",
        verified_only: bool = False,
    ) -> SourceResponse:
        """Verify *candidates* and return them as a response."""
        return self.verify(candidates).to_response(
            provider=self._name,
            query=query,
            verified_only=verified_only,
        )


Verifier = VerificationStage
"""Alias for :class:`VerificationStage`."""

TopicVerifier = VerificationStage
"""Alias for :class:`VerificationStage`."""

VerifyStage = VerificationStage
"""Alias for :class:`VerificationStage`."""

verification_stage = VerificationStage
"""Alias for :class:`VerificationStage`."""


def build_verification_stage(
    min_domains: int = DEFAULT_MIN_DOMAINS,
    *,
    name: str = "verify",
) -> VerificationStage:
    """Build a :class:`VerificationStage`."""
    return VerificationStage(min_domains, name=name)
