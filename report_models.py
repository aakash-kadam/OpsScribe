"""Data models for report generation."""

from __future__ import annotations

from dataclasses import dataclass, field

from agents import TokenUsage


@dataclass(frozen=True)
class ReportSectionSpec:
    """One report section prompt from a report spec file."""

    id: str
    title: str
    prompt: str


@dataclass(frozen=True)
class ReportSpec:
    """Report configuration loaded from YAML."""

    title: str
    sections: list[ReportSectionSpec]


@dataclass(frozen=True)
class ReportSectionResult:
    """Generated content for one report section."""

    id: str
    title: str
    content: str
    token_usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass(frozen=True)
class ReportResult:
    """Generated report content and metadata."""

    title: str
    sections: list[ReportSectionResult]
    dataset_overview: str

    @property
    def token_usage(self) -> TokenUsage:
        input_tokens = sum(section.token_usage.input_tokens for section in self.sections)
        output_tokens = sum(section.token_usage.output_tokens for section in self.sections)
        total_tokens = sum(section.token_usage.total_tokens for section in self.sections)
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
