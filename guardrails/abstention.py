"""Rules for forcing the existing abstain path after failed output checks."""


def should_force_abstain(
    sufficient: bool | None,
    groundedness_passed: bool,
    *,
    enabled: bool = True,
) -> bool:
    return enabled and sufficient is False and not groundedness_passed