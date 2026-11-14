"""Data-free fixed horizons and observation points for LM comparisons."""

from dataclasses import dataclass

TOKENS_PER_STEP = 1024


@dataclass(frozen=True)
class TrainingPlan:
    tokens: int = 999424
    schedule_tokens: int = 20000000
    topology_tokens: int | None = None

    def __post_init__(self):
        if self.topology_tokens is None:
            object.__setattr__(self, "topology_tokens", self.tokens)
        if any(
            type(v) is not int
            for v in (self.tokens, self.schedule_tokens, self.topology_tokens)
        ):
            raise ValueError("integer token horizons required")
        if (
            not TOKENS_PER_STEP
            <= self.tokens
            <= min(self.schedule_tokens, self.topology_tokens)
            or max(self.schedule_tokens, self.topology_tokens) > 20000000
        ):
            raise ValueError("positive bounded run within both fixed horizons required")

    @property
    def steps(self):
        return self.tokens // TOKENS_PER_STEP

    @property
    def schedule_steps(self):
        return self.schedule_tokens // TOKENS_PER_STEP

    @property
    def topology_steps(self):
        return self.topology_tokens // TOKENS_PER_STEP

    @property
    def planned_events(self):
        return tuple(range(250, int(0.75 * self.topology_steps), 250))

    @property
    def events(self):
        return tuple(k for k in self.planned_events if k <= self.steps)

    @property
    def milestones(self):
        return tuple(
            sorted(
                {
                    256,
                    self.steps,
                    *(
                        n // TOKENS_PER_STEP
                        for n in (1000000, 5000000, 10000000, 20000000)
                    ),
                }
                & set(range(1, self.steps + 1))
            )
        )

    def adam_updates(self, step):
        if type(step) is not int or not 0 <= step <= self.steps:
            raise ValueError("training step outside frozen budget")
        return step - sum(k <= step for k in self.events)
