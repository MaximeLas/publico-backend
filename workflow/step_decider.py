from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from configurations.constants import StepID
from workflow.session_state import SessionState



class StepDecider(ABC):
    @abstractmethod
    def determine_next_step(self, state: SessionState) -> StepID:
        pass


@dataclass
class FixedStepDecider(StepDecider):
    next_step: StepID

    def determine_next_step(self, state: SessionState) -> StepID:
        return self.next_step


@dataclass
class ConditionalStepDecider(StepDecider):
    condition: Callable[['SessionState'], bool]
    if_true_step: StepID
    if_false_step: StepID

    def determine_next_step(self, state: SessionState) -> StepID:
        return self.if_true_step if self.condition(state) else self.if_false_step


@dataclass
class MultiConditionalStepDecider(StepDecider):
    conditional_steps: list[Callable[[SessionState], bool], StepID]
    default_next_step: StepID

    def determine_next_step(self, state: SessionState) -> StepID:
        for condition, step in self.conditional_steps:
            if condition(state):
                return step
        return self.default_next_step
