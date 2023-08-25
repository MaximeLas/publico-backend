from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from constants import StepID
from context import UserContext


class StepDecider(ABC):
    @abstractmethod
    def determine_next_step(self, context: UserContext) -> StepID:
        pass


@dataclass
class FixedStepDecider(StepDecider):
    next_step: StepID

    def determine_next_step(self, context: 'UserContext') -> StepID:
        return self.next_step


@dataclass
class ConditionalStepDecider(StepDecider):
    condition: Callable[['UserContext'], bool]
    if_true_step: StepID
    if_false_step: StepID

    def determine_next_step(self, context: 'UserContext') -> StepID:
        return self.if_true_step if self.condition(context) else self.if_false_step


@dataclass
class MultiConditionalStepDecider(StepDecider):
    conditional_steps: list[Callable[['UserContext'], bool], StepID]
    default_next_step: StepID

    def determine_next_step(self, context: 'UserContext') -> StepID:
        for condition, step in self.conditional_steps:
            if condition(context):
                return step
        return self.default_next_step
