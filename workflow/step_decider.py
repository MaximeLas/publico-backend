from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from configurations.constants import StepID
from workflow.app_context import AppContext



class StepDecider(ABC):
    @abstractmethod
    def determine_next_step(self, context: AppContext) -> StepID:
        pass


@dataclass
class FixedStepDecider(StepDecider):
    next_step: StepID

    def determine_next_step(self, context: 'AppContext') -> StepID:
        return self.next_step


@dataclass
class ConditionalStepDecider(StepDecider):
    condition: Callable[['AppContext'], bool]
    if_true_step: StepID
    if_false_step: StepID

    def determine_next_step(self, context: 'AppContext') -> StepID:
        return self.if_true_step if self.condition(context) else self.if_false_step


@dataclass
class MultiConditionalStepDecider(StepDecider):
    conditional_steps: list[Callable[['AppContext'], bool], StepID]
    default_next_step: StepID

    def determine_next_step(self, context: 'AppContext') -> StepID:
        for condition, step in self.conditional_steps:
            if condition(context):
                return step
        return self.default_next_step
