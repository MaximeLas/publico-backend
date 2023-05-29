import gradio as gr
from enum import Enum
from typing import Optional

class UserInteractionType(Enum):
    YES_NO = 1
    FILES = 2
    TEXT = 3
    START = 4
    NONE = 5


class ComponentWrapper:
    
    def __init__(
        self,
        component: gr.Button | gr.Textbox | gr.Files,
        trigger_function_name: str,
        user_interaction_type: UserInteractionType,
        first_actions_after_trigger: list[dict] = []
    ):
        self.component = component
        self.trigger_to_proceed = getattr(component, trigger_function_name)
        self.user_interaction_type = user_interaction_type
        self.first_actions_after_trigger = first_actions_after_trigger
        
    def set_first_actions_after_trigger(self, first_actions_after_trigger):
        self.first_actions_after_trigger = first_actions_after_trigger

    def get_trigger_to_proceed(self):
        trigger = self.trigger_to_proceed(fn=lambda: print("triggered", type(self.component)), inputs=None, outputs=None)

        for action in self.first_actions_after_trigger:
            trigger = trigger.then(**action)

        return trigger


class ChatbotStep:
    def __init__(self, message: str, user_interaction_type: UserInteractionType):
        self.message = message
        self.user_interaction_type = user_interaction_type


chatbot_steps = [
    ChatbotStep( # 0
        message="Have you applied for this grant before?",
        user_interaction_type=UserInteractionType.YES_NO),
    ChatbotStep( # 1
        message="That's very useful! Please upload your prior grant application(s).",
        user_interaction_type=UserInteractionType.FILES),
    ChatbotStep( # 2
        message="You successfully uploaded {file_count} prior grant application(s)! 🎉",
        user_interaction_type=UserInteractionType.NONE),
    ChatbotStep( # 3
        message="Now, on to the first question! Please let me know what the first application question is, or copy and paste it from the application portal.",
        user_interaction_type=UserInteractionType.TEXT),
    ChatbotStep( # 4
        message="What is the word count?",
        user_interaction_type=UserInteractionType.TEXT)]


current_chatbot_step: Optional[ChatbotStep] = None
step = -1 # -1 because we increment it before using it


def update_current_chatbot_step():
    global step, current_chatbot_step
    step += 1
    print('Increment step to', str(step))
    current_chatbot_step = chatbot_steps[step] if step < len(chatbot_steps) else None


def is_visible_in_current_user_interaction(component_wrapper: ComponentWrapper):
    return component_wrapper.user_interaction_type is current_chatbot_step.user_interaction_type if current_chatbot_step else False


with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    
    with gr.Row():
        start_btn = ComponentWrapper(
            component=gr.Button("Start", visible=True),
            trigger_function_name='click',
            user_interaction_type=UserInteractionType.START)
        
        yes_btn = ComponentWrapper(
            component=gr.Button("Yes", visible=False),
            trigger_function_name='click',
            user_interaction_type=UserInteractionType.YES_NO)
        
        no_btn = ComponentWrapper(
            component=gr.Button("No", visible=False),
            trigger_function_name='click',
            user_interaction_type=UserInteractionType.YES_NO)
        
        user_text_box = ComponentWrapper(
            component=gr.Textbox(label="User", lines=3, visible=False, interactive=True, placeholder="Type your message here"),
            trigger_function_name='submit',
            user_interaction_type=UserInteractionType.TEXT)
        user_text_box.set_first_actions_after_trigger([{
            'fn': (lambda user_message, chat_history: ('', chat_history[:-1] + [[chat_history[-1][0], user_message]])),
            'inputs': [user_text_box.component, chatbot],
            'outputs': [user_text_box.component, chatbot]
        }])
        
        files = ComponentWrapper(
            component=gr.Files(label='Documents', visible=False, interactive=True),
            trigger_function_name='change',
            user_interaction_type=UserInteractionType.FILES)
        files.set_first_actions_after_trigger([{
            'fn': update_current_chatbot_step,
            'inputs': None,
            'outputs': None
        },{
            'fn': (lambda chat_history, files: chat_history + [[current_chatbot_step.message.format(file_count=len(files)), None]]),
            'inputs': [chatbot, files.component],
            'outputs': chatbot
        }])
        


    components = [start_btn, yes_btn, no_btn, files, user_text_box]

    for component in components:
        component.get_trigger_to_proceed().then(
            # increment step and update current chatbot message
            update_current_chatbot_step, None, None
        ).then(
            # stream chatbot message to chatbot component
            fn=lambda chat_history: chat_history + [[current_chatbot_step.message, None]],
            inputs=chatbot,
            outputs= chatbot
        ).then(
            # update visibility of components based on current chatbot step user interaction type
            fn=lambda: [gr.update(visible=is_visible_in_current_user_interaction(component)) for component in components],
            inputs=None,
            outputs=[component.component for component in components]
        )

demo.launch()