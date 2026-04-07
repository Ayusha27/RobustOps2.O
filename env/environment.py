from env.models import Observation, Action, Reward
from env.noise import inject_noise
from env.tasks.task_easy import get_task as easy_task
from env.tasks.task_medium import get_task as medium_task
from env.tasks.task_hard import get_task as hard_task
from env.tasks.graders import grade_task

class RobustOpsEnv:
    def __init__(self):
        self.current_step = 0
        self.done = False
        self.correct_label = "spam"
        self.agent_decision = None

    def reset(self):
        self.current_step = 0
        self.done = False
        self.agent_decision = None

        self.signals = inject_noise()

        task = easy_task()  # keep simple for now

        self.correct_label = task["correct_label"]
        self.task_id = task["task_id"]

        return Observation(
            task_id=self.task_id,
            step=self.current_step,
            message=f"{task['description']} | Signals: {self.signals}"
        )

    def step(self, action: Action):
        self.current_step += 1
        reward = 0.0

        if action.action_type == "classify":
            self.agent_decision = action.content

            if action.content == self.correct_label:
                reward = 0.5
            else:
                reward = -0.2

        elif action.action_type == "revise":
            if self.agent_decision != self.correct_label and action.content == self.correct_label:
                reward = 1.0
                self.done = True
            else:
                reward = -0.3

            self.agent_decision = action.content

        elif action.action_type == "flag_uncertain":
            reward = 0.2

        # termination logic
        if self.agent_decision == self.correct_label:
            self.done = True
        elif self.current_step >= 2:
            self.done = True

        # grading
        if self.done:
            final_score = grade_task(self.agent_decision, self.correct_label)
        else:
            final_score = 0.0

        observation = Observation(
            task_id=self.task_id,
            step=self.current_step,
            message=f"Updated signals: {self.signals}. You may revise your decision."
        )

        return observation, Reward(value=reward), self.done, {"score": final_score}


    def state(self):
        return {
            "step": self.current_step,
            "decision": self.agent_decision,
            "done": self.done
        }