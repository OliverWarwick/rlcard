Need to consider several aspects.

We need to implement an {game}_human.py inside the agents directory, which must implement:

    @staticmethod
    def step(state)
        Used for the agent to pick an action from possible actions.
        returns - action (int): The action decided by human

    def eval_step(self, state):
        Predict the action given the current state for evaluation. The same to step here.
        (TODO:) Look into what this does in the others.)
        Returns:
            action (int): the action predicted (randomly chosen) by the random agent
            probs (list): The list of action probabilities

    def _print_state(state, raw_legal_actions, action_record):
        Specific to our agent should be the printing of the board.
        (TODO:) Unsure how this is being pulled in.