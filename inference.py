from user_inf_simulator import UserInference
from dqn_agent import DQNAgent
from state_tracker import StateTracker
import pickle, argparse, json
from utils import remove_empty_slots


class RL_Agent():

    def __init__(self):

        # Load constants json into dict
        constants_file = 'constants.json'

        with open(constants_file) as f:
            self.constants = json.load(f)

        # Load file path constants
        file_path_dict = self.constants['db_file_paths']
        DATABASE_FILE_PATH = file_path_dict['database']

        # Load movie DB
        # Note: If you get an unpickling error here then run 'pickle_converter.py' and it should fix it
        self.database = pickle.load(open(DATABASE_FILE_PATH, 'rb'), encoding='latin1')

        # Clean DB
        remove_empty_slots(self.database)

        # user = User()
        self.user = UserInference(self.constants, self.database)
        self.state_tracker = StateTracker(self.database, self.constants)
        self.dqn_agent = DQNAgent(self.state_tracker.get_state_size(), self.constants)

        self.done = False


    def test_run(self):
        """
        Runs the loop that tests the agent.

        Tests the agent on the goal-oriented chatbot task. Only for evaluating a trained agent. Terminates when the episode
        reaches NUM_EP_TEST.

        """
        self.episode_reset()
        self.done = False
        # Get initial state from state tracker
        state = self.state_tracker.get_state()
        while not self.done:
            # Agent takes action given state tracker's representation of dialogue
            agent_action_index, agent_action = self.dqn_agent.get_action(state)
            # Update state tracker with the agent's action
            self.state_tracker.update_state_agent(agent_action)
            # User takes action given agent action
            user_action, reward, self.done, success = self.user.step(agent_action)
            # Update state tracker with user action
            self.state_tracker.update_state_user(user_action)
            # Grab "next state" as state
            state = self.state_tracker.get_state(self.done)        
            

    def episode_reset(self, goal):
        """Resets the episode/conversation in the testing loop."""

        # First reset the state tracker
        self.state_tracker.reset()
        # Then pick an init user action
        user_action = self.user.reset(goal)
        # And update state tracker
        self.state_tracker.update_state_user(user_action)
        # Finally, reset agent
        self.dqn_agent.reset()

