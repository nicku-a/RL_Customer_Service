from plug_and_play import NLU
from nlg_testing import NLG
from inference import RL_Agent
from deep_dialog.dialogue_config import NO_OUTCOME, usersim_intents, all_slots

class ConversationManager():

    def __init__(self):
        self.nlu = NLU()
        self.nlg = NLG()
        self.agent = RL_Agent()


    def translate_nl_to_agent(self, nl):
        nl = nl.lower()
        diaact = self.nlu.nl_to_diaact(nl)
        return diaact


    def translate_agent_to_nl(self, agent_action):
        # Translate agent -> NL
        a_act = {
            'diaact': agent_action['intent'],
            'inform_slots': agent_action['inform_slots'],
            'request_slots': agent_action['request_slots']
        }
        a_act_nl = self.nlg.diaact_to_nl(a_act)
        return a_act_nl

    
    def getSentenceCase(self, source: str):
        output = ""
        isFirstWord = True
        for c in source:
            if isFirstWord and not c.isspace():
                c = c.upper()
                isFirstWord = False
            elif not isFirstWord and c in ".!?":
                isFirstWord = True
            else:
                c = c.lower()
            output = output + c
        return output


    def start_conversation(self, user_input):
        self.information = []
        self.agent.episode_reset(goal = user_input)
        state = self.agent.state_tracker.get_state()
        # Agent takes action given state tracker's representation of dialogue
        agent_action_index, agent_action = self.agent.dqn_agent.get_action(state)
        # Update state tracker with the agent's action
        self.agent.state_tracker.update_state_agent(agent_action)
        passed, action, key, intent, success = self.agent.user.step_one(agent_action)
        return action, key, intent, success


    def process_user_input(self, action, key, intent):
        response = self.user_interaction(action, key)
        if intent == 'request':
            self.agent.user.return_request_response(response, key)
        elif intent == 'inform':
            self.agent.user.return_inform_response(response, key)


    def process_agent_output(self, action, key, intent):
        if intent == 'inform':
            self.information.append(action['inform_slots'][key])


    def conversation_step(self, success):
        passed = True
        while passed:
            user_action, reward, self.agent.done = self.agent.user.step_two(success)
            # Update state tracker with user action
            self.agent.state_tracker.update_state_user(user_action)
            state = self.agent.state_tracker.get_state()
            # Agent takes action given state tracker's representation of dialogue
            agent_action_index, agent_action = self.agent.dqn_agent.get_action(state)
            # Update state tracker with the agent's action
            self.agent.state_tracker.update_state_agent(agent_action)
            # User takes action given agent action
            passed, action, key, intent, success = self.agent.user.step_one(agent_action)
        return action, key, intent, success


    def user_interaction(self, agent_action=None, key=None):
        if key == 'ticket':
            ticket = self.agent.user.state['history_slots'][key]
            info_list = ', '.join(str(item) for item in self.information)
            output_dialogue = "Booking confirmed: # {}. {}.".format(ticket, info_list)
        elif agent_action != None:
            output_dialogue = self.translate_agent_to_nl(agent_action)
        else:
            output_dialogue = 'Hello! How can I help?'
        print('Lambastard: ' + self.getSentenceCase(output_dialogue))

        response = {'intent': '', 'inform_slots': {}, 'request_slots': {}}

        while True:
            input_string = input("User:       ")
            diaact = self.translate_nl_to_agent(input_string)

            chunks = diaact.split('/')

            intent_correct = True
            if chunks[0] not in usersim_intents:
                intent_correct = False
            response['intent'] = chunks[0]

            informs_correct = True
            if len(chunks[1]) > 0:
                informs_items_list = chunks[1].split(', ')
                for inf in informs_items_list:
                    inf = inf.split(': ')
                    if inf[0] not in all_slots:
                        informs_correct = False
                        break
                    response['inform_slots'][inf[0]] = inf[1]

            requests_correct = True
            if len(chunks[2]) > 0:
                requests_key_list = chunks[2].split(', ')
                for req in requests_key_list:
                    if req not in all_slots:
                        requests_correct = False
                        break
                    response['request_slots'][req] = 'UNK'

            if intent_correct and informs_correct and requests_correct:
                break

        if agent_action != None and key != None:
            if agent_action['intent'] == 'inform' and key not in response['inform_slots'].keys():
                response['inform_slots'][key] = agent_action['inform_slots'][key]

        return response



if __name__ == "__main__":

    CM = ConversationManager()


    # FIRST STEP OF CONVERSATION
    ################################
    goal = CM.user_interaction(None)  # Interacting in the cmd line
    ################################
    agent_action, key, intent, success = CM.start_conversation(goal)
    CM.process_user_input(agent_action, key, intent)

    # NOW GO BACK AND FORTH USER <-> AGENT IN LOOP
    while intent != 'done':
        agent_action, key, intent, success = CM.conversation_step(success)
        CM.process_agent_output(agent_action, key, intent)
        CM.process_user_input(agent_action, key, intent)
