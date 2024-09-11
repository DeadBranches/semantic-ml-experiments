HOST = 'localhost:5000'
URI = f'http://{HOST}/api/v1/generate'

class AlpacaLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if isinstance(stop, list):
            stop = stop + ["\n###","\nObservation:"]

        response = requests.post(
            URI,
            json={
                "prompt": prompt,
                "temperature": 0.7,
                "max_new_tokens": 256,
                "early_stopping": True,
                "stopping_strings": stop,
                'do_sample': True,
                'top_p': 0.1,
                'typical_p': 1,
                'repetition_penalty': 1.18,
                'top_k': 40,
                'min_length': 0,
                'no_repeat_ngram_size': 0,
                'num_beams': 1,
                'penalty_alpha': 0,
                'length_penalty': 1,
                'seed': -1,
                'add_bos_token': True,
                'truncation_length': 2048,
                'ban_eos_token': False,
                'skip_special_tokens': True,
            },
        )
        response.raise_for_status()
        return response.json()['results'][0]['text']
  
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}

llm = AlpacaLLM()


from langchain.utilities import GoogleSerperAPIWrapper
import os
os.environ["SERPER_API_KEY"] = 'YOUR_API_KEY'

search = GoogleSerperAPIWrapper()
tool_google = Tool(
        name="Google Search",
        func=search.run,
        description="useful for when you need to ask with search"
    )

tool_names = [tool_google]


template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Answer the following questions as best you can. You have access to the following tools:

Google Search: A wrapper around Google Search. Useful for when you need to answer questions about current events. The input is the question to search relavant information.

Strictly use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [Google Search]
Action Input: the input to the action, should be a question.
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

For examples:
Question: How old is CEO of Microsoft wife?
Thought: First, I need to find who is the CEO of Microsoft.
Action: Google Search
Action Input: Who is the CEO of Microsoft?
Observation: Satya Nadella is the CEO of Microsoft.
Thought: Now, I should find out Satya Nadella's wife.
Action: Google Search
Action Input: Who is Satya Nadella's wife?
Observation: Satya Nadella's wife's name is Anupama Nadella.
Thought: Then, I need to check Anupama Nadella's age.
Action: Google Search
Action Input: How old is Anupama Nadella?
Observation: Anupama Nadella's age is 50.
Thought: I now know the final answer.
Final Answer: Anupama Nadella is 50 years old.

### Input:
{input}

### Response:
{agent_scratchpad}"""

temp_Ins = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Question: {thought}
Query: {query}
Observation: {observation}

### Input:
Make a short summary of useful information from the result observation that is related to the question.

### Response:"""

prompt_Ins = PromptTemplate(
    input_variables=["thought", "query", "observation"],
    template=temp_Ins,
)

class CustomPromptTemplate(StringPromptTemplate):

    input_variables: List[str]
    """A list of the names of the variables the prompt template expects."""

    template: str
    """The prompt template."""

    template_format: str = "f-string"
    """The format of the prompt template. Options are: 'f-string', 'jinja2'."""

    validate_template: bool = False
    """Whether or not to try validating the template."""

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        # Refine the observation
        if len(intermediate_steps) > 0:
            regex = r"Thought\s*\d*\s*:(.*?)\nAction\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)\nObservation"
            text_match = intermediate_steps[-1][0].log
            if len(intermediate_steps) > 1:
                text_match = 'Thought: ' + text_match
            match = re.search(regex, text_match, re.DOTALL)           
            my_list = list(intermediate_steps[-1])
            p_INS_temp = prompt_Ins.format(thought=match.group(1).strip(), query=match.group(3).strip(), observation=my_list[1])
            my_list[1] = llm(p_INS_temp)
            my_tuple = tuple(my_list)            
            intermediate_steps[-1] = my_tuple
            
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f" {observation}\nThought:"
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        return self.template.format(**kwargs)
    
    
prompt = CustomPromptTemplate(input_variables=["input", "intermediate_steps"],
                              template=template,validate_template=False)

                              agent = LLMSingleActionAgent(
    llm_chain=llm_chain, 
    output_parser=output_parser,
    stop=["\nObservation:"], 
    allowed_tools=tool_names
)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tool_names, verbose=True)

agent_executor.run("Is Eminem an actor?")