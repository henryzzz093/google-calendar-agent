import operator
import pytz
from datetime import datetime
from typing import Annotated, List, Optional
from typing_extensions import TypedDict

from langchain_core.messages import AIMessage, ToolMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool as langchain_tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from config import settings


async def get_todays_date_and_time() -> str:
    """
    What is todays date and time in PST?

    Return:
        String contains formatted date and time message
    """
    pacific_tz = pytz.timezone("US/Pacific")
    message = datetime.now(pacific_tz).strftime("%A, %d %B %Y %I: %M %p")
    return message


class State(TypedDict):
    messages: Annotated[list, operator.add]


class BaseAgent:
    instructions = ""
    tools = [get_todays_date_and_time]
    model_name = "gpt-4o-mini"
    max_tokens = 1000
    temperature = 0.1
    openai_api_key = settings.OPENAI_API_KEY

    def __init__(self):
        self.agent_tools = self._setup_tools(self.tools)
        self.chain = self._setup_chain()
        self.workflow = self._setup_workflow()
        self.message_history = []

    def _setup_tools(self, tools):
        return [langchain_tool(t) for t in tools]

    def _setup_chain(self):
        prompt_template = ChatPromptTemplate(
            [
                ("system", self.instructions),
                MessagesPlaceholder(variable_name = "history"),
            ]
        )

        model = ChatOpenAI(
            openai_api_key = self.openai_api_key,
            model_name = self.model_name,
            max_tokens = self.max_tokens,
            temperature = self.temperature,
            model_kwargs = {"parallel_tool_calls": False},
        )
        
        model = model.bind_tools(self.agent_tools)
        return prompt_template | model

    async def _call_model(self, messages: List[BaseMessage]):
        response = await self.chain.ainvoke(messages)
        return response

    async def _call_tools(self, messages: List[BaseMessage]):
        last_message = messages[-1]
        tool_call = last_message.tool_calls[0]
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        selected_tool = next((t for t in self.agent_tools if t.name == tool_name), None)
        response = await selected_tool.ainvoke(tool_args)
        return ToolMessage(content=response, tool_call_id = tool_call['id'])


    async def _model_node(self, state: State):
        messages = state["messages"]
        response = await self._call_model(messages)
        return {"messages": [response]}

    async def _tools_node(self, state: State):
        messages = state["messages"]
        response = await self._call_tools(messages)
        return {"messages": [response]}
    
    def _decision_node(self, state):
        messages = state['messages']
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return "__end__"

    def _setup_workflow(self):
        workflow = StateGraph(State)
        workflow.add_node("model", self._model_node)
        workflow.add_node("tools", self._tools_node)
        workflow.set_entry_point("model")
        workflow.add_conditional_edges("model", self._decision_node)
        workflow.add_edge("tools", "model")
        return workflow.compile()

    async def arun(self, message: str) -> str:
        messages = self.message_history + [HumanMessage(content = message)]
        response = await self.workflow.ainvoke({"messages": messages})

        self.message_history = messages + [response["messages"][-1]]
        final_response = response['messages'][-1].content
        return final_response
