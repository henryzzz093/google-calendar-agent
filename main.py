from agents.base_agent import BaseAgent 
import asyncio

async def main():
    agent = BaseAgent()

    response = await agent.arun("whats todays date?")
    print(response)



if __name__ == "__main__":
      asyncio.run(main())