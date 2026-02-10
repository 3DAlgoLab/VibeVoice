import asyncio
from contextlib import asynccontextmanager

async def simple_lifespan(app):
    print("Starting up - loading model...")
    print("Model loaded!")
    
    yield  # App serves requests here
    
    print("Shutting down - cleaning up...")
    print("Cleanup complete!")

class MockFastAPI:
    def __init__(self, lifespan_func):
        self.lifespan_func = lifespan_func
    
    async def run(self):
        print("=== FastAPI App Starting ===")
        async with asynccontextmanager(self.lifespan_func)(self):
            print("=== Serving Requests ===")
            print("Handling HTTP requests...")
            await asyncio.sleep(2)  # Simulate serving time
            print("More requests...")
            await asyncio.sleep(1)
        print("=== App Shutdown ===")

async def main():
    app = MockFastAPI(simple_lifespan)
    await app.run()

if __name__ == "__main__":
    asyncio.run(main())