import asyncio

class SimpleLifespan:
    def __init__(self, lifespan_func):
        self.lifespan_func = lifespan_func
    
    async def __aenter__(self):
        print("=== Entering lifespan context ===")
        self.gen = self.lifespan_func()
        # Run up to the first yield
        try:
            await self.gen.__anext__()
        except StopAsyncIteration:
            raise RuntimeError("Lifespan function must have at least one yield")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("=== Exiting lifespan context ===")
        try:
            # Run after the yield (cleanup)
            await self.gen.__anext__()
        except StopAsyncIteration:
            pass  # Normal completion
        finally:
            self.gen.aclose()

class MockFastAPI:
    def __init__(self, lifespan_func):
        self.lifespan_manager = SimpleLifespan(lifespan_func)
    
    async def run(self):
        print("=== FastAPI App Starting ===")
        async with self.lifespan_manager:
            print("=== Serving Requests ===")
            print("Handling HTTP requests...")
            await asyncio.sleep(2)  # Simulate serving time
            print("More requests...")
            await asyncio.sleep(1)
        print("=== App Shutdown ===")

async def simple_lifespan():
    print("Starting up - loading model...")
    print("Model loaded!")
    
    yield  # App serves requests here
    
    print("Shutting down - cleaning up...")
    print("Cleanup complete!")

async def main():
    app = MockFastAPI(simple_lifespan)
    await app.run()

if __name__ == "__main__":
    asyncio.run(main())