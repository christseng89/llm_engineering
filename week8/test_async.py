import asyncio

async def functionA():
    print("Start A")
    await asyncio.sleep(1)
    print("End A")

async def functionB():
    print("Start B")
    await functionA()
    print("End B")

# Run the async function
asyncio.run(functionB())
