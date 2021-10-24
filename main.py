import asyncio
from color_processing import color_process_temperature
from test import test

async def main():
    await color_process_temperature().get_data()
    # await test().get_data()

if __name__ == "__main__":
    asyncio.run(main())