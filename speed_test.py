import asyncio

from viam.robot.client import RobotClient
from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.components.switch import Switch
from viam.components.camera import Camera
from viam.components.generic import Generic as GenericComponent
from viam.services.motion import MotionClient
from viam.services.mlmodel import MLModelClient
from viam.services.vision import VisionClient
from viam.services.generic import Generic as GenericService

async def connect():
    opts = RobotClient.Options.with_api_key(
        api_key='9mutj1rhbn3oypsmohhzoci3bqxbuw9f',
        api_key_id='eef423e7-6863-4920-9c07-a514b833bae5'
    )

    return await RobotClient.at_address('nines-photographer.g02wbvukyo.viam.cloud', opts)

async def main():
    async with await connect() as machine:
        arm = Arm.from_robot(machine, "arm")
        camera = Camera.from_robot(machine, "color-correction")
        time_start = asyncio.get_event_loop().time()
        resp = await camera.do_command(
            {
                "capture": {
                    "capture_options": {"af": False},
                    "output_dir": "Users/brad.grigsby/projects/image-processing"
                }
            }
        )
        print(f"Image captured: {resp}")
        time_end = asyncio.get_event_loop().time()
        print(f"Time taken to capture image: {time_end - time_start} seconds")

if __name__ == '__main__':
    asyncio.run(main())
        