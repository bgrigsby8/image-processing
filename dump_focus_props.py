"""
One-off probe: does this body expose native absolute focus over CrSDK?

Run from ~/focus on the machine:  python3 dump_focus_props.py

Prints every camera property whose name mentions focus, plus the module's
current focus state. If a writable focus-position property shows up, the
sony config can set `focus_emulation: "off"` and per-station focus becomes
exact: hand-focus by eye, read the position, store it.
"""

import asyncio

from viam.robot.client import RobotClient
from viam.components.camera import Camera


async def connect():
    opts = RobotClient.Options.with_api_key(
        api_key='9mutj1rhbn3oypsmohhzoci3bqxbuw9f',
        api_key_id='eef423e7-6863-4920-9c07-a514b833bae5'
    )
    return await RobotClient.at_address('nines-photographer.g02wbvukyo.viam.cloud', opts)


async def main():
    async with await connect() as machine:
        sony = Camera.from_robot(machine, "sony")

        status = await sony.do_command({"command": "get_status"})
        print(f"status: {status}\n")

        got = await sony.do_command({"command": "get_focus_position"})
        print(f"get_focus_position: {got}\n")

        resp = await sony.do_command({"command": "dump_properties"})
        props = resp.get("properties", [])
        print(f"{len(props)} properties total; focus-related:\n")
        hits = [p for p in props if "focus" in str(p).lower()]
        for p in hits:
            print(f"  {p}")
        if not hits:
            print("  (none mention focus - paste the full dump instead)")


if __name__ == '__main__':
    asyncio.run(main())
